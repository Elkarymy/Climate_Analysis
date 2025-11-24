# epw_app.py
# EPW Viewer – Wind & Climate dashboard + fancy PDF report
# - Wind roses (annual + monthly grid, interactive downloads)
# - Temperature monthly min/max (dry-bulb & dew-point)
# - Humidity monthly min/max
# - Radiation monthly totals: DHI + DNI stacked bars + GHI line
# - Summary metrics + monthly table
# - PDF report with styled pages and vector/PNG charts
# - Caching: PDF stored the first time and reused subsequently

import json
import os
import hashlib
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

# ----- PDF deps -----
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch  # noqa: F401 (reserved if you want to tweak)
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from PIL import Image  # noqa: F401 (ImageReader relies on PIL under the hood)

# Optional crisp vector rendering for charts in the PDF
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    _SVG_OK = True
except Exception:
    _SVG_OK = False


# ---------------- Page / constants ----------------
st.set_page_config(page_title="EPW Viewer – Wind & Climate", layout="wide")

# Wind-speed bins and colors
SPEED_LABELS = ["<0.5 m/s", "0.5–2 m/s", "2–4 m/s", "4–6 m/s", "6–8 m/s", "8–10 m/s", ">10 m/s"]
CUSTOM_COLORS = {
    "<0.5 m/s": "#00bfff",  # light blue
    "0.5–2 m/s": "#00ff00", # green
    "2–4 m/s": "#ffff00",   # yellow
    "4–6 m/s": "#ff8000",   # orange
    "6–8 m/s": "#404080",   # dark blue
    "8–10 m/s": "#800080",  # purple
    ">10 m/s": "#00ffff"    # cyan
}

DIR_LABELS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
              'S','SSW','SW','WSW','W','WNW','NW','NNW']
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MAP = {i+1: m for i, m in enumerate(MONTH_NAMES)}

# Radiation names/colors
NAME_MAP = {
    "GHI": "Global Horizontal Radiation",
    "DNI": "Direct Normal Radiation",
    "DHI": "Diffuse Horizontal Radiation",
}
COLOR_MAP_RAD = {
    "Global Horizontal Radiation": "#fdb45c",
    "Direct Normal Radiation":     "#ff9f40",
    "Diffuse Horizontal Radiation": "#ff66cc",
}

# ---------------- Cache helpers ----------------
APP_CACHE_VERSION = "v1"  # bump if you change layout/logic

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def _cache_dir() -> Path:
    d = Path(".epw_cache")
    d.mkdir(exist_ok=True)
    return d


# ---------------- EPW parsing ----------------
def parse_epw_header(lines):
    meta = {}
    if lines and lines[0].startswith("LOCATION"):
        p = [x.strip() for x in lines[0].split(",")]
        if len(p) >= 10:
            meta = {
                "city": p[1], "state_region": p[2], "country": p[3],
                "source": p[4], "wmo": p[5], "latitude": p[6],
                "longitude": p[7], "timezone": p[8], "elevation_m": p[9]
            }
    return meta


def epw_to_dataframe(epw_text):
    lines = epw_text.splitlines(True)
    meta = parse_epw_header(lines)
    df = pd.read_csv(StringIO("".join(lines[8:])), header=None)

    # Columns used
    df.rename(columns={
        0:"Year",1:"Month",2:"Day",3:"Hour",
        6:"DryBulbC",7:"DewPointC",8:"RelHumPct",
        13:"GHI",14:"DNI",15:"DHI",
        20:"WindDirection",21:"WindSpeed"
    }, inplace=True)

    for c in ["Year","Month","Day","Hour","DryBulbC","DewPointC","RelHumPct",
              "GHI","DNI","DHI","WindDirection","WindSpeed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean
    df = df[(df["WindSpeed"] >= 0) & (df["WindSpeed"] < 100)]
    df = df[(df["WindDirection"] >= 0) & (df["WindDirection"] < 360)]
    if "RelHumPct" in df: df["RelHumPct"] = df["RelHumPct"].clip(0, 100)
    for r in ["GHI","DNI","DHI"]:
        if r in df: df[r] = df[r].where(df[r] >= 0)

    # Timestamp (Hour 1..24 → 0..23)
    try:
        hour0 = (df["Hour"].astype("Int64") - 1).clip(0, 23).astype(int)
        base = pd.to_datetime(dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int)
        ), errors="coerce")
        df["Timestamp"] = base + pd.to_timedelta(hour0, unit="h")
    except Exception:
        df["Timestamp"] = pd.RangeIndex(len(df))
    return df, meta


# ---------------- Wind summaries ----------------
def bin_direction_labels(series: pd.Series) -> pd.Categorical:
    deg = pd.to_numeric(series, errors="coerce") % 360.0
    idx = np.floor((deg + 11.25) / 22.5).astype("Int64") % 16
    labels = pd.Series(idx).map(lambda i: DIR_LABELS[int(i)] if pd.notna(i) else np.nan)
    return pd.Categorical(labels, categories=DIR_LABELS, ordered=True)


def summarize_wind(df):
    dir_cat = bin_direction_labels(df["WindDirection"])
    speed_bins = [0, 0.5, 2, 4, 6, 8, 10, 100]
    speed_cat = pd.cut(df["WindSpeed"], bins=speed_bins, labels=SPEED_LABELS, right=False)
    tmp = pd.DataFrame({"dir_bin": dir_cat, "speed_bin": speed_cat}).dropna()
    if tmp.empty:
        return pd.DataFrame({"dir_bin": [], "speed_bin": [], "frequency": []})
    counts = tmp.groupby(["dir_bin","speed_bin"], observed=True).size().reset_index(name="count")
    counts["frequency"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts[["dir_bin","speed_bin","frequency"]]


def summarize_wind_by_month(df):
    out = {}
    for m in range(1, 13):
        sub = df[df["Month"] == m]
        out[m] = summarize_wind(sub) if not sub.empty else pd.DataFrame(
            [{"dir_bin": d, "speed_bin": s, "frequency": 0.0} for d in DIR_LABELS for s in SPEED_LABELS]
        )
    return out


# ---------------- Figures ----------------
def make_wind_rose(counts, title, hide_radial_ticks=True, size_px=700):
    """
    Wind rose with legend on the right side.
    Uses go.Barpolar for broad Plotly compatibility.
    """
    pivot = (
        counts.pivot(index="dir_bin", columns="speed_bin", values="frequency")
              .reindex(index=DIR_LABELS, columns=SPEED_LABELS)
              .fillna(0.0)
    )

    fig = go.Figure()
    for s in SPEED_LABELS:
        r_vals = pivot[s].tolist()
        fig.add_trace(go.Barpolar(
            r=r_vals,
            theta=DIR_LABELS,
            name=s,
            marker_color=CUSTOM_COLORS[s],
            opacity=1.0,
            text=r_vals,  # reliable numbers for hover
            hovertemplate="%{theta}<br>%{fullData.name}: <b>%{text:.2f}%</b><extra></extra>"
        ))

    # Legend on the RIGHT side; add right margin so nothing is clipped
    fig.update_layout(
    template=None,                       # remove dark theme
    paper_bgcolor="white",               # white outside the plot
    plot_bgcolor="white",                # white inside polar area
    title=title,
    autosize=True,
    polar=dict(
        bgcolor="white",                 # white polar circle
        radialaxis=dict(
            showticklabels=not hide_radial_ticks,
            gridcolor="#ccc",
            linecolor="#999"
        ),
        angularaxis=dict(
            categoryarray=DIR_LABELS,
            categoryorder="array",
            direction="clockwise",
            rotation=90,
            gridcolor="#ccc",
            linecolor="#999"
        ),
        barmode="stack"
    ),

        legend_title_text="",
        legend=dict(
            orientation="v",
            y=0.5, yanchor="middle",
            x=1.02, xanchor="left",         # push to the right of the plotting area
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=180, t=60, b=40),  # extra right margin for legend
        hovermode="closest"
    )
    return fig





def radiation_monthly_ghi_dni_dhi(df):
    """Monthly radiation:
       - Bars: DHI + DNI (stacked, kWh/m²)
       - Line: GHI (kWh/m²)
    """
    have = {c: (c in df.columns) for c in ("GHI","DNI","DHI")}
    if not any(have.values()):
        return go.Figure().update_layout(template="plotly_dark")

    agg_cols = [c for c in ("DHI","DNI","GHI") if c in df.columns]
    m = (df.groupby("Month")[agg_cols]
           .sum(min_count=1)
           .reindex(range(1, 13))
           .fillna(0.0) / 1000.0)  # convert Wh/m² → kWh/m²
    m["MonthName"] = m.index.map(MONTH_MAP)

    fig = go.Figure()
    if "DHI" in m.columns:
        fig.add_trace(go.Bar(
            x=m["MonthName"], y=m["DHI"],
            name=NAME_MAP["DHI"],
            marker_color=COLOR_MAP_RAD["Diffuse Horizontal Radiation"],
            hovertemplate="<b>%{x}</b><br>DHI: %{y:.1f} kWh/m²<extra></extra>"
        ))
    if "DNI" in m.columns:
        fig.add_trace(go.Bar(
            x=m["MonthName"], y=m["DNI"],
            name=NAME_MAP["DNI"],
            marker_color=COLOR_MAP_RAD["Direct Normal Radiation"],
            hovertemplate="<b>%{x}</b><br>DNI: %{y:.1f} kWh/m²<extra></extra>"
        ))
    if "GHI" in m.columns:
        fig.add_trace(go.Scatter(
            x=m["MonthName"], y=m["GHI"],
            mode="lines+markers",
            name=NAME_MAP["GHI"],
            line=dict(color=COLOR_MAP_RAD["Global Horizontal Radiation"], width=3),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>GHI: %{y:.1f} kWh/m²<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Solar Radiation (kWh/m²)",
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified"
    )
    return fig


def temperature_monthly_minmax_lines(df, height=420):
    need_cols = ["DryBulbC", "DewPointC", "Month"]
    if not all(c in df.columns for c in need_cols):
        return go.Figure().update_layout(template="plotly_dark", height=height)

    monthly = (
        df.groupby("Month")[["DryBulbC", "DewPointC"]]
          .agg(["min", "max"])
          .reindex(range(1, 13))
    )
    monthly.columns = [f"{v}_{s}" for v, s in monthly.columns]
    monthly = monthly.reset_index()
    monthly["MonthName"] = monthly["Month"].map(MONTH_MAP)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["DryBulbC_max"],
        mode="lines+markers", name="Max Drybulb Temp",
        line=dict(color="#ff4f5e", width=2),
        hovertemplate="%{x}: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["DryBulbC_min"],
        mode="lines+markers", name="Min Drybulb Temp",
        line=dict(color="#ff6ea6", width=2),
        hovertemplate="%{x}: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["DewPointC_max"],
        mode="lines+markers", name="Max Dewpoint Temp",
        line=dict(color="#5b7cff", width=2),
        hovertemplate="%{x}: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["DewPointC_min"],
        mode="lines+markers", name="Min Dewpoint Temp",
        line=dict(color="#b4b4c7", width=2),
        hovertemplate="%{x}: %{y:.1f}°C<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark", height=height, hovermode="x unified",
        yaxis_title="Temperature (°C)", xaxis_title="",
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_yaxes(
        range=[-20, 40], tick0=-20, dtick=10,
        showline=True, linewidth=1, linecolor="#888", mirror=True,
        zeroline=True, zerolinewidth=1, zerolinecolor="#666"
    )
    try:
        fig.add_hline(y=-20, line_width=1, line_color="#888", line_dash="dot", opacity=0.6)
        fig.add_hline(y=40,  line_width=1, line_color="#888", line_dash="dot", opacity=0.6)
    except Exception:
        fig.update_layout(shapes=[
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=-20, y1=-20,
                 line=dict(color="#888", width=1, dash="dot")),
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=40,  y1=40,
                 line=dict(color="#888", width=1, dash="dot")),
        ])
    return fig


def humidity_monthly_minmax_lines(df, height=420):
    if "RelHumPct" not in df.columns or "Month" not in df.columns:
        return go.Figure().update_layout(template="plotly_dark", height=height)

    monthly = (
        df.groupby("Month")[["RelHumPct"]]
          .agg(["min", "max"])
          .reindex(range(1, 13))
    )
    monthly.columns = [f"{v}_{s}" for v, s in monthly.columns]
    monthly = monthly.reset_index()
    monthly["MonthName"] = monthly["Month"].map(MONTH_MAP)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["RelHumPct_max"],
        mode="lines+markers", name="Max RH (%)",
        line=dict(color="#00c2b2", width=2),
        hovertemplate="%{x}: %{y:.0f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["MonthName"], y=monthly["RelHumPct_min"],
        mode="lines+markers", name="Min RH (%)",
        line=dict(color="#8bdad1", width=2),
        hovertemplate="%{x}: %{y:.0f}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark", height=height, hovermode="x unified",
        yaxis_title="Relative Humidity (%)", xaxis_title="",
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_yaxes(
        range=[0, 100], tick0=0, dtick=10,
        showline=True, linewidth=1, linecolor="#888", mirror=True,
        zeroline=False
    )
    try:
        fig.add_hline(y=0,   line_width=1, line_color="#888", line_dash="dot", opacity=0.5)
        fig.add_hline(y=100, line_width=1, line_color="#888", line_dash="dot", opacity=0.5)
    except Exception:
        fig.update_layout(shapes=[
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0,   y1=0,
                 line=dict(color="#888", width=1, dash="dot")),
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                 line=dict(color="#888", width=1, dash="dot")),
        ])
    return fig


# ---------------- Summary helpers ----------------
def safe_series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

def annual_summary(df):
    dry  = safe_series(df, "DryBulbC")
    dew  = safe_series(df, "DewPointC")
    rh   = safe_series(df, "RelHumPct")
    ws   = safe_series(df, "WindSpeed")
    ghi  = safe_series(df, "GHI")
    dni  = safe_series(df, "DNI")
    dhi  = safe_series(df, "DHI")

    out = {}
    out["temp_min"]  = float(np.nanmin(dry))  if not dry.dropna().empty else None
    out["temp_max"]  = float(np.nanmax(dry))  if not dry.dropna().empty else None
    out["temp_mean"] = float(np.nanmean(dry)) if not dry.dropna().empty else None
    out["rh_min"]  = float(np.nanmin(rh))   if not rh.dropna().empty else None
    out["rh_max"]  = float(np.nanmax(rh))   if not rh.dropna().empty else None
    out["rh_mean"] = float(np.nanmean(rh))  if not rh.dropna().empty else None
    out["wind_min"]  = float(np.nanmin(ws))  if not ws.dropna().empty else None
    out["wind_max"]  = float(np.nanmax(ws))  if not ws.dropna().empty else None
    out["wind_mean"] = float(np.nanmean(ws)) if not ws.dropna().empty else None

    def _tot_kwh(series):
        return float(np.nansum(series) / 1000.0) if not series.dropna().empty else None
    out["ghi_kwhm2"] = _tot_kwh(ghi)
    out["dni_kwhm2"] = _tot_kwh(dni)
    out["dhi_kwhm2"] = _tot_kwh(dhi)
    return out

def monthly_summary_table(df):
    means_cols = [c for c in ["DryBulbC","DewPointC","RelHumPct","WindSpeed"] if c in df.columns]
    means = (df.groupby("Month")[means_cols]
               .mean()
               .rename(columns={
                   "DryBulbC":"DryBulbC_mean",
                   "DewPointC":"DewPointC_mean",
                   "RelHumPct":"RelHumPct_mean",
                   "WindSpeed":"WindSpeed_mean"
               }))

    sols_cols = [c for c in ["GHI","DNI","DHI"] if c in df.columns]
    sols = (df.groupby("Month")[sols_cols]
              .sum(min_count=1)
              .rename(columns={"GHI":"GHI_kWhm2","DNI":"DNI_kWhm2","DHI":"DHI_kWhm2"}) / 1000.0)

    tbl = pd.concat([means, sols], axis=1).reindex(range(1,13))
    tbl["Month"] = tbl.index.map(MONTH_MAP)
    ordered = ["Month","DryBulbC_mean","DewPointC_mean","RelHumPct_mean","WindSpeed_mean",
               "GHI_kWhm2","DNI_kWhm2","DHI_kWhm2"]
    tbl = tbl.reset_index(drop=True)[[c for c in ordered if c in tbl.columns]]

    disp = tbl.copy()
    for c in disp.columns:
        if c != "Month":
            disp[c] = disp[c].astype(float).round(1)
    return disp


# ---------- Interactive HTML download helpers ----------
def fig_to_html_bytes(fig, filename_prefix="chart"):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return html.encode("utf-8")


def build_monthly_wind_grid_html(by_month, size_px=240, caption_height=22,
                                 hide_radial_ticks=True, show_modebar=True):
    DIR_ORDER = DIR_LABELS
    SPEEDS = SPEED_LABELS

    legend_items = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{CUSTOM_COLORS[s]};"></span>{s}</div>'
        for s in SPEEDS
    )

    specs = []
    for m in range(1, 13):
        dfm = by_month[m].copy()
        pivot = (
            dfm.pivot(index="dir_bin", columns="speed_bin", values="frequency")
               .reindex(index=DIR_ORDER, columns=SPEEDS)
               .fillna(0.0)
        )
        traces = []
        for s in SPEEDS:
            vals = pivot[s].tolist()
            traces.append({
                "type": "barpolar",
                "r": vals,
                "theta": DIR_ORDER,
                "name": s,
                "text": vals,
                "marker": {"color": CUSTOM_COLORS[s]},
                "hovertemplate": "%{theta}<br>%{fullData.name}: <b>%{text:.2f}%</b><extra></extra>",
                "opacity": 1.0
            })
        specs.append({
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "width": size_px, "height": size_px,
                "margin": {"l": 24, "r": 24, "t": 44, "b": 44},
                "polar": {
                    "radialaxis": {"showticklabels": not hide_radial_ticks},
                    "angularaxis": {
                        "categoryarray": DIR_ORDER, "categoryorder": "array",
                        "direction": "clockwise", "rotation": 90
                    },
                    "barmode": "stack"
                },
                "showlegend": False
            }
        })

    specs_json  = json.dumps(specs)
    labels_json = json.dumps(MONTH_NAMES)

    # ---- 3 columns × 4 rows + side legend ----
    gap = 16
    cols, rows = 3, 4
    legend_w = 180
    grid_w = cols * size_px + (cols - 1) * gap
    total_w = grid_w + gap + legend_w
    # Recommended height: rows*(chart+caption) + gaps + header + extra padding
    total_h = rows * (size_px + caption_height) + (rows - 1) * gap + 40 + 120
    display_modebar = 'true' if show_modebar else 'false'

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Monthly Wind Roses</title>
  <style>
    :root {{
      --gap: {gap}px;
      --size: {size_px}px;
      --cap: {caption_height}px;
      --legend-w: {legend_w}px;
    }}
    body {{
      background:#111; color:#ddd; font-family:system-ui,Arial,Helvetica,sans-serif;
    }}
    #wrap {{
      width:{total_w}px; margin:24px auto; display:flex; gap: var(--gap);
      align-items:flex-start;
    }}
    #grid {{
      width:{grid_w}px;
      display:grid;
      grid-template-columns: repeat(3, var(--size));
      gap: var(--gap);
    }}
    .cell {{ display:flex; flex-direction:column; align-items:center; }}
    .chart {{ width:var(--size); height:var(--size); }}
    .cap {{
      height:var(--cap); line-height:var(--cap);
      text-align:center; font-size:13px; color:#ddd; margin-top:4px; user-select:none;
    }}
    #legend {{
      width:var(--legend-w);
      border-left:1px solid #2a2a2a;
      padding-left:12px;
      position:sticky; top:16px;
    }}
    #legend h3 {{ margin:4px 0 8px; font-size:16px; color:#e0e0e0; }}
    .legend-row {{
      display:flex; align-items:center; gap:8px; margin:6px 0;
      font-size:13px; color:#ddd;
    }}
    .swatch {{
      width:14px; height:14px; border-radius:3px; display:inline-block;
      border:1px solid #333;
    }}
  </style>
</head>
<body>
  <h2 style="text-align:center;margin:16px 0;">Monthly Wind Roses</h2>

  <div id="wrap">
    <div id="grid">
      {''.join([f'<div class="cell"><div id="mchart-{i}" class="chart"></div><div class="cap" id="mcap-{i}"></div></div>' for i in range(12)])}
    </div>

    <aside id="legend">
      <h3>Legend</h3>
      {legend_items}
    </aside>
  </div>

  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>
    const specs  = {specs_json};
    const labels = {labels_json};

    function renderOne(i){{
      const el = document.getElementById("mchart-"+i);
      const cap = document.getElementById("mcap-"+i);
      cap.textContent = labels[i];
      const spec = specs[i];
      Plotly.newPlot(el, spec.data, spec.layout, {{
        displaylogo:false,
        displayModeBar:{display_modebar},
        responsive:true,
        toImageButtonOptions:{{format:'png', filename:'wind_rose_month_'+(i+1)}}
      }}).then(()=>{{
        const n = spec.data.length, dim = 0.20;
        function resetAll(){{ for(let t=0;t<n;t++) Plotly.restyle(el,{{opacity:1.0}},[t]); }}
        el.on('plotly_hover', ev=>{{
          if(!ev||!ev.points||!ev.points.length) return;
          const active = ev.points[0].curveNumber;
          for(let t=0;t<n;t++) Plotly.restyle(el,{{opacity:(t===active?1.0:dim)}},[t]);
        }});
        el.on('plotly_unhover', ()=>resetAll());
        window.addEventListener('resize', ()=>Plotly.Plots.resize(el));
      }});
    }}
    for(let i=0;i<12;i++) renderOne(i);
  </script>
</body>
</html>
"""
    return html.encode("utf-8"), int(total_h)




# ---------------- Wind-rose renderer (hover highlight) ----------------
def render_with_hover_highlight(fig, dom_id: str, width_px=700, height_px=700, filename_prefix="wind_rose"):
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    html = f"""
<div id="{dom_id}" style="width:100%; max-width:{width_px}px; height:{height_px + 90}px; margin:0 auto;"></div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
  const el = document.getElementById('{dom_id}');
  const fig = {fig_json};
  Plotly.newPlot(el, fig.data, fig.layout, {{
    displaylogo: false,
    responsive: true,                    // 👈 key
    toImageButtonOptions: {{format: 'png', filename: '{filename_prefix}'}}
  }}).then(() => {{
    // keep it responsive on container resize
    window.addEventListener('resize', () => Plotly.Plots.resize(el));

    // highlight-on-hover (as before)
    const n = fig.data.length, dim = 0.20;
    function resetAll() {{ for (let i = 0; i < n; i++) Plotly.restyle(el, {{opacity: 1.0}}, [i]); }}
    el.on('plotly_hover', (ev) => {{
      if (!ev || !ev.points || !ev.points.length) return;
      const active = ev.points[0].curveNumber;
      for (let i = 0; i < n; i++) Plotly.restyle(el, {{opacity: (i === active ? 1.0 : dim)}}, [i]);
    }});
    el.on('plotly_unhover', () => resetAll());
  }});
</script>
"""
    components.html(html, height=height_px + 140, scrolling=False)



# ---------- Plotly → image helpers for PDF ----------
def _plotly_to_png(fig, width, height, scale=2):
    return fig.to_image(format="png", width=width, height=height, scale=scale)

def _plotly_to_svg(fig, width, height, scale=1):
    return fig.to_image(format="svg", width=width, height=height, scale=scale)

def draw_plotly_on_canvas(c, fig, x, y, w, h, prefer_svg=True):
    """
    Draw a Plotly fig on a reportlab canvas at (x,y) with width=w, height=h.
    Tries SVG (vector) if available, otherwise PNG fallback.
    """
    if prefer_svg and _SVG_OK:
        try:
            svg_bytes = _plotly_to_svg(fig, int(w), int(h), scale=1)
            drawing = svg2rlg(BytesIO(svg_bytes))
            sx = w / drawing.width
            sy = h / drawing.height
            s = min(sx, sy)
            drawing.width *= s
            drawing.height *= s
            drawing.scale(s, s)
            renderPDF.draw(drawing, c, x, y)
            return
        except Exception:
            pass
    png = _plotly_to_png(fig, int(w), int(h), scale=2)
    img = ImageReader(BytesIO(png))
    c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')


def make_monthly_wind_figs(by_month, size_px=520):
    figs = []
    for m in range(1, 13):
        title = f"Wind Rose – {MONTH_MAP[m]}"
        fig = make_wind_rose(by_month[m], title, hide_radial_ticks=True, size_px=size_px)
        fig.update_layout(title_text=MONTH_MAP[m], margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15))
        figs.append(fig)
    return figs


# ---------- Fancy PDF builder ----------
def build_pdf_report(meta, df,
                     fig_wind_annual, by_month,
                     fig_temp, fig_hum, fig_rad,
                     counts_annual):
    """
    Fancy multi-page PDF:
      - Cover with accent bar
      - Header/footer with page numbers
      - Summary page with formatted monthly table
      - Wind (annual), Wind (monthly x 3 pages)
      - Temperature, Humidity, Radiation
    """
    # Ensure kaleido exists (Plotly image export)
    try:
        _ = fig_wind_annual.to_image
    except Exception as e:
        raise RuntimeError("Kaleido is required for PDF charts. Install: pip install -U kaleido") from e

    W, H = A4
    M = 36  # margin
    c = canvas.Canvas(BytesIO(), pagesize=A4)

    def header(text):
        c.setFillColorRGB(0.12, 0.12, 0.12)
        c.rect(0, H-48, W, 48, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(M, H-34, text)

    def footer(page_num=True):
        c.setStrokeColor(colors.HexColor("#444"))
        c.setLineWidth(0.3)
        c.line(M, 40, W-M, 40)
        if page_num:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#888"))
            c.drawRightString(W-M, 26, f"Page {c.getPageNumber()}")

    def paragraph(text, x, y, width=W-2*M, leading=14, font="Helvetica", size=10, color=colors.black):
        c.setFont(font, size)
        c.setFillColor(color)
        from textwrap import wrap
        lines = []
        for ln in text.split("\n"):
            lines.extend(wrap(ln, int(width/5.2)))
        yy = y
        for ln in lines:
            c.drawString(x, yy, ln)
            yy -= leading
        return yy

    def new_page(title):
        c.showPage()
        header(title)
        footer()

    # COVER
    header("EPW Wind & Climate Report")
    footer(page_num=False)
    y = H - 80
    c.setFillColor(colors.HexColor("#ddd"))
    c.setFont("Helvetica", 11)
    if meta:
        c.drawString(M, y, f"{meta.get('city','')}, {meta.get('state_region','')}, {meta.get('country','')}")
        y -= 16
        c.drawString(M, y, f"Lat/Lon: {meta.get('latitude','')} , {meta.get('longitude','')}   "
                           f"TZ: {meta.get('timezone','')}   Elev (m): {meta.get('elevation_m','')}")
        y -= 16
        c.drawString(M, y, f"Source: {meta.get('source','')}   WMO: {meta.get('wmo','')}")
        y -= 24
    intro = ("This report summarizes wind, temperature, humidity, and solar radiation statistics "
             "derived from the uploaded EPW weather file. All charts are based on the hourly series "
             "and aggregated monthly or annually as indicated.")
    paragraph(intro, M, y, color=colors.HexColor("#ddd"))
    c.setFillColor(colors.HexColor("#1f77b4"))
    c.rect(0, 0, W, 8, stroke=0, fill=1)

    # SUMMARY
    new_page("Summary Metrics")
    s = annual_summary(df)
    y = H - 96
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    c.drawString(M, y, f"Temperature (°C): min {s['temp_min']:.1f}, mean {s['temp_mean']:.1f}, max {s['temp_max']:.1f}"); y -= 16
    c.drawString(M, y, f"Relative Humidity (%): min {s['rh_min']:.0f}, mean {s['rh_mean']:.0f}, max {s['rh_max']:.0f}"); y -= 16
    c.drawString(M, y, f"Wind Speed (m/s): min {s['wind_min']:.1f}, mean {s['wind_mean']:.1f}, max {s['wind_max']:.1f}"); y -= 16
    c.drawString(M, y, f"Annual Solar (kWh/m²): GHI {s['ghi_kwhm2']:.0f}, DNI {s['dni_kwhm2']:.0f}, DHI {s['dhi_kwhm2']:.0f}")
    y -= 28

    # Monthly summary table
    tbl = monthly_summary_table(df)
    data = [tbl.columns.tolist()] + tbl.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c2f33")),
        ("FONT", (0,1), (-1,-1), "Helvetica", 9),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.Color(0.96,0.96,0.96)]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#aaa")),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    avail_h = y - 40
    w, h = table.wrapOn(c, W-2*M, avail_h)
    table.drawOn(c, M, y-h)

    # WIND – ANNUAL
    new_page("Wind – Annual Rose")
    y_top = H - 96
    paragraph(("The wind rose shows the distribution of wind by direction and speed bins. "
               "Bars are stacked by speed categories; angle indicates direction (N at top, clockwise)."),
              M, y_top)
    box_w = (W - 2*M)
    box_h = box_w * 0.9
    draw_plotly_on_canvas(c, fig_wind_annual, M, y_top - box_h - 16, box_w, box_h)

    # WIND – MONTHLY (4 per page * 3 pages)
    figs = make_monthly_wind_figs(by_month, size_px=520)
    grid = [(0,0), (1,0), (0,1), (1,1)]
    for i in range(0, 12, 4):
        new_page("Wind – Monthly Roses")
        cell_w = (W - 3*M) / 2
        cell_h = cell_w
        x0 = M; y0 = H - 96
        for j in range(4):
            if i+j >= len(figs): break
            gx, gy = grid[j]
            x = M + gx*(cell_w + M)
            y = y0 - (gy+1)*(cell_h + 40) + 40
            draw_plotly_on_canvas(c, figs[i+j], x, y, cell_w, cell_h)
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#555"))
            c.drawCentredString(x + cell_w/2, y - 12, MONTH_MAP[i+j+1])

    # TEMPERATURE
    new_page("Temperature – Monthly Min / Max")
    y_top = H - 96
    paragraph(("Lines show monthly minima and maxima for dry-bulb and dew-point temperatures. "
               "Axis is fixed to −20…40 °C for comparison."), M, y_top)
    draw_plotly_on_canvas(c, fig_temp, M, 90, W-2*M, y_top-140)

    # HUMIDITY
    new_page("Humidity – Monthly Min / Max")
    y_top = H - 96
    paragraph(("Lines show monthly minima and maxima of relative humidity. Axis spans 0–100%."), M, y_top)
    draw_plotly_on_canvas(c, fig_hum, M, 90, W-2*M, y_top-140)

    # RADIATION
    new_page("Solar Radiation – Monthly Totals")
    y_top = H - 96
    paragraph(("Stacked bars show diffuse (DHI) and direct normal (DNI) monthly totals; "
               "the line overlays global horizontal (GHI). Units are kWh/m²."), M, y_top)
    draw_plotly_on_canvas(c, fig_rad, M, 90, W-2*M, y_top-140)

    c.save()
    return c.getpdfdata()


def build_pdf_report_cached(epw_bytes: bytes, meta, df,
                            fig_wind_annual, by_month, fig_temp, fig_hum, fig_rad,
                            counts_annual) -> bytes:
    """Write once, read many: cache PDF by EPW+version hash."""
    h = _hash_bytes(epw_bytes) + "-" + APP_CACHE_VERSION
    out_path = _cache_dir() / f"epw_report_{h}.pdf"
    if out_path.exists():
        return out_path.read_bytes()
    pdf_bytes = build_pdf_report(meta, df, fig_wind_annual, by_month, fig_temp, fig_hum, fig_rad, counts_annual)
    out_path.write_bytes(pdf_bytes)
    return pdf_bytes


# ---------------- UI ----------------
st.title("EPW Viewer – Wind & Climate")

with st.sidebar:
    st.subheader("Sections")
    section = st.radio("", ["Summary", "Wind rose", "Temperature", "Humidity", "Radiation"], index=0)

uploaded = st.file_uploader("Upload EPW file", type=["epw"], label_visibility="collapsed")
if uploaded is None:
    st.info("👆 Upload an `.epw` file to begin.")
    st.stop()

file_name = getattr(uploaded, "name", "EPW file")
try:
    epw_raw = uploaded.getvalue()
    epw_text = epw_raw.decode("latin1", errors="ignore")
    df, meta = epw_to_dataframe(epw_text)
except Exception as e:
    st.error(f"Error reading EPW: {e}")
    st.stop()

# Metadata
with st.expander("File metadata", expanded=False):
    st.write(f"**File:** {file_name}")
    if meta:
        st.markdown(
            f"**Location:** {meta.get('city','')}, {meta.get('state_region','')}, {meta.get('country','')}  \n"
            f"**Source:** {meta.get('source','')}  **WMO:** {meta.get('wmo','')}  \n"
            f"**Lat/Lon:** {meta.get('latitude','')}, {meta.get('longitude','')}  "
            f"**TZ:** {meta.get('timezone','')}  **Elev (m):** {meta.get('elevation_m','')}"
        )

# Precompute wind summaries
counts_annual = summarize_wind(df)
by_month = summarize_wind_by_month(df)

# -------- Sections --------
if section == "Summary":
    st.header("Summary")

    summ = annual_summary(df)
    monthly_tbl = monthly_summary_table(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temp min (°C)",  f"{summ['temp_min']:.1f}"  if summ['temp_min']  is not None else "—")
    with col2:
        st.metric("Temp mean (°C)", f"{summ['temp_mean']:.1f}" if summ['temp_mean'] is not None else "—")
    with col3:
        st.metric("Temp max (°C)",  f"{summ['temp_max']:.1f}"  if summ['temp_max']  is not None else "—")
    with col4:
        st.metric("RH mean (%)",    f"{summ['rh_mean']:.0f}"   if summ['rh_mean']   is not None else "—")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Wind min (m/s)",  f"{summ['wind_min']:.1f}"  if summ['wind_min']  is not None else "—")
    with c6:
        st.metric("Wind mean (m/s)", f"{summ['wind_mean']:.1f}" if summ['wind_mean'] is not None else "—")
    with c7:
        st.metric("Wind max (m/s)",  f"{summ['wind_max']:.1f}"  if summ['wind_max']  is not None else "—")

    c8, c9, c10 = st.columns(3)
    with c8:
        st.metric("GHI total (kWh/m²)", f"{summ['ghi_kwhm2']:.0f}" if summ['ghi_kwhm2'] is not None else "—")
    with c9:
        st.metric("DNI total (kWh/m²)", f"{summ['dni_kwhm2']:.0f}" if summ['dni_kwhm2'] is not None else "—")
    with c10:
        st.metric("DHI total (kWh/m²)", f"{summ['dhi_kwhm2']:.0f}" if summ['dhi_kwhm2'] is not None else "—")

    st.subheader("Monthly summary")
    st.caption("Temp, dew-point, RH, wind are **means**. Solar values are **monthly totals** in kWh/m².")
    st.dataframe(monthly_tbl, use_container_width=True)

    st.download_button(
        "⬇️ Download monthly summary (CSV)",
        data=monthly_tbl.to_csv(index=False),
        file_name="monthly_summary.csv",
        mime="text/csv"
    )

    # Build charts for PDF and enable download
    fig_wind_annual = make_wind_rose(counts_annual, "Annual", hide_radial_ticks=True, size_px=740)
    fig_temp = temperature_monthly_minmax_lines(df)
    fig_hum  = humidity_monthly_minmax_lines(df)
    fig_rad  = radiation_monthly_ghi_dni_dhi(df)

    try:
        pdf_bytes = build_pdf_report_cached(
            epw_raw, meta, df,
            fig_wind_annual, by_month,
            fig_temp, fig_hum, fig_rad,
            counts_annual
        )
        st.download_button(
            "📄 Download full PDF report (all charts + explanations)",
            data=pdf_bytes,
            file_name="epw_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(str(e))
        st.info("Install exporters: pip install -U kaleido reportlab pillow  (optional: svglib)")

elif section == "Wind rose":
    st.header("Wind rose")

    # ---- Annual on top ----
    st.subheader("Annual")
    fig_annual = make_wind_rose(counts_annual, "Annual", hide_radial_ticks=True, size_px=740)
    render_with_hover_highlight(
        fig_annual,
        dom_id="annual-rose",
        width_px=980,    # max width; scales down
        height_px=740,
        filename_prefix="wind_rose_annual"
    )
    st.download_button(
        "⬇️ Download annual wind rose (HTML, interactive)",
        data=fig_to_html_bytes(fig_annual, "wind_rose_annual"),
        file_name="wind_rose_annual.html",
        mime="text/html"
    )

    st.markdown("---")

    # ---- Monthly below ----
    st.subheader("Monthly")
    grid_html_bytes, grid_h = build_monthly_wind_grid_html(
         by_month,
         size_px=260, caption_height=22,
         hide_radial_ticks=True, show_modebar=True
    )
    components.html(grid_html_bytes.decode("utf-8"), height=grid_h, scrolling=False)

    st.download_button(
       "⬇️ Download monthly wind roses (HTML, interactive)",
       data=grid_html_bytes,
       file_name="wind_rose_monthly_grid.html",
       mime="text/html"
    )


elif section == "Temperature":
    st.header("Temperatures – Monthly min & max (Dry-bulb / Dew-point)")
    fig_temp = temperature_monthly_minmax_lines(df)
    st.plotly_chart(fig_temp, use_container_width=True)
    st.download_button(
        "⬇️ Download temperature chart (HTML, interactive)",
        data=fig_to_html_bytes(fig_temp, "temperature_monthly_minmax"),
        file_name="temperature_monthly_minmax.html",
        mime="text/html"
    )

elif section == "Humidity":
    st.header("Relative Humidity – Monthly min & max (%)")
    fig_hum = humidity_monthly_minmax_lines(df)
    st.plotly_chart(fig_hum, use_container_width=True)
    st.download_button(
        "⬇️ Download humidity chart (HTML, interactive)",
        data=fig_to_html_bytes(fig_hum, "humidity_monthly_minmax"),
        file_name="humidity_monthly_minmax.html",
        mime="text/html"
    )

elif section == "Radiation":
    st.header("Solar radiation – Monthly totals (kWh/m²)")
    st.caption("Bars: **Diffuse (DHI)** + **Direct Normal (DNI)**. Line: **Global Horizontal (GHI)**.")
    fig_rad = radiation_monthly_ghi_dni_dhi(df)
    st.plotly_chart(fig_rad, use_container_width=True)
    st.download_button(
        "⬇️ Download radiation chart (HTML, interactive)",
        data=fig_to_html_bytes(fig_rad, "radiation_monthly_ghi_dni"),
        file_name="radiation_monthly.html",
        mime="text/html"
    )

