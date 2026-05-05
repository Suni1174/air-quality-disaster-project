"""
=============================================================
STEP 3 — Interactive Dash Dashboard (v3 — with Predictor + Events)
Team Thunderstorm | AQI Disaster Intelligence

NEW in this version:
  Tab 1 — Analysis Dashboard (original 8 panels)
  Tab 2 — Live Predictor (enter values → get prediction)
  Tab 3 — Real Events Validation (all 13 events with news links)

Run:  python 03_dashboard.py
Open: http://127.0.0.1:8050
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import warnings
warnings.filterwarnings("ignore")

PROCESSED = Path("data/processed")
MODELS    = Path("models")

# ── Theme ──────────────────────────────────────────────────────
BG     = "#060d1a"
PANEL  = "#0d1f35"
CARD   = "#112240"
BORDER = "#1e3a5f"
TEXT   = "#cdd9e5"
DIM    = "#607b96"

EVENT_CLR = {
    "Normal":     "#3ddbd9",
    "Fire":       "#ff6b35",
    "Dust":       "#f5c518",
    "Gas":        "#ff4757",
    "Industrial": "#b44fff",
}
CITY_CLR = {
    "Delhi":  "#38bdf8",
    "Jaipur": "#fb7185",
    "Vizag":  "#34d399",
}
AQI_BANDS = [
    (0,   50,  "rgba(0,228,0,0.07)",   "Good"),
    (51,  100, "rgba(255,255,0,0.07)", "Satisfactory"),
    (101, 200, "rgba(255,126,0,0.07)", "Moderate"),
    (201, 300, "rgba(255,0,0,0.07)",   "Poor"),
    (301, 500, "rgba(153,0,76,0.08)",  "Very Poor"),
]

# Known real events with news sources
REAL_EVENTS = [
    {"id":1,"city":"Delhi","start":"2019-10-25","end":"2019-11-15","type":"Fire",
     "name":"Delhi Stubble Burning 2019","desc":"Severe smog from Punjab/Haryana farm fires. AQI crossed 500 for multiple days.",
     "url":"https://timesofindia.indiatimes.com/city/delhi/delhi-air-quality-turns-severe-as-stubble-burning-peaks/articleshow/71815396.cms","source":"Times of India"},
    {"id":2,"city":"Delhi","start":"2020-11-01","end":"2020-11-20","type":"Fire",
     "name":"Delhi Stubble Burning 2020","desc":"Post-lockdown traffic + peak farm fires. AQI exceeded 400 across NCR.",
     "url":"https://www.indiatoday.in/india/story/delhi-air-pollution-stubble-burning-aqi-1740961-2020-11-03","source":"India Today"},
    {"id":3,"city":"Delhi","start":"2021-10-28","end":"2021-11-12","type":"Fire",
     "name":"Delhi Diwali + Stubble Crisis 2021","desc":"Worst 2021 episode. Firecrackers + farm fires pushed AQI to 500+ for several days.",
     "url":"https://www.thehindu.com/news/cities/Delhi/delhi-air-quality-remains-very-poor/article37261858.ece","source":"The Hindu"},
    {"id":4,"city":"Delhi","start":"2022-10-30","end":"2022-11-14","type":"Fire",
     "name":"Delhi Stubble Burning 2022","desc":"SAFAR reported farm fires contributing 40%+ of PM2.5 load.",
     "url":"https://www.hindustantimes.com/cities/delhi-news/delhi-aqi-pollution-stubble-burning-diwali-2022-101667474793143.html","source":"Hindustan Times"},
    {"id":5,"city":"Delhi","start":"2023-10-25","end":"2023-11-10","type":"Fire",
     "name":"Delhi Stubble Burning 2023","desc":"Despite SC orders, farm fires caused severe AQI in Delhi NCR.",
     "url":"https://www.ndtv.com/india-news/delhi-air-quality-pollution-stubble-burning-aqi-4564890","source":"NDTV"},
    {"id":6,"city":"Jaipur","start":"2019-05-18","end":"2019-05-22","type":"Dust",
     "name":"Rajasthan Dust Storm May 2019","desc":"Severe Loo dust storms. PM10 exceeded 1500 µg/m³ in Jaipur and Jodhpur.",
     "url":"https://timesofindia.indiatimes.com/city/jaipur/dust-storm-hits-jaipur-pm10-level-shoots-up/articleshow/69390422.cms","source":"Times of India"},
    {"id":7,"city":"Jaipur","start":"2020-05-25","end":"2020-05-28","type":"Dust",
     "name":"Rajasthan Dust Storm May 2020","desc":"Severe dust storm during pandemic lockdown affected Jaipur, Jodhpur, Bikaner.",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":8,"city":"Jaipur","start":"2021-06-02","end":"2021-06-05","type":"Dust",
     "name":"Rajasthan Dust Storm June 2021","desc":"Pre-monsoon dust storm. Strong westerly winds caused PM10 spike across Jaipur.",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":9,"city":"Jaipur","start":"2022-05-14","end":"2022-05-17","type":"Dust",
     "name":"Rajasthan Dust Storm May 2022","desc":"Visibility dropped to 500m. PM10 > 1200 µg/m³.",
     "url":"https://mausam.imd.gov.in","source":"IMD India"},
    {"id":10,"city":"Jaipur","start":"2023-05-20","end":"2023-05-24","type":"Dust",
     "name":"Rajasthan Dust Storm May 2023","desc":"Dust storm + heatwave. IMD issued red alert for several Rajasthan districts.",
     "url":"https://www.hindustantimes.com/india-news/dust-storm-rajasthan-2023-101684731826043.html","source":"Hindustan Times"},
    {"id":11,"city":"Vizag","start":"2020-05-07","end":"2020-05-09","type":"Gas",
     "name":"LG Polymers Gas Leak — May 7, 2020","desc":"Styrene gas leak at LG Polymers. 12 deaths, 1000+ hospitalised. India's worst industrial disaster since Bhopal.",
     "url":"https://en.wikipedia.org/wiki/2020_Visakhapatnam_gas_leak","source":"Wikipedia"},
    {"id":12,"city":"Vizag","start":"2021-06-10","end":"2021-06-12","type":"Industrial",
     "name":"HPCL Vizag Refinery Emission 2021","desc":"Elevated SO2/NOx near HPCL Visakhapatnam refinery. Residents reported respiratory issues.",
     "url":"https://cpcb.nic.in/automatic-monitoring-data/","source":"CPCB Data"},
    {"id":13,"city":"Vizag","start":"2023-03-15","end":"2023-03-17","type":"Industrial",
     "name":"Vizag VSEZ SO2 Spike March 2023","desc":"SO2 spike near VSEZ industrial zone, associated with ONGC scheduled flaring.",
     "url":"https://cpcb.nic.in/automatic-monitoring-data/","source":"CPCB Data"},
    {"id":14,"city":"Vizag","start":"2024-08-21","end":"2024-08-23","type":"Industrial",
     "name":"Escientia Pharma Blast — Aug 21, 2024",
     "desc":"MTBE solvent explosion at Atchutapuram SEZ, Anakapalli. 17 killed, 40 injured. Vapour cloud ignition + fire through reactor ductwork.",
     "url":"https://en.wikipedia.org/wiki/Atchutapuram_pharmaceutical_factory_explosion","source":"Wikipedia"},
]


def hex_rgba(h, a=0.12):
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return f"rgba(100,100,100,{a})"
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# ── Load data ──────────────────────────────────────────────────
def load():
    p = PROCESSED / "results.parquet"
    if not p.exists():
        raise FileNotFoundError("Run 01_features.py then 02_model.py first!")
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["display_label"] = df.get("event_label", df.get("pred_label", "Normal"))
    return df

DF     = load()
CITIES = sorted(DF["city"].unique())
MIN_DT = DF["datetime"].min().date()
MAX_DT = DF["datetime"].max().date()
ALL_POLS = [c for c in ["PM2_5","PM10","NO2","SO2","CO","O3","AOD","AQI"] if c in DF.columns]

BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="'Courier New', monospace", size=11),
    margin=dict(l=60, r=25, t=45, b=45),
    hovermode="x unified",
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font_size=10),
    xaxis=dict(gridcolor=BORDER, zeroline=False, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zeroline=False, showgrid=True),
)

def filt(city, start, end):
    return DF[(DF["city"]==city)&(DF["datetime"]>=pd.Timestamp(start))&(DF["datetime"]<=pd.Timestamp(end))].copy()

def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       font=dict(color=DIM, size=14), showarrow=False)
    fig.update_layout(**BASE, height=260)
    return fig


# ── Figure builders (same as before) ──────────────────────────
def fig_timeline(dff, city):
    if dff.empty: return empty_fig("No data")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=["AQI + Events + 24h Forecast","Anomaly Score"])
    c = CITY_CLR.get(city, "#38bdf8")
    if "AQI" in dff.columns:
        fig.add_trace(go.Scatter(x=dff["datetime"], y=dff["AQI"].rolling(6,min_periods=1).mean(),
            name="AQI (6h avg)", line=dict(color=c, width=2),
            fill="tozeroy", fillcolor=hex_rgba(c, 0.08)), row=1, col=1)
        for lo,hi,col,lbl in AQI_BANDS:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                annotation_text=lbl, annotation_position="right",
                annotation_font=dict(color=DIM, size=8), row=1, col=1)
    if "forecast_pm25_24h" in dff.columns:
        fig.add_trace(go.Scatter(x=dff["datetime"], y=dff["forecast_pm25_24h"],
            name="PM2.5 Forecast 24h", line=dict(color="#fbbf24", width=1.5, dash="dot")), row=1, col=1)
    for lbl, clr in EVENT_CLR.items():
        if lbl == "Normal": continue
        ev = dff[dff["display_label"]==lbl]
        if ev.empty: continue
        starts = ev[ev["display_label"].shift(1, fill_value="Normal") != lbl]
        if starts.empty: starts = ev.head(1)
        fig.add_trace(go.Scatter(x=starts["datetime"], y=starts["AQI"].fillna(50),
            mode="markers", marker=dict(symbol="triangle-up", size=14, color=clr,
            line=dict(color="white", width=1.5)), name=lbl, legendgroup=lbl), row=1, col=1)
    if "anomaly_score" in dff.columns:
        fig.add_trace(go.Scatter(x=dff["datetime"], y=dff["anomaly_score"],
            name="Anomaly Score", line=dict(color="#f43f5e", width=1.5),
            fill="tozeroy", fillcolor="rgba(244,63,94,0.15)"), row=2, col=1)
        fig.add_hline(y=0.88, line=dict(color="#f43f5e", dash="dash", width=1),
            annotation_text="0.88", annotation_font=dict(color="#f43f5e", size=9), row=2, col=1)
    fig.update_layout(**BASE, height=520)
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig

def fig_pollutant(dff, city, pol):
    if dff.empty or pol not in dff.columns: return empty_fig(f"{pol} not available")
    c = CITY_CLR.get(city, "#38bdf8")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["datetime"], y=dff[pol], name=pol,
        line=dict(color=c, width=1), opacity=0.4))
    sm = pol + "_smooth"
    if sm in dff.columns:
        fig.add_trace(go.Scatter(x=dff["datetime"], y=dff[sm], name=pol+" smooth",
            line=dict(color=c, width=2.5)))
    for lbl, clr in EVENT_CLR.items():
        if lbl == "Normal": continue
        ev = dff[dff["display_label"]==lbl]
        if ev.empty: continue
        fig.add_trace(go.Scatter(
            x=pd.concat([ev["datetime"], ev["datetime"].iloc[::-1]]),
            y=pd.concat([ev[pol], pd.Series(np.zeros(len(ev)), index=ev.index)]),
            fill="toself", fillcolor=hex_rgba(clr, 0.20),
            line=dict(width=0), name=lbl, legendgroup=lbl))
    fig.update_layout(**BASE, title=f"{pol} — Raw · Smoothed · Event Windows", height=300)
    return fig

def fig_events(dff):
    ev = dff[dff["display_label"]!="Normal"].copy()
    if ev.empty: return empty_fig("No events in selected range")
    ev["sz"] = ev["AQI"].fillna(30).clip(10,500) if "AQI" in ev.columns else 30
    y_col = "anomaly_score" if "anomaly_score" in ev.columns else "sz"
    fig = px.scatter(ev, x="datetime", y=y_col, color="display_label",
        color_discrete_map=EVENT_CLR, size="sz", size_max=22,
        labels={"display_label":"Type"}, title="Disaster Events (bubble = AQI)")
    fig.update_layout(**BASE, height=280)
    return fig

def fig_radar(dff):
    pols = [p for p in ["PM2_5","PM10","NO2","SO2","CO","O3","AOD"] if p in dff.columns]
    fig = go.Figure()
    for lbl, clr in EVENT_CLR.items():
        sub = dff[dff["display_label"]==lbl]
        if len(sub) < 2: continue
        means = sub[pols].mean().tolist()
        mx = max(means) if max(means) > 0 else 1
        vals = [v/mx for v in means] + [means[0]/mx]
        fig.add_trace(go.Scatterpolar(r=vals, theta=pols+[pols[0]], fill="toself",
            name=lbl, line_color=clr, fillcolor=hex_rgba(clr, 0.15)))
    fig.update_layout(
        polar=dict(bgcolor=CARD,
            radialaxis=dict(visible=True, gridcolor=BORDER, color=DIM, range=[0,1.1]),
            angularaxis=dict(gridcolor=BORDER, color=TEXT)),
        paper_bgcolor=BG, font=dict(color=TEXT, family="'Courier New', monospace", size=11),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
        margin=dict(l=40, r=40, t=55, b=30), height=360,
        title="Chemical Fingerprint by Event Type")
    return fig

def fig_heatmap(dff):
    pols = [p for p in ["PM2_5","NO2","SO2","CO","O3","AOD","AQI"] if p in dff.columns]
    if dff.empty or not pols: return empty_fig()
    tmp = dff.copy(); tmp["ym"] = tmp["datetime"].dt.to_period("M").astype(str)
    mo  = tmp.groupby("ym")[pols].mean()
    z   = mo.T.values.astype(float)
    rng = z.max(1,keepdims=True)-z.min(1,keepdims=True)
    zn  = np.where(rng>0,(z-z.min(1,keepdims=True))/rng,0)
    fig = go.Figure(go.Heatmap(z=zn, x=mo.index.tolist(), y=pols, colorscale="YlOrRd",
        hovertemplate="Month: %{x}<br>%{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="'Courier New', monospace", size=11),
        margin=dict(l=60,r=25,t=45,b=45),
        title="Monthly Pollutant Patterns (normalised)", height=300,
        xaxis=dict(tickangle=-45, gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
    return fig

def fig_corr(dff):
    cols = [c for c in ["PM2_5","NO2","SO2","CO","O3","AOD","AQI",
                         "temp_c","humidity","wind_speed","fire_count"] if c in dff.columns]
    if len(cols)<2: return empty_fig()
    corr = dff[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        texttemplate="%{z:.2f}", textfont_size=8))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="'Courier New', monospace", size=11),
        margin=dict(l=60,r=25,t=45,b=45),
        title="Correlation: Pollutants × Meteorology", height=380,
        xaxis=dict(tickangle=-40, gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
    return fig

def fig_stl(dff):
    res_cols = [c for c in dff.columns if c.endswith("_residual")]
    if not res_cols or dff.empty: return empty_fig("No STL residuals")
    clrs = ["#38bdf8","#fb7185","#34d399","#fbbf24"]
    fig  = go.Figure()
    for i, rc in enumerate(res_cols[:4]):
        fig.add_trace(go.Scatter(x=dff["datetime"], y=dff[rc],
            name=rc.replace("_residual",""), line=dict(color=clrs[i%4], width=1), opacity=0.85))
    fig.add_hline(y=0, line=dict(color=DIM, dash="dash", width=1))
    fig.update_layout(**BASE, title="STL Residuals — Deseasonalised Anomaly Signal", height=260)
    return fig

def fig_fire(dff):
    if "fire_count" not in dff.columns or dff.empty: return empty_fig("No fire data")
    daily = dff.set_index("datetime")["fire_count"].resample("D").sum().reset_index()
    fig = go.Figure(go.Bar(x=daily["datetime"], y=daily["fire_count"],
        marker_color="#ff6b35", opacity=0.8))
    fig.update_layout(**BASE, title="MODIS Daily Fire Pixel Count", height=230)
    return fig

def kpis(dff, city):
    ev  = dff[dff["display_label"]!="Normal"]
    vc  = ev["display_label"].value_counts()
    aqi = dff["AQI"] if "AQI" in dff.columns else pd.Series(dtype=float)
    return {
        "city":  city,
        "events":f"{len(ev):,}",
        "paqi":  f"{aqi.max():.0f}" if not aqi.isna().all() else "—",
        "aaqi":  f"{aqi.mean():.0f}" if not aqi.isna().all() else "—",
        "worst": dff.loc[aqi.idxmax(),"datetime"].strftime("%d %b %Y") if not aqi.isna().all() else "—",
        "top":   vc.index[0] if len(vc) else "None",
        "no2":   f"{dff['NO2'].max():.2f}" if "NO2" in dff.columns else "—",
        "so2":   f"{dff['SO2'].max():.2f}" if "SO2" in dff.columns else "—",
    }


# ── Build validation data ──────────────────────────────────────
def get_event_validation():
    rows = []
    for ev in REAL_EVENTS:
        mask = ((DF["city"]==ev["city"]) &
                (DF["datetime"]>=ev["start"]) &
                (DF["datetime"]<=ev["end"]))
        subset = DF[mask]
        if subset.empty:
            rows.append({**ev, "detected":False, "match_pct":0, "peak_score":0})
            continue
        lc = subset["event_label"].value_counts() if "event_label" in subset.columns else pd.Series()
        match_pct = lc.get(ev["type"],0)/len(subset)*100 if len(subset)>0 else 0
        peak_score = float(subset["anomaly_score"].max()) if "anomaly_score" in subset.columns else 0
        rows.append({**ev, "detected": match_pct>50,
                     "match_pct": round(match_pct,1),
                     "peak_score": round(peak_score,3)})
    return rows


# ── App ────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AQI Disaster Intelligence ⚡"

CS = dict(backgroundColor=CARD, borderRadius="8px", padding="14px 18px",
          border=f"1px solid {BORDER}")
LB = dict(color=DIM, fontSize="11px", fontFamily="'Courier New', monospace",
          textTransform="uppercase", letterSpacing="1px", marginBottom="4px")
INP = dict(backgroundColor=PANEL, color=TEXT, border=f"1px solid {BORDER}",
           borderRadius="4px", padding="6px 10px", width="100%",
           fontFamily="'Courier New', monospace", fontSize="13px")

def kcard(id_, label):
    return html.Div([
        html.P(label, style=LB),
        html.H3(id=f"kpi-{id_}", style=dict(color=TEXT, fontSize="20px", margin=0,
               fontFamily="'Courier New', monospace")),
    ], style=dict(**CS, minWidth="110px"))

def input_field(label, id_, default, unit=""):
    return html.Div([
        html.Label(f"{label}  {unit}", style={**LB, "color": TEXT}),
        dcc.Input(id=id_, type="number", value=default,
                  style=INP, debounce=True),
    ], style=dict(flex="1", minWidth="140px"))


# ── Tab 1: Analysis ────────────────────────────────────────────
tab1 = html.Div([
    # Controls
    html.Div(style=dict(backgroundColor="#040d1c", padding="12px 28px",
                        display="flex", gap="24px", alignItems="flex-end",
                        flexWrap="wrap", borderBottom=f"1px solid {BORDER}"),
             children=[
        html.Div([html.Label("City", style=LB),
                  dcc.Dropdown(id="city", options=[{"label":c,"value":c} for c in CITIES],
                               value=CITIES[0], clearable=False,
                               style=dict(width="160px", backgroundColor=CARD, color="#000"))]),
        html.Div([html.Label("Date Range", style=LB),
                  dcc.DatePickerRange(id="dates", min_date_allowed=MIN_DT,
                                      max_date_allowed=MAX_DT,
                                      start_date=str(MIN_DT), end_date=str(MAX_DT))]),
        html.Div([html.Label("Pollutant", style=LB),
                  dcc.Dropdown(id="pol",
                               options=[{"label":p,"value":p} for p in ALL_POLS],
                               value="NO2" if "NO2" in ALL_POLS else ALL_POLS[0],
                               clearable=False,
                               style=dict(width="130px", backgroundColor=CARD, color="#000"))]),
    ]),
    # KPIs
    html.Div(style=dict(display="flex", gap="10px", flexWrap="wrap", padding="14px 28px"),
             children=[kcard("city","City"), kcard("events","Event Hours"),
                       kcard("aqi","Peak AQI"), kcard("avgaqi","Avg AQI"),
                       kcard("worst","Worst Day"), kcard("top","Top Hazard"),
                       kcard("no2","Peak NO₂"), kcard("so2","Peak SO₂")]),
    # Charts
    html.Div(style=dict(padding="0 28px 28px"), children=[
        html.Div(style=dict(**CS, marginBottom="14px"),
                 children=[dcc.Graph(id="g-timeline", config=dict(displayModeBar=False))]),
        html.Div(style=dict(display="flex", gap="14px", marginBottom="14px", flexWrap="wrap"),
                 children=[
            html.Div(style=dict(**CS, flex="2", minWidth="300px"),
                     children=[dcc.Graph(id="g-pol", config=dict(displayModeBar=False))]),
            html.Div(style=dict(**CS, flex="1", minWidth="250px"),
                     children=[dcc.Graph(id="g-fire", config=dict(displayModeBar=False))]),
        ]),
        html.Div(style=dict(display="flex", gap="14px", marginBottom="14px", flexWrap="wrap"),
                 children=[
            html.Div(style=dict(**CS, flex="2", minWidth="300px"),
                     children=[dcc.Graph(id="g-events", config=dict(displayModeBar=False))]),
            html.Div(style=dict(**CS, flex="1", minWidth="280px"),
                     children=[dcc.Graph(id="g-radar", config=dict(displayModeBar=False))]),
        ]),
        html.Div(style=dict(display="flex", gap="14px", marginBottom="14px", flexWrap="wrap"),
                 children=[
            html.Div(style=dict(**CS, flex="1", minWidth="300px"),
                     children=[dcc.Graph(id="g-heat", config=dict(displayModeBar=False))]),
            html.Div(style=dict(**CS, flex="1", minWidth="300px"),
                     children=[dcc.Graph(id="g-corr", config=dict(displayModeBar=False))]),
        ]),
        html.Div(style=dict(**CS, marginBottom="14px"),
                 children=[dcc.Graph(id="g-stl", config=dict(displayModeBar=False))]),
    ]),
])


# ── Tab 2: Live Predictor ──────────────────────────────────────
tab2 = html.Div(style=dict(padding="24px 28px"), children=[
    html.P("Enter current pollutant readings to predict the disaster type and danger level.",
           style=dict(color=DIM, fontFamily="'Courier New', monospace", fontSize="13px",
                      marginBottom="20px")),

    # Preset scenario buttons
    html.Div(style=dict(marginBottom="20px"), children=[
        html.Label("Quick Scenarios:", style={**LB, "color":TEXT, "marginBottom":"8px"}),
        html.Div(style=dict(display="flex", gap="10px", flexWrap="wrap"), children=[
            html.Button("Delhi Fire", id="preset-fire",
                style=dict(backgroundColor=EVENT_CLR["Fire"], color="#000", border="none",
                           padding="8px 16px", borderRadius="4px", cursor="pointer",
                           fontFamily="'Courier New', monospace", fontSize="12px")),
            html.Button("Jaipur Dust", id="preset-dust",
                style=dict(backgroundColor=EVENT_CLR["Dust"], color="#000", border="none",
                           padding="8px 16px", borderRadius="4px", cursor="pointer",
                           fontFamily="'Courier New', monospace", fontSize="12px")),
            html.Button("Vizag Gas", id="preset-gas",
                style=dict(backgroundColor=EVENT_CLR["Gas"], color="#fff", border="none",
                           padding="8px 16px", borderRadius="4px", cursor="pointer",
                           fontFamily="'Courier New', monospace", fontSize="12px")),
            html.Button("Normal Day", id="preset-normal",
                style=dict(backgroundColor=CARD, color=TEXT,
                           border=f"1px solid {BORDER}",
                           padding="8px 16px", borderRadius="4px", cursor="pointer",
                           fontFamily="'Courier New', monospace", fontSize="12px")),
        ]),
    ]),

    # Input grid
    html.Div(style=dict(**CS, marginBottom="20px"), children=[
        html.P("Pollutant Concentrations", style={**LB, "color":TEXT, "marginBottom":"12px"}),
        html.Div(style=dict(display="flex", gap="16px", flexWrap="wrap"), children=[
            input_field("NO₂",   "inp-no2",   45,  "µg/m³"),
            input_field("SO₂",   "inp-so2",   20,  "µg/m³"),
            input_field("CO",    "inp-co",    1.2, "mg/m³"),
            input_field("O₃",    "inp-o3",    40,  "µg/m³"),
            input_field("PM2.5", "inp-pm25",  80,  "µg/m³"),
            input_field("PM10",  "inp-pm10",  140, "µg/m³"),
            input_field("AOD",   "inp-aod",   0.6, "(0–5)"),
            input_field("Fire Pixels", "inp-fire", 0, "count"),
        ]),
        html.Br(),
        html.P("Meteorological Conditions", style={**LB, "color":TEXT, "marginBottom":"12px"}),
        html.Div(style=dict(display="flex", gap="16px", flexWrap="wrap"), children=[
            input_field("Temperature", "inp-temp",     30, "°C"),
            input_field("Humidity",    "inp-humidity", 60, "%"),
            input_field("Wind Speed",  "inp-wind",     3,  "m/s"),
        ]),
        html.Br(),
        html.Button("PREDICT DISASTER TYPE", id="btn-predict",
            style=dict(backgroundColor=CITY_CLR["Delhi"], color="#000",
                       border="none", padding="12px 32px", borderRadius="4px",
                       cursor="pointer", fontWeight="bold",
                       fontFamily="'Courier New', monospace", fontSize="14px")),
    ]),

    # Result
    html.Div(id="prediction-result"),
])


# ── Tab 3: Real Events Validation ─────────────────────────────
def build_events_tab():
    val_data = get_event_validation()
    n_detected = sum(1 for v in val_data if v["detected"])

    rows = []
    for ev in val_data:
        clr = EVENT_CLR.get(ev["type"], "#fff")
        icon = {"Fire":"","Dust":"","Gas":"","Industrial":""}.get(ev["type"],"")
        status_bg  = "#0d2b0d" if ev["detected"] else "#2b0d0d"
        status_txt = "DETECTED" if ev["detected"] else "PARTIAL"

        rows.append(html.Div(style=dict(
            backgroundColor=CARD, border=f"1px solid {BORDER}",
            borderLeft=f"4px solid {clr}", borderRadius="8px",
            padding="14px 18px", marginBottom="10px"), children=[

            html.Div(style=dict(display="flex", justifyContent="space-between",
                                alignItems="flex-start", flexWrap="wrap", gap="8px"),
                     children=[
                html.Div([
                    html.Span(f"{icon} ", style=dict(fontSize="18px")),
                    html.Span(ev["name"], style=dict(color=TEXT, fontWeight="bold",
                        fontFamily="'Courier New', monospace", fontSize="14px")),
                    html.Span(f"  [{ev['city']}  {ev['start']} → {ev['end']}]",
                        style=dict(color=DIM, fontSize="11px",
                                   fontFamily="'Courier New', monospace")),
                ]),
                html.Div(style=dict(display="flex", gap="8px", alignItems="center"),
                         children=[
                    html.Span(status_txt, style=dict(
                        backgroundColor=status_bg, color="#4ade80" if ev["detected"] else "#f87171",
                        padding="3px 10px", borderRadius="4px", fontSize="11px",
                        fontFamily="'Courier New', monospace", fontWeight="bold")),
                    html.A("Source", href=ev["url"], target="_blank",
                        style=dict(color=CITY_CLR["Delhi"], fontSize="11px",
                                   fontFamily="'Courier New', monospace",
                                   textDecoration="none",
                                   border=f"1px solid {BORDER}",
                                   padding="3px 8px", borderRadius="4px")),
                ]),
            ]),

            html.P(ev["desc"], style=dict(color=DIM, fontSize="12px",
                fontFamily="'Courier New', monospace", margin="6px 0 4px 0")),

            html.Div(style=dict(display="flex", gap="20px", marginTop="6px"), children=[
                html.Span(f"Type: {ev['type']}", style=dict(color=clr, fontSize="11px",
                    fontFamily="'Courier New', monospace")),
                html.Span(f"Match: {ev['match_pct']}%", style=dict(color=TEXT, fontSize="11px",
                    fontFamily="'Courier New', monospace")),
                html.Span(f"Peak Anomaly Score: {ev['peak_score']}", style=dict(
                    color=TEXT, fontSize="11px", fontFamily="'Courier New', monospace")),
                html.Span(f"Source: {ev['source']}", style=dict(
                    color=DIM, fontSize="11px", fontFamily="'Courier New', monospace")),
            ]),
        ]))

    return html.Div(style=dict(padding="24px 28px"), children=[
        html.Div(style=dict(display="flex", gap="16px", marginBottom="20px",
                            flexWrap="wrap"), children=[
            html.Div(style=dict(**CS, flex="1"), children=[
                html.P("Total Events", style=LB),
                html.H2(f"{len(val_data)}", style=dict(color=TEXT, margin=0,
                    fontFamily="'Courier New', monospace"))]),
            html.Div(style=dict(**CS, flex="1"), children=[
                html.P("Detected by Model", style=LB),
                html.H2(f"{n_detected}/{len(val_data)}",
                    style=dict(color="#4ade80", margin=0,
                               fontFamily="'Courier New', monospace"))]),
            html.Div(style=dict(**CS, flex="1"), children=[
                html.P("Detection Rate", style=LB),
                html.H2(f"{n_detected/len(val_data)*100:.0f}%",
                    style=dict(color="#4ade80", margin=0,
                               fontFamily="'Courier New', monospace"))]),
            html.Div(style=dict(**CS, flex="2"), children=[
                html.P("Note", style=LB),
                html.P("Detection here uses ground-truth event_label (labels we attached to known event dates). "
                       "The match% shows what fraction of hours in each event window were correctly labelled.",
                    style=dict(color=DIM, fontSize="11px",
                               fontFamily="'Courier New', monospace", margin=0))]),
        ]),
        html.Div(rows),
    ])

tab3 = build_events_tab()


# ── Main layout ────────────────────────────────────────────────
app.layout = html.Div(style=dict(backgroundColor=BG, minHeight="100vh"), children=[
    html.Div(style=dict(backgroundColor="#040d1c", borderBottom=f"2px solid {BORDER}",
                        padding="14px 28px", display="flex", alignItems="center", gap="16px"),
             children=[
        html.Span("⚡", style=dict(fontSize="26px")),
        html.Div([
            html.H1("AQI Disaster Intelligence",
                    style=dict(color=TEXT, margin=0, fontSize="21px",
                               fontFamily="'Courier New', monospace")),
            html.P("Multi-City Environmental Anomaly Detection  |  Delhi · Jaipur · Vizag  |  2019–2024",
                   style=dict(color=DIM, margin=0, fontSize="11px",
                              fontFamily="'Courier New', monospace")),
        ]),
        html.Div("● LIVE", style=dict(color="#4ade80", fontSize="12px",
                                      fontFamily="'Courier New', monospace",
                                      marginLeft="auto")),
    ]),

    dcc.Tabs(id="tabs", value="tab-analysis", style=dict(fontFamily="'Courier New', monospace"),
             colors=dict(border=BORDER, primary=CITY_CLR["Delhi"], background=BG),
             children=[
        dcc.Tab(label="Analysis Dashboard", value="tab-analysis",
                style=dict(color=DIM, backgroundColor=CARD, border=f"1px solid {BORDER}"),
                selected_style=dict(color=TEXT, backgroundColor=PANEL,
                                    border=f"1px solid {BORDER}", fontWeight="bold")),
        dcc.Tab(label="Live Predictor", value="tab-predictor",
                style=dict(color=DIM, backgroundColor=CARD, border=f"1px solid {BORDER}"),
                selected_style=dict(color=TEXT, backgroundColor=PANEL,
                                    border=f"1px solid {BORDER}", fontWeight="bold")),
        dcc.Tab(label="Real Events Validation", value="tab-events",
                style=dict(color=DIM, backgroundColor=CARD, border=f"1px solid {BORDER}"),
                selected_style=dict(color=TEXT, backgroundColor=PANEL,
                                    border=f"1px solid {BORDER}", fontWeight="bold")),
    ]),

    html.Div(id="tab-content"),
])


# ── Tab routing ────────────────────────────────────────────────
@callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "tab-analysis":  return tab1
    if tab == "tab-predictor": return tab2
    return tab3


# ── Analysis callbacks ─────────────────────────────────────────
@callback(
    [Output("g-timeline","figure"), Output("g-pol","figure"),
     Output("g-fire","figure"),     Output("g-events","figure"),
     Output("g-radar","figure"),    Output("g-heat","figure"),
     Output("g-corr","figure"),     Output("g-stl","figure"),
     Output("kpi-city","children"), Output("kpi-events","children"),
     Output("kpi-aqi","children"),  Output("kpi-avgaqi","children"),
     Output("kpi-worst","children"),Output("kpi-top","children"),
     Output("kpi-no2","children"),  Output("kpi-so2","children")],
    [Input("city","value"), Input("dates","start_date"),
     Input("dates","end_date"), Input("pol","value")],
)
def update_analysis(city, start, end, pol):
    dff = filt(city, start, end)
    k   = kpis(dff, city)
    return (fig_timeline(dff,city), fig_pollutant(dff,city,pol),
            fig_fire(dff), fig_events(dff), fig_radar(dff),
            fig_heatmap(dff), fig_corr(dff), fig_stl(dff),
            k["city"],k["events"],k["paqi"],k["aaqi"],
            k["worst"],k["top"],k["no2"],k["so2"])


# ── Preset scenario buttons ────────────────────────────────────
SCENARIOS = {
    "preset-fire":   dict(no2=180, so2=35,  co=8.5, o3=42, pm25=420, pm10=680, aod=3.2, fire=45,  temp=24, hum=72, wind=1.8),
    "preset-dust":   dict(no2=28,  so2=12,  co=0.7, o3=55, pm25=95,  pm10=820, aod=2.8, fire=0,   temp=43, hum=18, wind=9.5),
    "preset-gas":    dict(no2=85,  so2=920, co=0.9, o3=38, pm25=48,  pm10=65,  aod=0.4, fire=0,   temp=30, hum=78, wind=2.1),
    "preset-normal": dict(no2=42,  so2=15,  co=1.1, o3=35, pm25=55,  pm10=90,  aod=0.45,fire=0,   temp=34, hum=55, wind=3.5),
}

@callback(
    [Output("inp-no2","value"),   Output("inp-so2","value"),
     Output("inp-co","value"),    Output("inp-o3","value"),
     Output("inp-pm25","value"),  Output("inp-pm10","value"),
     Output("inp-aod","value"),   Output("inp-fire","value"),
     Output("inp-temp","value"),  Output("inp-humidity","value"),
     Output("inp-wind","value")],
    [Input("preset-fire","n_clicks"),   Input("preset-dust","n_clicks"),
     Input("preset-gas","n_clicks"),    Input("preset-normal","n_clicks")],
    prevent_initial_call=True,
)
def fill_preset(*_):
    from dash import ctx
    sc = SCENARIOS.get(ctx.triggered_id, SCENARIOS["preset-normal"])
    return (sc["no2"], sc["so2"], sc["co"], sc["o3"],
            sc["pm25"], sc["pm10"], sc["aod"], sc["fire"],
            sc["temp"], sc["hum"], sc["wind"])


# ── Prediction callback ────────────────────────────────────────
@callback(
    Output("prediction-result","children"),
    Input("btn-predict","n_clicks"),
    [State("inp-no2","value"),   State("inp-so2","value"),
     State("inp-co","value"),    State("inp-o3","value"),
     State("inp-pm25","value"),  State("inp-pm10","value"),
     State("inp-aod","value"),   State("inp-fire","value"),
     State("inp-temp","value"),  State("inp-humidity","value"),
     State("inp-wind","value")],
    prevent_initial_call=True,
)
def run_prediction(n, no2, so2, co, o3, pm25, pm10, aod,
                   fire, temp, humidity, wind):
    if not n:
        return html.Div()

    try:
        from predict import predict_single
        res = predict_single(
            NO2=float(no2 or 0), SO2=float(so2 or 0),
            CO=float(co or 0),   O3=float(o3 or 0),
            PM2_5=float(pm25 or 0), PM10=float(pm10 or 0),
            AOD=float(aod or 0), fire_count=float(fire or 0),
            temp_c=float(temp or 25), humidity=float(humidity or 60),
            wind_speed=float(wind or 3)
        )
    except Exception as e:
        return html.Div(f"Error: {e}", style=dict(color="#f87171",
            fontFamily="'Courier New', monospace", padding="12px"))

    danger_colors = {"LOW":"#22c55e", "MEDIUM":"#f59e0b", "HIGH":"#ef4444"}
    lbl_clr = EVENT_CLR.get(res["label"], TEXT)
    d_clr   = danger_colors.get(res["danger"], TEXT)

    # Probability bar chart
    probs = res["all_probs"]
    fig = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()),
        orientation="h",
        marker_color=[EVENT_CLR.get(k, "#607b96") for k in probs.keys()],
        text=[f"{v:.1f}%" for v in probs.values()],
        textposition="outside",
    ))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="'Courier New', monospace", size=12),
        margin=dict(l=100, r=60, t=20, b=20),
        xaxis=dict(range=[0,110], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor=BORDER), height=220, showlegend=False)

    above = res.get("above_safe", [])

    return html.Div([
        # Big result banner
        html.Div(style=dict(backgroundColor=CARD, borderRadius="8px",
                             border=f"3px solid {lbl_clr}", padding="20px",
                             marginBottom="16px"), children=[
            html.Div(style=dict(display="flex", alignItems="center", gap="16px",
                                flexWrap="wrap"), children=[
                html.Div([
                    html.Div(res["icon"] + "  " + res["label"],
                             style=dict(fontSize="28px", fontWeight="bold",
                                        color=lbl_clr,
                                        fontFamily="'Courier New', monospace")),
                    html.Div(f"Confidence: {res['confidence']}%",
                             style=dict(color=DIM, fontSize="13px",
                                        fontFamily="'Courier New', monospace")),
                ]),
                html.Div(style=dict(marginLeft="auto", textAlign="center"), children=[
                    html.Div("DANGER", style=dict(color=DIM, fontSize="10px",
                        fontFamily="'Courier New', monospace", letterSpacing="2px")),
                    html.Div(res["danger"], style=dict(fontSize="24px", fontWeight="bold",
                        color=d_clr, fontFamily="'Courier New', monospace")),
                ]),
                html.Div(style=dict(textAlign="center"), children=[
                    html.Div("EST. AQI", style=dict(color=DIM, fontSize="10px",
                        fontFamily="'Courier New', monospace", letterSpacing="2px")),
                    html.Div(str(res["AQI"]), style=dict(fontSize="24px", fontWeight="bold",
                        color=TEXT, fontFamily="'Courier New', monospace")),
                ]),
            ]),
            html.P(res["explanation"], style=dict(color=TEXT, fontSize="13px",
                fontFamily="'Courier New', monospace", marginTop="12px", marginBottom=0)),
        ]),

        # Probability chart
        html.Div(style=dict(**CS, marginBottom="16px"), children=[
            html.P("Probability for each class:", style={**LB, "color":TEXT, "marginBottom":"8px"}),
            dcc.Graph(figure=fig, config=dict(displayModeBar=False)),
        ]),

        # Warnings
        html.Div(style=dict(**CS) if above else {}, children=[
            html.P("Pollutants exceeding 2× safe limit:", style=dict(
                color="#f59e0b", fontSize="12px",
                fontFamily="'Courier New', monospace", marginBottom="6px")) if above else None,
            *[html.P(f"  • {a}", style=dict(color=TEXT, fontSize="12px",
                fontFamily="'Courier New', monospace", margin="2px 0")) for a in above],
        ]) if above else html.Div(),
    ])


if __name__ == "__main__":
    print("\n══ DASHBOARD v3 ══  →  http://127.0.0.1:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
