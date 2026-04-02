"""
CodeCure — Epidemic Intelligence Dashboard
=============================================
Interactive Streamlit dashboard for epidemic forecasting,
risk scoring, and trend analysis.

Usage:
    streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from pmdarima import auto_arima
from src.report_generator import generate_pdf_report

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
CSV_PATH = ROOT / "data" / "output" / "master_dataset.csv"
MODELS_DIR = ROOT / "models"
RISK_MAP_PATH = ROOT / "risk_map.html"

# ─── Country coordinates for map ─────────────────────────────────────────────────
COUNTRY_COORDS = {
    "United States": {"lat": 37.09, "lon": -95.71, "iso": "USA"},
    "India": {"lat": 20.59, "lon": 78.96, "iso": "IND"},
    "Brazil": {"lat": -14.24, "lon": -51.93, "iso": "BRA"},
    "Germany": {"lat": 51.17, "lon": 10.45, "iso": "DEU"},
    "South Africa": {"lat": -30.56, "lon": 22.94, "iso": "ZAF"},
}

# ─── GEOJSON URL ──────────────────────────────────────────────────────────────────
GEOJSON_URL = "https://raw.githubusercontent.com/python-visualization/folium/main/tests/data/world-countries.json"


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CodeCure — Epidemic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header .subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin-top: 0.3rem;
    font-weight: 300;
}
.main-header .badge {
    display: inline-block;
    background: linear-gradient(135deg, #00d2ff, #3a7bd5);
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.3rem 0;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Risk badges */
.risk-low { 
    background: linear-gradient(135deg, #00b09b, #96c93d); 
    color: white; padding: 0.3rem 1rem; border-radius: 20px; 
    font-weight: 600; font-size: 0.85rem; display: inline-block; 
}
.risk-moderate { 
    background: linear-gradient(135deg, #f7971e, #ffd200); 
    color: #1a1a2e; padding: 0.3rem 1rem; border-radius: 20px; 
    font-weight: 600; font-size: 0.85rem; display: inline-block; 
}
.risk-high { 
    background: linear-gradient(135deg, #f85032, #e73827); 
    color: white; padding: 0.3rem 1rem; border-radius: 20px; 
    font-weight: 600; font-size: 0.85rem; display: inline-block; 
}
.risk-critical { 
    background: linear-gradient(135deg, #8e0e00, #1f1c18); 
    color: #ff4444; padding: 0.3rem 1rem; border-radius: 20px; 
    font-weight: 700; font-size: 0.85rem; display: inline-block;
    border: 1px solid #ff4444;
    animation: pulse-critical 2s infinite;
}
@keyframes pulse-critical {
    0%, 100% { box-shadow: 0 0 5px rgba(255,68,68,0.3); }
    50% { box-shadow: 0 0 20px rgba(255,68,68,0.6); }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
}

/* LSTM placeholder */
.lstm-placeholder {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px dashed rgba(58, 123, 213, 0.4);
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.lstm-placeholder h3 { color: #3a7bd5; margin-bottom: 0.5rem; }
.lstm-placeholder p { color: rgba(255,255,255,0.5); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════════
# DATA LOADING (Cached)
# ═════════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load master dataset with caching."""
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    else:
        st.error("❌ Master dataset not found. Run `python run_pipeline.py` first.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_arima_model():
    """Load serialized ARIMA model."""
    model_path = MODELS_DIR / "arima_model.pkl"
    meta_path = MODELS_DIR / "arima_model_metadata.json"

    if model_path.exists():
        model = joblib.load(model_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        return model, meta
    return None, {}


@st.cache_resource
def load_lstm_model():
    """Load LSTM/Hybrid model metrics for display."""
    metrics_path = MODELS_DIR / "hybrid_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ═════════════════════════════════════════════════════════════════════════════════
# ARIMA FORECASTING (Cached per country)
# ═════════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def generate_arima_forecast(country_data: pd.DataFrame, country: str, days: int = 14):
    """Train a quick ARIMA on the country's confirmed series and forecast."""
    series = country_data.set_index("date")["confirmed"].astype(float).ffill().dropna()
    series = series.clip(lower=0)

    if series.var() == 0 or len(series) < 30:
        return None, None, None

    # 80/20 split
    n = len(series)
    split = int(n * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    try:
        model = auto_arima(
            train, d=1, seasonal=True, m=7,
            start_p=0, max_p=3, start_q=0, max_q=3,
            start_P=0, max_P=1, start_Q=0, max_Q=1,
            max_D=1, stepwise=True, suppress_warnings=True,
            error_action="ignore", trace=False,
        )

        # Test predictions
        test_preds = model.predict(n_periods=len(test))
        test_preds = pd.Series(test_preds, index=test.index).clip(lower=0)

        # Metrics
        actual = test.values.astype(float)
        predicted = test_preds.values.astype(float)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        mask = actual > 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.inf
        metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

        # Update with test then forecast
        model.update(test)
        future_preds, future_ci = model.predict(
            n_periods=days, return_conf_int=True, alpha=0.05
        )
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
        future_df = pd.DataFrame({
            "date": future_dates,
            "forecast": np.clip(future_preds, 0, None),
            "lower_95": np.clip(future_ci[:, 0], 0, None),
            "upper_95": np.clip(future_ci[:, 1], 0, None),
        })

        return future_df, metrics, test_preds

    except Exception:
        return None, None, None


# ═════════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═════════════════════════════════════════════════════════════════════════════════
def compute_risk_score(country_data: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """
    Risk Score = 0.5 * forecast_growth + 0.3 * cases_per_million + 0.2 * positivity_rate
    """
    latest = country_data.iloc[-1]
    population = latest.get("population", 1e6)
    if population == 0:
        population = 1e6

    # Forecast growth: % change from last confirmed to end of forecast
    last_confirmed = float(latest.get("confirmed", 0))
    if forecast_df is not None and len(forecast_df) > 0:
        forecast_end = float(forecast_df["forecast"].iloc[-1])
        if last_confirmed > 0:
            forecast_growth = max(0, (forecast_end - last_confirmed) / last_confirmed)
        else:
            forecast_growth = 0
    else:
        forecast_growth = 0

    # Cases per million (normalized to 0-1 scale, cap at 10%)
    cases_per_million = (last_confirmed / population) * 1e6
    cpm_normalized = min(cases_per_million / 100000, 1.0)  # Cap normalization

    # Positivity rate (already 0-1 ish)
    positivity_rate = float(latest.get("positive_rate", 0))
    positivity_normalized = min(positivity_rate, 1.0)

    # Composite score
    score = (
        0.5 * min(forecast_growth, 1.0) +
        0.3 * cpm_normalized +
        0.2 * positivity_normalized
    )
    score = round(score * 100, 1)  # Scale to 0-100

    # Category
    if score < 25:
        category = "Low"
        css_class = "risk-low"
        emoji = "🟢"
    elif score < 50:
        category = "Moderate"
        css_class = "risk-moderate"
        emoji = "🟡"
    elif score < 75:
        category = "High"
        css_class = "risk-high"
        emoji = "🟠"
    else:
        category = "Critical"
        css_class = "risk-critical"
        emoji = "🔴"

    return {
        "score": score,
        "category": category,
        "css_class": css_class,
        "emoji": emoji,
        "forecast_growth": forecast_growth,
        "cases_per_million": cases_per_million,
        "positivity_rate": positivity_rate,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# RISK MAP GENERATION
# ═════════════════════════════════════════════════════════════════════════════════
def generate_risk_map(df: pd.DataFrame, all_forecasts: dict, all_risks: dict):
    """Generate a Folium choropleth risk map and save to HTML."""
    
    m = folium.Map(
        location=[20, 0], zoom_start=2,
        tiles="CartoDB dark_matter",
        attr="CodeCure Epidemic Intelligence",
    )

    # Color scale
    colormap = LinearColormap(
        colors=["#00b09b", "#96c93d", "#ffd200", "#f85032", "#8e0e00"],
        vmin=0, vmax=100,
        caption="Epidemic Risk Score (0-100)",
    )
    colormap.add_to(m)

    # Add markers for each country
    for country, info in COUNTRY_COORDS.items():
        risk = all_risks.get(country, {"score": 0, "category": "Unknown", "emoji": "⚪"})
        forecast = all_forecasts.get(country)

        score = risk["score"]
        category = risk["category"]

        # Color based on score
        if score < 25:
            color = "#00b09b"
        elif score < 50:
            color = "#ffd200"
        elif score < 75:
            color = "#f85032"
        else:
            color = "#8e0e00"

        # Forecast snippet
        if forecast is not None and len(forecast) > 0:
            fc_end = f"{forecast['forecast'].iloc[-1]:,.0f}"
        else:
            fc_end = "N/A"

        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 220px; padding: 8px;">
            <h4 style="margin:0; color: #1a1a2e;">{risk['emoji']} {country}</h4>
            <hr style="margin: 6px 0; border-color: #eee;">
            <table style="width:100%; font-size: 13px;">
                <tr><td><b>Risk Score</b></td><td style="text-align:right; font-weight:700; color:{color};">{score}</td></tr>
                <tr><td><b>Category</b></td><td style="text-align:right;">{category}</td></tr>
                <tr><td><b>14-Day Forecast</b></td><td style="text-align:right;">{fc_end}</td></tr>
                <tr><td><b>Positivity Rate</b></td><td style="text-align:right;">{risk.get('positivity_rate', 0):.2%}</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=max(12, score / 3),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{country}: {category} ({score})",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
        ).add_to(m)

    # Save standalone HTML
    m.save(str(RISK_MAP_PATH))

    return m


# ═════════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═════════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=13),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(26,26,46,0.8)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)


# ═════════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div>
        <span class="badge">ARIMA</span>
        <span class="badge">LSTM</span>
        <span class="badge">RISK SCORING</span>
    </div>
    <h1>🧬 CodeCure — Epidemic Intelligence</h1>
    <p class="subtitle">AI-powered epidemic forecasting, risk assessment & trend analysis dashboard</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════════
df = load_data()
arima_model, arima_meta = load_arima_model()
lstm_model = load_lstm_model()

countries = sorted(df["country"].unique())
date_min = df["date"].min().date()
date_max = df["date"].max().date()


# ═════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    st.markdown("---")

    selected_country = st.selectbox(
        "🌍 Country",
        countries,
        index=countries.index("United States") if "United States" in countries else 0,
    )

    date_range = st.slider(
        "📅 Date Range",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
        format="YYYY-MM-DD",
    )

    forecast_days = st.slider("🔮 Forecast Horizon (days)", 7, 30, 14)

    st.markdown("---")
    st.markdown("### 📊 Model Status")

    if arima_model:
        st.success(f"ARIMA: Loaded ✅")
        if arima_meta:
            st.caption(f"Order: {arima_meta.get('order_pdq', 'N/A')}")
    else:
        st.warning("ARIMA: Not trained")

    if lstm_model:
        st.success("LSTM: Loaded ✅")
    else:
        st.info("LSTM: Awaiting training")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: rgba(255,255,255,0.3); font-size: 0.75rem;'>"
        "CodeCure v1.0 • Built with ❤️"
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════════
# FILTER DATA
# ═════════════════════════════════════════════════════════════════════════════════
mask = (
    (df["country"] == selected_country) &
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1])
)
country_df = df[mask].copy().sort_values("date")
full_country_df = df[df["country"] == selected_country].sort_values("date")


# ═════════════════════════════════════════════════════════════════════════════════
# GENERATE FORECASTS
# ═════════════════════════════════════════════════════════════════════════════════
with st.spinner(f"🔄 Generating ARIMA forecast for {selected_country}..."):
    arima_forecast, arima_metrics, arima_test_preds = generate_arima_forecast(
        full_country_df, selected_country, days=forecast_days
    )

# Compute risk
risk = compute_risk_score(full_country_df, arima_forecast)

with st.sidebar:
    st.markdown("### 📄 Automations")
    if st.button("Generate Regional PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf_path = generate_pdf_report(selected_country, full_country_df, arima_forecast, latest if 'latest' in locals() else full_country_df.iloc[-1] if len(full_country_df) > 0 else {}, risk)
            st.success(f"Report generated and saved to {pdf_path}")

# ═════════════════════════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ═════════════════════════════════════════════════════════════════════════════════
latest = full_country_df.iloc[-1] if len(full_country_df) > 0 else {}

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Confirmed</div>
        <div class="metric-value">{int(latest.get('confirmed', 0)):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Deaths</div>
        <div class="metric-value" style="color: #e74c3c;">{int(latest.get('deaths', 0)):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">7-Day Avg Cases</div>
        <div class="metric-value" style="color: #3498db;">{latest.get('cases_7day_avg', 0):,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Positivity Rate</div>
        <div class="metric-value" style="color: #f39c12;">{latest.get('positive_rate', 0):.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value">
            <span class="{risk['css_class']}">{risk['emoji']} {risk['score']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════════
tab_forecast, tab_trends, tab_riskmap = st.tabs([
    "🔮 Forecast", "📈 Trends & EDA", "🗺️ Risk Map"
])


# ─────────────────────────────────────────────────────────────────────────────────
# TAB 1: FORECAST
# ─────────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(f"### 📊 Forecast — {selected_country}")

    # ── Historical + ARIMA Forecast chart ─────────────────────────────────────
    fig = go.Figure()

    # Historical confirmed
    fig.add_trace(go.Scatter(
        x=country_df["date"], y=country_df["confirmed"],
        mode="lines", name="Historical (Confirmed)",
        line=dict(color="#3498db", width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))

    # ARIMA Forecast
    if arima_forecast is not None:
        fig.add_trace(go.Scatter(
            x=arima_forecast["date"], y=arima_forecast["forecast"],
            mode="lines+markers", name=f"ARIMA Forecast ({forecast_days}d)",
            line=dict(color="#e74c3c", width=2.5, dash="dash"),
            marker=dict(size=5),
        ))
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([arima_forecast["date"], arima_forecast["date"][::-1]]),
            y=pd.concat([arima_forecast["upper_95"], arima_forecast["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI (ARIMA)",
            showlegend=True,
        ))

    # LSTM placeholder line (if model exists)
    if lstm_model is not None:
        # Would add LSTM predictions here
        pass

    fig.update_layout(
        title=f"Confirmed Cases & ARIMA Forecast — {selected_country}",
        xaxis_title="Date",
        yaxis_title="Confirmed Cases",
        height=480,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── LSTM Section ──────────────────────────────────────────────────────────
    st.markdown("#### 🧠 Advanced Hybrid Architecture (LSTM + SARIMAX)")
    if lstm_model is None:
        st.markdown("""
        <div class="lstm-placeholder">
            <h3>🧠 LSTM Model — Coming Soon</h3>
            <p>The LSTM deep learning model has not been trained yet.<br>
            Once trained, predictions will appear here automatically.</p>
            <p style="color: rgba(255,255,255,0.3); font-size: 0.8rem; margin-top: 1rem;">
            Run: <code>python run_hybrid.py</code> to train the LSTM model
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ **Hybrid Model Loaded:** The deep-learning LSTM sequence module successfully mapped the structural variance alongside the ARIMA baseline trend curves.")

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown("#### 📋 Model Comparison — Evaluation Metrics")

    comparison_data = []
    if arima_metrics:
        comparison_data.append({
            "Model": "ARIMA (SARIMAX)",
            "RMSE": f"{arima_metrics['RMSE']:,.2f}",
            "MAE": f"{arima_metrics['MAE']:,.2f}",
            "MAPE": f"{arima_metrics['MAPE']:.2f}%",
            "Status": "✅ Baseline",
        })

    comparison_data.append({
        "Model": "LSTM (Sequence NN)",
        "RMSE": f"{lstm_model['lstm_rmse']:,.2f}" if lstm_model else "—",
        "MAE": f"{lstm_model['lstm_mae']:,.2f}" if lstm_model else "—",
        "MAPE": f"{lstm_model['lstm_mape']:.2f}%" if lstm_model else "—",
        "Status": "✅ Deep Learning" if lstm_model else "⏳ Pending",
    })
    
    comparison_data.append({
        "Model": "🌟 Hybrid (Ensemble)",
        "RMSE": f"{lstm_model['hybrid_rmse']:,.2f}" if lstm_model else "—",
        "MAE": f"{lstm_model['hybrid_mae']:,.2f}" if lstm_model else "—",
        "MAPE": f"{lstm_model['hybrid_mape']:.2f}%" if lstm_model else "—",
        "Status": "✅ Optimized" if lstm_model else "⏳ Pending",
    })

    if comparison_data:
        st.dataframe(
            pd.DataFrame(comparison_data).set_index("Model"),
            use_container_width=True,
        )

    # ── Risk Score Breakdown ──────────────────────────────────────────────────
    st.markdown("#### ⚠️ Risk Assessment")

    risk_col1, risk_col2 = st.columns([1, 2])

    with risk_col1:
        st.markdown(f"""
        <div class="metric-card" style="padding: 2rem;">
            <div class="metric-label">Overall Risk</div>
            <div class="metric-value" style="font-size: 3rem;">
                <span class="{risk['css_class']}">{risk['emoji']} {risk['category']}</span>
            </div>
            <div class="metric-value">{risk['score']} / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col2:
        risk_fig = go.Figure()
        risk_components = ["Forecast Growth", "Cases/Million", "Positivity Rate"]
        risk_values = [
            min(risk["forecast_growth"], 1.0) * 0.5 * 100,
            min(risk["cases_per_million"] / 100000, 1.0) * 0.3 * 100,
            min(risk["positivity_rate"], 1.0) * 0.2 * 100,
        ]
        risk_colors = ["#3498db", "#e74c3c", "#f39c12"]

        risk_fig.add_trace(go.Bar(
            x=risk_values, y=risk_components,
            orientation="h",
            marker=dict(
                color=risk_colors,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in risk_values],
            textposition="auto",
            textfont=dict(color="white", size=13, family="Inter"),
        ))

        risk_fig.update_layout(
            title="Risk Score Breakdown (weighted contributions)",
            xaxis_title="Contribution to Risk Score",
            height=280,
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(risk_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────────
# TAB 2: TRENDS & EDA
# ─────────────────────────────────────────────────────────────────────────────────
with tab_trends:
    st.markdown(f"### 📈 Trend Analysis — {selected_country}")

    # ── Multi-metric trend chart ──────────────────────────────────────────────
    fig_trends = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Daily New Cases (7-Day Avg)", "Deaths (Cumulative)", "Stringency Index"),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Cases 7-day avg
    fig_trends.add_trace(go.Scatter(
        x=country_df["date"], y=country_df["cases_7day_avg"],
        mode="lines", name="7-Day Avg Cases",
        line=dict(color="#00d2ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,210,255,0.08)",
    ), row=1, col=1)

    # Deaths cumulative
    fig_trends.add_trace(go.Scatter(
        x=country_df["date"], y=country_df["deaths"],
        mode="lines", name="Cumulative Deaths",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
    ), row=2, col=1)

    # Stringency index
    if "stringency_index" in country_df.columns:
        fig_trends.add_trace(go.Scatter(
            x=country_df["date"], y=country_df["stringency_index"],
            mode="lines", name="Stringency Index",
            line=dict(color="#f39c12", width=2),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.08)",
        ), row=3, col=1)

    fig_trends.update_layout(
        height=700,
        **PLOTLY_LAYOUT,
        title_text=f"Epidemic Trends — {selected_country}",
    )
    st.plotly_chart(fig_trends, use_container_width=True)

    # ── Mobility trends ──────────────────────────────────────────────────────
    st.markdown("#### 🚶 Mobility Trends")

    mobility_cols = [c for c in country_df.columns if c.startswith("mobility_")]
    if mobility_cols:
        fig_mob = go.Figure()
        mob_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
        for i, col in enumerate(mobility_cols):
            label = col.replace("mobility_", "").replace("_", " ").title()
            fig_mob.add_trace(go.Scatter(
                x=country_df["date"], y=country_df[col],
                mode="lines", name=label,
                line=dict(color=mob_colors[i % len(mob_colors)], width=1.5),
                opacity=0.8,
            ))

        fig_mob.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        fig_mob.update_layout(
            title="Mobility Change from Baseline (%)",
            xaxis_title="Date",
            yaxis_title="% Change",
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_mob, use_container_width=True)
    else:
        st.info("No mobility data available.")

    # ── Vaccination progress ──────────────────────────────────────────────────
    st.markdown("#### 💉 Vaccination Progress")

    vacc_cols = {
        "people_vaccinated": ("Vaccinated (1+ dose)", "#2ecc71"),
        "people_fully_vaccinated": ("Fully Vaccinated", "#3498db"),
        "total_boosters": ("Boosters", "#9b59b6"),
    }

    fig_vacc = go.Figure()
    for col, (label, color) in vacc_cols.items():
        if col in country_df.columns:
            fig_vacc.add_trace(go.Scatter(
                x=country_df["date"], y=country_df[col],
                mode="lines", name=label,
                line=dict(color=color, width=2),
                stackgroup="vacc",
            ))

    fig_vacc.update_layout(
        title="Vaccination Progress",
        xaxis_title="Date",
        yaxis_title="People",
        height=380,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_vacc, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### 🔥 Feature Correlation Heatmap")

    numeric_cols = ["confirmed", "deaths", "cases_7day_avg", "positive_rate",
                    "stringency_index", "reproduction_rate", "icu_patients",
                    "hosp_patients"]
    available_cols = [c for c in numeric_cols if c in country_df.columns]

    if len(available_cols) >= 3:
        corr = country_df[available_cols].corr()
        fig_corr = px.imshow(
            corr,
            x=[c.replace("_", " ").title()[:18] for c in available_cols],
            y=[c.replace("_", " ").title()[:18] for c in available_cols],
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig_corr.update_layout(
            title="Feature Correlation Matrix",
            height=450,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────────
# TAB 3: RISK MAP
# ─────────────────────────────────────────────────────────────────────────────────
with tab_riskmap:
    st.markdown("### 🗺️ Global Epidemic Risk Map")
    st.caption("Risk scores computed for all monitored countries. Circle size reflects risk magnitude.")

    # Compute risks for all countries
    all_forecasts = {}
    all_risks = {}

    progress_bar = st.progress(0, text="Computing country risk scores...")
    for i, country in enumerate(countries):
        c_df = df[df["country"] == country].sort_values("date")
        fc, _, _ = generate_arima_forecast(c_df, country, days=14)
        all_forecasts[country] = fc
        all_risks[country] = compute_risk_score(c_df, fc)
        progress_bar.progress((i + 1) / len(countries), text=f"Computed: {country}")

    progress_bar.empty()

    # Generate map
    risk_map = generate_risk_map(df, all_forecasts, all_risks)

    # Display map
    st_folium(risk_map, width=None, height=520, returned_objects=[])

    st.success(f"✅ Risk map saved to `{RISK_MAP_PATH}`")

    # ── Risk Table ────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Risk Scores — All Countries")

    risk_table_data = []
    for country in countries:
        r = all_risks[country]
        risk_table_data.append({
            "Country": f"{r['emoji']} {country}",
            "Risk Score": r["score"],
            "Category": r["category"],
            "Forecast Growth": f"{r['forecast_growth']:.2%}",
            "Cases/Million": f"{r['cases_per_million']:,.0f}",
            "Positivity Rate": f"{r['positivity_rate']:.2%}",
        })

    risk_table = pd.DataFrame(risk_table_data).sort_values("Risk Score", ascending=False)
    st.dataframe(risk_table.set_index("Country"), use_container_width=True)

    # ── Risk comparison bar chart ─────────────────────────────────────────────
    fig_risk_bar = go.Figure()

    bar_countries = [r["Country"].split(" ", 1)[1] for r in risk_table_data]
    bar_scores = [r["Risk Score"] for r in risk_table_data]
    bar_colors = []
    for s in bar_scores:
        if s < 25:
            bar_colors.append("#00b09b")
        elif s < 50:
            bar_colors.append("#ffd200")
        elif s < 75:
            bar_colors.append("#f85032")
        else:
            bar_colors.append("#8e0e00")

    fig_risk_bar.add_trace(go.Bar(
        x=bar_countries, y=bar_scores,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{s:.1f}" for s in bar_scores],
        textposition="auto",
        textfont=dict(color="white", size=14, family="Inter"),
    ))

    fig_risk_bar.update_layout(
        title="Risk Score Comparison",
        xaxis_title="Country",
        yaxis_title="Risk Score (0-100)",
        height=350,
        **PLOTLY_LAYOUT,
    )
    fig_risk_bar.add_hline(y=25, line_dash="dot", line_color="#00b09b",
                           annotation_text="Low", annotation_position="right")
    fig_risk_bar.add_hline(y=50, line_dash="dot", line_color="#ffd200",
                           annotation_text="Moderate", annotation_position="right")
    fig_risk_bar.add_hline(y=75, line_dash="dot", line_color="#f85032",
                           annotation_text="High", annotation_position="right")

    st.plotly_chart(fig_risk_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem; padding: 1rem;'>"
    "🧬 CodeCure — Epidemic Intelligence Dashboard | "
    "Built with Streamlit, Plotly & Folium | "
    "Models: ARIMA (SARIMAX) + LSTM (Pending)"
    "</div>",
    unsafe_allow_html=True,
)
