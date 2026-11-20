"""A small interactive dashboard for the AQI prediction project."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "air_quality_ml_ready.csv"
MODEL_PATH = ROOT / "models" / "hourly_aqi_model.joblib"
PREDICTIONS_PATH = ROOT / "models" / "test_predictions.csv"
REPORT_PATH = ROOT / "models" / "metrics_report.json"

PAPER = "#f4f0e7"
INK = "#292820"
MUTED = "#716e62"
RUST = "#b4543b"
OLIVE = "#71794f"
GOLD = "#c59643"
LINE = "#ddd6c8"

st.set_page_config(
    page_title="AQI Predictor | Chandni Chowk",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: light; }}
    .stApp {{ background: {PAPER}; color: {INK}; }}
    [data-testid="stHeader"] {{ background: rgba(244, 240, 231, 0.92); }}
    .block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
    h1, h2, h3 {{ color: {INK}; letter-spacing: -0.035em; }}
    h1 {{ font-size: clamp(2.6rem, 6vw, 4.8rem) !important; line-height: .98 !important; }}
    p, li, label {{ color: {INK}; }}
    .eyebrow {{ color: {RUST}; font-size: .78rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }}
    .lede {{ max-width: 720px; color: {MUTED}; font-size: 1.12rem; line-height: 1.65; }}
    .note {{ border-left: 3px solid {GOLD}; padding: .8rem 1rem; background: #ebe5d8; border-radius: 0 9px 9px 0; color: {INK}; }}
    [data-testid="stMetric"] {{ background: #fbf8f1; border: 1px solid {LINE}; border-radius: 13px; padding: 1rem 1.1rem; }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .85rem; }}
    [data-testid="stMetricValue"] {{ color: {INK}; }}
    [data-testid="stTabs"] button {{ color: {MUTED}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {RUST}; }}
    .stButton button {{ background: {RUST}; color: white; border: 0; border-radius: 9px; }}
    .stButton button:hover {{ background: #95432f; color: white; border: 0; }}
    hr {{ border-color: {LINE}; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_report(path: str) -> dict:
    with open(path, encoding="utf-8") as report_file:
        return json.load(report_file)


@st.cache_resource(show_spinner=False)
def read_model(path: str) -> dict:
    return joblib.load(path)


def make_latest_features(source: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the model's current-time features without crossing missing hours."""
    frame = source.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame["AQI"] = pd.to_numeric(frame["AQI"], errors="coerce")
    frame = frame.dropna(subset=["Timestamp", "AQI"])
    frame = frame.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    frame = frame.set_index("Timestamp").sort_index()
    if frame.empty:
        raise ValueError("The data file does not contain any valid timestamped AQI readings.")

    hours = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    hourly = frame.reindex(hours)
    features = pd.DataFrame(index=hours)
    features["AQI"] = hourly["AQI"]
    for lag in (1, 3, 6, 24):
        features[f"AQI_lag_{lag}h"] = hourly["AQI"].shift(lag)
    for window in (3, 6, 24):
        rolling = hourly["AQI"].rolling(window=window, min_periods=window)
        features[f"AQI_rolling_{window}h_mean"] = rolling.mean()
        features[f"AQI_rolling_{window}h_std"] = rolling.std()
    features["Hour"] = hourly.index.hour
    features["DayOfWeek"] = hourly.index.dayofweek
    features["Month"] = hourly.index.month

    excluded = {"Timestamp", "AQI", "AQI_Category", "Target_Next_Hour_AQI"}
    measures = [
        name for name in frame.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]
    for name in measures:
        features[name] = hourly[name]

    latest_time = frame.index.max()
    latest = features.loc[[latest_time]].reindex(columns=columns)
    return latest, latest_time


def draw_metric_chart(metrics: dict) -> plt.Figure:
    names = ["Persistence", "Random Forest", "Extra Trees"]
    keys = ["Persistence", "RandomForest", "ExtraTrees"]
    values = [metrics[key]["MAE"] for key in keys]
    colors = [GOLD, RUST, OLIVE]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#fbf8f1")
    ax.set_facecolor("#fbf8f1")
    bars = ax.barh(names, values, color=colors, height=.56)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE · AQI points (lower is better)", color=MUTED)
    ax.tick_params(axis="both", colors=INK, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="x", color=LINE, linewidth=.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + .025, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, max(values) * 1.22)
    fig.tight_layout()
    return fig


def draw_prediction_chart(predictions: pd.DataFrame, points: int) -> plt.Figure:
    view = predictions.tail(points).copy()
    view["Timestamp"] = pd.to_datetime(view["Timestamp"])
    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#fbf8f1")
    ax.set_facecolor("#fbf8f1")
    ax.plot(view["Timestamp"], view["Actual_AQI"], color=INK, linewidth=1.8, label="Observed AQI")
    ax.plot(view["Timestamp"], view["Predicted_AQI"], color=RUST, linewidth=1.4, label="Random Forest")
    ax.plot(view["Timestamp"], view["Baseline_AQI"], color=OLIVE, linewidth=1.1, alpha=.85, label="Persistence")
    ax.set_ylabel("AQI", color=MUTED)
    ax.tick_params(axis="both", colors=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="y", color=LINE, linewidth=.7)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def draw_india_map() -> pdk.Deck:
    """Show the CPCB/DPCC station location in its India-wide context."""
    station = pd.DataFrame({
        "longitude": [77.2272],
        "latitude": [28.6568],
        "label": ["Chandni Chowk · Delhi"],
    })
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=station,
            get_position="[longitude, latitude]",
            get_radius=26000,
            radius_min_pixels=7,
            get_fill_color=[197, 150, 67, 105],
            get_line_color=[197, 150, 67, 220],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=station,
            get_position="[longitude, latitude]",
            get_radius=8000,
            radius_min_pixels=4,
            get_fill_color=[180, 84, 59, 255],
            get_line_color=[251, 248, 241, 255],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=station,
            get_position="[longitude, latitude]",
            get_text="label",
            get_size=16,
            get_color=[41, 40, 32, 255],
            get_pixel_offset=[18, -12],
            pickable=False,
        ),
    ]
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=21.5,
            longitude=82.0,
            zoom=2.95,
            pitch=0,
            bearing=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{label}\n{latitude}° N, {longitude}° E"},
    )


if not all(path.exists() for path in (DATA_PATH, MODEL_PATH, PREDICTIONS_PATH, REPORT_PATH)):
    st.error("A required project data or model file is missing. Keep app.py, Data/, and models/ together.")
    st.stop()

try:
    with st.spinner("Loading the saved AQI model…"):
        data = read_table(str(DATA_PATH))
        predictions = read_table(str(PREDICTIONS_PATH))
        report = read_report(str(REPORT_PATH))
        bundle = read_model(str(MODEL_PATH))
    latest_features, latest_time = make_latest_features(data, bundle["features"])
    latest_aqi = float(latest_features["AQI"].iloc[0])
    forecast_aqi = float(bundle["model"].predict(latest_features)[0])
except Exception as error:
    st.error(f"The project could not load its saved model or data: {error}")
    st.stop()

test_metrics = report["test_metrics"]
rf_metrics = test_metrics["RandomForest"]
et_metrics = test_metrics["ExtraTrees"]
baseline_metrics = test_metrics["Persistence"]
test_accuracy = float(np.mean(np.abs(predictions["Actual_AQI"] - predictions["Predicted_AQI"]) <= 5) * 100)

st.markdown('<div class="eyebrow">A student project · Chandni Chowk, Delhi, India</div>', unsafe_allow_html=True)
st.title("AQI, one hour ahead.")
st.markdown(
    '<p class="lede">A small look at what hourly pollution readings can tell us about the next hour. '
    'The saved Random Forest model is compared with a simple “same as now” baseline.</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

left, right = st.columns([1.0, 1.0], gap="large")
with left:
    st.markdown("### A forecast from the latest project reading")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Observed AQI", f"{latest_aqi:.0f}", help="Last AQI value in the project's ML-ready data.")
    metric_cols[1].metric("Model estimate · next hour", f"{forecast_aqi:.1f}", f"{forecast_aqi - latest_aqi:+.1f} from current")
    st.caption(f"Forecast origin: {latest_time:%d %b %Y, %H:%M} · target hour: {(latest_time + pd.Timedelta(hours=1)):%d %b %Y, %H:%M}")
with right:
    st.markdown("### The monitoring site")
    st.pydeck_chart(draw_india_map(), height=315, width="stretch")
    st.caption("India → Delhi → Chandni Chowk · CPCB / DPCC monitoring station")
    st.markdown("[View station details and coordinates ↗](https://aqi.delhi.gov.in/stations/chandni-chowk)")

st.markdown(
    f'<div class="note">This is a demonstration using the project’s latest available row, '
    f'<b>{latest_time:%d %b %Y}</b>. It is not a live CPCB reading or a current AQI alert.</div>',
    unsafe_allow_html=True,
)

st.markdown("### Test-set snapshot")
score_cols = st.columns(3)
score_cols[0].metric("Random Forest · MAE", f"{rf_metrics['MAE']:.3f}")
score_cols[1].metric("Within ±5 AQI points", f"{test_accuracy:.1f}%")
score_cols[2].metric("Persistence · MAE", f"{baseline_metrics['MAE']:.3f}")
st.caption("Test set: later timestamps kept out of model fitting and selection.")

overview, results_tab, method_tab = st.tabs(["Explore the results", "How the models compare", "How it works"])

with overview:
    chart_col, story_col = st.columns([1.35, .8], gap="large")
    with chart_col:
        st.markdown("### The held-out predictions")
        points = st.select_slider("Hours to show", options=[72, 168, 336, 720, len(predictions)], value=336)
        st.pyplot(draw_prediction_chart(predictions, int(points)), width="stretch")
    with story_col:
        st.markdown("### Reading the chart")
        st.write("The charcoal line is the measured AQI. Rust shows the Random Forest estimate, while olive shows persistence: the current AQI carried forward one hour.")
        st.write("The scores are close because AQI often moves gradually. Sharp changes are harder for either model to predict.")
        st.metric("Random Forest · RMSE", f"{rf_metrics['RMSE']:.3f}")
        st.metric("Random Forest · R²", f"{rf_metrics['R2']:.3f}")

with results_tab:
    st.markdown("### Test error by model")
    st.pyplot(draw_metric_chart(test_metrics), width="stretch")
    results = pd.DataFrame([
        {"Model": "Persistence", "MAE": baseline_metrics["MAE"], "RMSE": baseline_metrics["RMSE"], "R²": baseline_metrics["R2"]},
        {"Model": "Random Forest · selected on validation", "MAE": rf_metrics["MAE"], "RMSE": rf_metrics["RMSE"], "R²": rf_metrics["R2"]},
        {"Model": "Extra Trees · comparison", "MAE": et_metrics["MAE"], "RMSE": et_metrics["RMSE"], "R²": et_metrics["R2"]},
    ])
    st.dataframe(results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}), hide_index=True, width="stretch")
    st.caption("MAE and RMSE are AQI points; lower is better. R² summarizes explained variation and should not be used alone to judge the model.")

with method_tab:
    st.markdown("### The prediction setup")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**01 · Read the hour**  \nPollutant readings, weather, time of day, current AQI, and recent AQI history.")
    c2.markdown("**02 · Look one hour ahead**  \nPredict the AQI exactly one calendar hour later. Missing hours break the history windows.")
    c3.markdown("**03 · Check later data**  \nChronological split: 70% training, 15% validation, 15% test. Tuning uses validation only.")
    st.markdown("### A few details that matter")
    st.write("Random Forest was selected using validation MAE. Extra Trees is shown as a comparison; it is not selected based on its test score.")
    st.write("Missing predictor values are filled by a median imputer fitted on training data only. Missing AQI targets are not filled.")
    st.write("The existing AQI labels are provisional and have not been independently checked against every CPCB sub-index rule. This project is an educational prediction demo, not health guidance.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built for an academic AQI prediction project · Data: CPCB hourly measurements, Chandni Chowk, Delhi, India")
