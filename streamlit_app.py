from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part_a.agent import ask_agent
from part_b.features import FEATURE_COLUMNS, build_fleet_hourly_frame


CLEANED_TELEMETRY_PATH = ROOT / "data_pack" / "turbine_90_cleaned.csv"
MODEL_PATH = ROOT / "models" / "gradient_boosting_model.pkl"
MODEL_COMPARISON_PATH = ROOT / "data_pack" / "model_comparison.csv"
HISTORICAL_END = pd.Timestamp("2026-05-29 23:00:00")
DEFAULT_FORECAST_DATE = pd.Timestamp("2026-05-30 00:00:00")
MODEL_DISPLAY_NAME = "Gradient Boosting"


st.set_page_config(
    page_title="Greenko Renewable Energy Operations",
    page_icon="🌱",
    layout="wide",
)


@st.cache_data
def load_cleaned_telemetry() -> pd.DataFrame:
    return pd.read_csv(CLEANED_TELEMETRY_PATH, parse_dates=["timestamp"])


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_resource
def load_forecast_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def build_hourly_history() -> pd.DataFrame:
    cleaned = load_cleaned_telemetry()
    return build_fleet_hourly_frame(cleaned)


def init_session_state() -> None:
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("forecast_result", None)
    st.session_state.setdefault("show_metrics", False)


def format_clock(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%H:%M")


def build_recursive_forecast(selected_date: pd.Timestamp) -> pd.DataFrame:
    model = load_forecast_model()
    history = build_hourly_history()

    if history.empty:
        raise ValueError("No historical telemetry is available for forecasting.")

    history = history.sort_values("timestamp").reset_index(drop=True)
    history_series = history.set_index("timestamp")["total_power_kw"].astype(float)
    history_series = history_series.dropna()

    if len(history_series) < 168:
        raise ValueError("Unable to generate a forecast for this date with the available historical data.")

    last_actual_timestamp = history_series.index.max()
    target_start = pd.Timestamp(selected_date)
    target_end = target_start + pd.Timedelta(hours=23)

    if target_start <= last_actual_timestamp.floor("D"):
        raise ValueError("Unable to generate a forecast for this date with the available historical data.")

    full_roll_forward_index = pd.date_range(last_actual_timestamp + pd.Timedelta(hours=1), target_end, freq="h")
    if len(full_roll_forward_index) < 24:
        raise ValueError("Unable to generate a forecast for this date with the available historical data.")

    extended_series = history_series.copy()
    predictions: list[dict[str, object]] = []

    for timestamp in full_roll_forward_index:
        if len(extended_series) < 168:
            raise ValueError("Unable to generate a forecast for this date with the available historical data.")

        recent_24 = extended_series.iloc[-24:]
        if recent_24.isna().any():
            raise ValueError("Unable to generate a forecast for this date with the available historical data.")

        feature_row = pd.DataFrame([
            {
                "lag_1": float(extended_series.iloc[-1]),
                "lag_24": float(extended_series.iloc[-24]),
                "lag_48": float(extended_series.iloc[-48]),
                "lag_168": float(extended_series.iloc[-168]),
                "rolling_mean_24": float(recent_24.mean()),
                "rolling_std_24": float(recent_24.std()),
                "hour": int(timestamp.hour),
                "day_of_week": int(timestamp.dayofweek),
            }
        ])[FEATURE_COLUMNS]

        predicted_power = float(model.predict(feature_row)[0])
        extended_series = pd.concat([extended_series, pd.Series([predicted_power], index=[timestamp])])
        predictions.append({
            "timestamp": timestamp,
            "Time": format_clock(timestamp),
            "Predicted Power (kW)": predicted_power,
        })

    forecast_df = pd.DataFrame(predictions)
    if len(forecast_df) != 24:
        raise ValueError("Unable to generate a full 24-hour forecast for the selected date.")
    return forecast_df


@st.cache_data
def forecast_next_day() -> pd.DataFrame:
    forecast = build_recursive_forecast(DEFAULT_FORECAST_DATE)
    return forecast.rename(columns={"Predicted Power (kW)": "predicted_power_kw"})


def submit_agent_question(question: str) -> None:
    st.session_state.agent_messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Thinking..."):
            answer = ask_agent(question)
    except Exception:
        answer = "The operations agent could not answer that question right now. Please try again or choose a different question."
    st.session_state.agent_messages.append({"role": "assistant", "content": answer})
    st.rerun()


def render_agent_tab() -> None:
    st.title("🤖 Renewable Operations Agent")
    st.caption("Ask questions about turbine performance, DAM prices and RFNBO compliance.")

    quick_questions = [
        "What is the average DAM price?",
        "What is the capacity factor of T01?",
        "What is the capacity factor of T05?",
        "What is the average power output of T01?",
        "According to the rulebook, what is temporal correlation?",
    ]

    st.subheader("Quick Questions")
    question_columns = st.columns(2)
    for index, question in enumerate(quick_questions):
        with question_columns[index % 2]:
            if st.button(question, key=f"quick_{index}", use_container_width=True):
                submit_agent_question(question)

    st.divider()

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about operations, prices or compliance")
    if user_question:
        submit_agent_question(user_question)

    with st.expander("Data Quality & Preprocessing", expanded=False):
        st.markdown(
            """
            - Temporal completeness checked in the completed Part B EDA.
            - Missing interval investigated before modelling.
            - Turbine anomalies investigated and validated in the notebook.
            - Final cleaned dataset used for forecasting: `data_pack/turbine_90_cleaned.csv`.
            """
        )


def render_forecast_metrics() -> None:
    metrics = load_model_comparison().copy()
    expected_columns = {"Model", "MAE", "RMSE", "MAPE", "sMAPE"}
    if not expected_columns.issubset(metrics.columns):
        st.error("Historical model metrics are not available in the expected format.")
        return

    st.subheader("Historical Model Performance")
    st.dataframe(metrics[["Model", "MAE", "RMSE", "MAPE", "sMAPE"]], use_container_width=True, hide_index=True)

    baseline = metrics.loc[metrics["Model"] == "24-hour Persistence"].iloc[0]
    stronger = metrics.loc[metrics["Model"] == MODEL_DISPLAY_NAME].iloc[0]
    mae_improvement = (1 - stronger["MAE"] / baseline["MAE"]) * 100
    rmse_improvement = (1 - stronger["RMSE"] / baseline["RMSE"]) * 100

    metric_columns = st.columns(2)
    metric_columns[0].metric("MAE improvement", f"{mae_improvement:.2f}%")
    metric_columns[1].metric("RMSE improvement", f"{rmse_improvement:.2f}%")
    st.caption("Gradient Boosting produced lower MAE and RMSE than the 24-hour persistence baseline during walk-forward evaluation.")


def render_forecast_tab() -> None:
    st.title("📈 Future Renewable Generation Forecast")
    st.caption("Prediction for 30 May 2026")

    try:
        forecast_result = forecast_next_day()
    except Exception as exc:
        st.error(str(exc))
        forecast_result = None

    if forecast_result is not None:
        st.subheader("Forecast Summary")
        summary_columns = st.columns(5)
        summary_columns[0].metric("Forecast Date", "30 May 2026")
        summary_columns[1].metric("Forecast Horizon", "24 Hours")
        summary_columns[2].metric("Model", MODEL_DISPLAY_NAME)
        summary_columns[3].metric("Peak Predicted Power", f"{forecast_result['predicted_power_kw'].max():.1f} kW")
        summary_columns[4].metric("Average Predicted Power", f"{forecast_result['predicted_power_kw'].mean():.1f} kW")

        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(
            forecast_result["timestamp"],
            forecast_result["predicted_power_kw"],
            marker="o",
            linewidth=2,
            color="#2E7D32",
        )
        axis.set_xlabel("Hour")
        axis.set_ylabel("Predicted Power (kW)")
        axis.set_title("Predicted Renewable Generation — 30 May 2026")
        axis.grid(True, alpha=0.2)
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)

        with st.expander("View Hourly Forecast", expanded=False):
            hourly_table = forecast_result[["timestamp", "predicted_power_kw"]].copy()
            hourly_table["Time"] = hourly_table["timestamp"].dt.strftime("%H:%M")
            hourly_table = hourly_table[["Time", "predicted_power_kw"]]
            hourly_table.columns = ["Time", "Predicted Power (kW)"]
            st.dataframe(hourly_table, use_container_width=True, hide_index=True)

        st.caption("This is a recursive 24-hour forecast generated from historical values and earlier predictions only.")

    st.markdown("---")
    if st.button("📊 Show Model Metrics", use_container_width=True):
        st.session_state.show_metrics = True

    if st.session_state.show_metrics:
        render_forecast_metrics()
        if st.button("Hide Metrics"):
            st.session_state.show_metrics = False
            st.rerun()


def render_sidebar() -> None:
    st.sidebar.title("🌱 Greenko")
    st.sidebar.subheader("Renewable Energy Operations")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Modules**")
    st.sidebar.markdown("🤖 Operations Agent")
    st.sidebar.markdown("📈 Generation Forecast")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data**")
    st.sidebar.markdown("Turbine telemetry")
    st.sidebar.markdown("DAM prices")
    st.sidebar.markdown("RFNBO compliance rulebook")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Forecasting**")
    st.sidebar.markdown("Horizon: 24 hours")
    st.sidebar.markdown(f"Model: {MODEL_DISPLAY_NAME}")
    st.sidebar.caption(f"Historical telemetry ends at {HISTORICAL_END.strftime('%Y-%m-%d %H:%M:%S')}")


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("Greenko Renewable Energy Operations")
    st.caption("A single Streamlit front end for the completed Part A agent and Part B forecasting pipeline.")

    agent_tab, forecast_tab = st.tabs(["🤖 Operations Agent", "📈 Generation Forecast"])
    with agent_tab:
        render_agent_tab()
    with forecast_tab:
        render_forecast_tab()


if __name__ == "__main__":
    main()
