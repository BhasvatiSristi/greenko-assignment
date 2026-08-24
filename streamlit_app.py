from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# Make the project root available for imports.
ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from part_a.agent import ask_agent
from part_b.features import FEATURE_COLUMNS, build_fleet_hourly_frame


# Files used by the frontend.
CLEANED_TELEMETRY_PATH = (
    ROOT / "data_pack" / "turbine_90_cleaned.csv"
)

MODEL_PATH = (
    ROOT / "models" / "gradient_boosting_model.pkl"
)

# Forecast date.
FORECAST_DATE = pd.Timestamp("2026-05-30 00:00:00")

MODEL_DISPLAY_NAME = "Gradient Boosting"


# ---------------------------------------------------------
# Load data and model
# ---------------------------------------------------------

@st.cache_data
def load_cleaned_telemetry() -> pd.DataFrame:
    """Load the cleaned turbine telemetry."""
    return pd.read_csv(
        CLEANED_TELEMETRY_PATH,
        parse_dates=["timestamp"],
    )


@st.cache_resource
def load_forecast_model():
    """Load the trained Gradient Boosting model."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def build_hourly_history() -> pd.DataFrame:
    """Convert turbine-level data into hourly fleet-level data."""
    cleaned = load_cleaned_telemetry()
    return build_fleet_hourly_frame(cleaned)


# ---------------------------------------------------------
# Forecasting
# ---------------------------------------------------------

def build_recursive_forecast(
    selected_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Generate a 24-hour recursive forecast.

    Each prediction is added to the history and can be
    used to generate the next hour's prediction.
    """

    model = load_forecast_model()
    history = build_hourly_history()

    if history.empty:
        raise ValueError(
            "No historical telemetry is available."
        )

    history = (
        history
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Use historical fleet power as the starting series.
    history_series = (
        history
        .set_index("timestamp")["total_power_kw"]
        .astype(float)
        .dropna()
    )

    if len(history_series) < 168:
        raise ValueError(
            "Not enough historical data to generate the forecast."
        )

    last_actual_timestamp = history_series.index.max()

    target_start = pd.Timestamp(selected_date)
    target_end = (
        target_start + pd.Timedelta(hours=23)
    )

    if target_start <= last_actual_timestamp.floor("D"):
        raise ValueError(
            "The selected forecast date must be after "
            "the last available historical date."
        )

    # Create the 24 future hourly timestamps.
    future_hours = pd.date_range(
        start=last_actual_timestamp + pd.Timedelta(hours=1),
        end=target_end,
        freq="h",
    )

    if len(future_hours) != 24:
        raise ValueError(
            "Could not create a complete 24-hour forecast."
        )

    # This series grows as we generate predictions.
    extended_series = history_series.copy()

    predictions = []

    for timestamp in future_hours:

        # Previous 24 hours.
        recent_24 = extended_series.iloc[-24:]

        # Create the same features used during training.
        feature_row = pd.DataFrame([
            {
                "lag_1": float(
                    extended_series.iloc[-1]
                ),

                "lag_24": float(
                    extended_series.iloc[-24]
                ),

                "lag_48": float(
                    extended_series.iloc[-48]
                ),

                "lag_168": float(
                    extended_series.iloc[-168]
                ),

                "rolling_mean_24": float(
                    recent_24.mean()
                ),

                "rolling_std_24": float(
                    recent_24.std()
                ),

                "hour": int(
                    timestamp.hour
                ),

                "day_of_week": int(
                    timestamp.dayofweek
                ),
            }
        ])[FEATURE_COLUMNS]

        # Generate prediction for this hour.
        predicted_power = float(
            model.predict(feature_row)[0]
        )

        # Add prediction to the history so that
        # it can be used for the next hour.
        extended_series = pd.concat([
            extended_series,
            pd.Series(
                [predicted_power],
                index=[timestamp],
            ),
        ])

        predictions.append({
            "timestamp": timestamp,
            "Time": timestamp.strftime("%H:%M"),
            "Predicted Power (kW)": predicted_power,
        })

    return pd.DataFrame(predictions)


@st.cache_data
def forecast_next_day() -> pd.DataFrame:
    """Generate the forecast for the selected day."""
    return build_recursive_forecast(FORECAST_DATE)


# ---------------------------------------------------------
# Operations Agent
# ---------------------------------------------------------

def submit_agent_question(question: str) -> None:
    """Send a user question to the Part A agent."""

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    st.session_state.agent_messages.append({
        "role": "user",
        "content": question,
    })

    try:
        with st.spinner("Thinking..."):
            answer = ask_agent(question)

    except Exception as exc:
        answer = (
            "The operations agent could not answer "
            "that question right now.\n\n"
            f"Error: {exc}"
        )

    st.session_state.agent_messages.append({
        "role": "assistant",
        "content": answer,
    })

    st.rerun()


def render_agent_tab() -> None:

    st.title("🤖 Renewable Operations Agent")

    st.caption(
        "Ask questions about turbine telemetry, "
        "DAM prices and RFNBO compliance."
    )

    quick_questions = [
        "What is the average DAM price?",
        "What is the total power output of all turbines?",
        "What is the capacity factor of T01?",
        "Compare the capacity factor of T01 and T05.",
        "What was T01's power output on April 1st 2026?",
        "According to the rulebook, what is temporal correlation?",
    ]

    st.subheader("Quick Questions")

    question_columns = st.columns(2)

    for index, question in enumerate(quick_questions):

        with question_columns[index % 2]:

            if st.button(
                question,
                key=f"quick_{index}",
                use_container_width=True,
            ):
                submit_agent_question(question)

    st.divider()

    # Display previous conversation messages.
    for message in st.session_state.get(
        "agent_messages",
        []
    ):

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Normal chat input.
    user_question = st.chat_input(
        "Ask a question about operations, prices or compliance"
    )

    if user_question:
        submit_agent_question(user_question)


# ---------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------

def render_forecast_tab() -> None:

    st.title("📈 Future Renewable Generation Forecast")

    st.caption(
        "24-hour renewable generation forecast "
        "using the trained Gradient Boosting model."
    )

    try:
        forecast = forecast_next_day()

    except Exception as exc:
        st.error(str(exc))
        return

    # Summary information.
    st.subheader("Forecast Summary")

    summary_columns = st.columns(4)

    summary_columns[0].metric(
        "Forecast Date",
        "30 May 2026",
    )

    summary_columns[1].metric(
        "Forecast Horizon",
        "24 Hours",
    )

    summary_columns[2].metric(
        "Model",
        MODEL_DISPLAY_NAME,
    )

    summary_columns[3].metric(
        "Average Power",
        f"{forecast['Predicted Power (kW)'].mean():.1f} kW",
    )

    # Forecast graph.
    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    axis.plot(
        forecast["timestamp"],
        forecast["Predicted Power (kW)"],
        marker="o",
        linewidth=2,
    )

    axis.set_xlabel("Hour")
    axis.set_ylabel("Predicted Power (kW)")
    axis.set_title(
        "Predicted Renewable Generation — 30 May 2026"
    )

    axis.grid(True, alpha=0.2)
    axis.tick_params(axis="x", rotation=45)

    figure.tight_layout()

    st.pyplot(
        figure,
        clear_figure=True,
    )

    # Show hourly predictions in a table.
    with st.expander(
        "View Hourly Forecast"
    ):

        st.dataframe(
            forecast[
                ["Time", "Predicted Power (kW)"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "The forecast is generated recursively using "
        "historical values and earlier predictions."
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

def render_sidebar() -> None:

    st.sidebar.title("🌱 Greenko")

    st.sidebar.subheader(
        "Renewable Energy Operations"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Modules")
    st.sidebar.markdown(
        "🤖 Operations Agent"
    )
    st.sidebar.markdown(
        "📈 Generation Forecast"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Data")
    st.sidebar.markdown(
        "Turbine telemetry"
    )
    st.sidebar.markdown(
        "DAM prices"
    )
    st.sidebar.markdown(
        "RFNBO compliance rulebook"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Forecasting")
    st.sidebar.markdown(
        "Horizon: 24 hours"
    )
    st.sidebar.markdown(
        f"Model: {MODEL_DISPLAY_NAME}"
    )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main() -> None:

    st.set_page_config(
        page_title="Greenko Renewable Energy Operations",
        page_icon="🌱",
        layout="wide",
    )

    render_sidebar()

    st.title(
        "Greenko Renewable Energy Operations"
    )

    st.caption(
        "Part A operations agent and "
        "Part B generation forecasting."
    )

    agent_tab, forecast_tab = st.tabs([
        "🤖 Operations Agent",
        "📈 Generation Forecast",
    ])

    with agent_tab:
        render_agent_tab()

    with forecast_tab:
        render_forecast_tab()


if __name__ == "__main__":
    main()