from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "lag_1",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "hour",
    "day_of_week",
]
TARGET_COLUMN = "total_power_kw"


def build_fleet_hourly_frame(clean_frame: pd.DataFrame) -> pd.DataFrame:
    frame = clean_frame.copy()

    fleet_frame = (
        frame.groupby("timestamp", as_index=False)
        .agg(
            total_power_kw=("power_kw", "sum"),
            mean_wind_speed=("wind_speed_clean", "mean"),
            available_turbines=("availability", "sum"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    full_hours = pd.date_range(
        start=fleet_frame["timestamp"].min(),
        end=fleet_frame["timestamp"].max(),
        freq="h",
    )

    fleet_frame = (
        fleet_frame.set_index("timestamp")
        .reindex(full_hours)
        .rename_axis("timestamp")
        .reset_index()
    )
    fleet_frame = fleet_frame.sort_values("timestamp").reset_index(drop=True)
    return fleet_frame


def add_forecasting_features(fleet_frame: pd.DataFrame) -> pd.DataFrame:
    forecast_frame = fleet_frame.copy()

    forecast_frame["lag_1"] = forecast_frame[TARGET_COLUMN].shift(1)
    forecast_frame["lag_24"] = forecast_frame[TARGET_COLUMN].shift(24)
    forecast_frame["lag_48"] = forecast_frame[TARGET_COLUMN].shift(48)
    forecast_frame["lag_168"] = forecast_frame[TARGET_COLUMN].shift(168)

    # The calendar features let the model learn the daily cycle without looking ahead.
    forecast_frame["hour"] = forecast_frame["timestamp"].dt.hour
    forecast_frame["day_of_week"] = forecast_frame["timestamp"].dt.dayofweek

    # Rolling statistics summarize the recent generation pattern using only past hours.
    past_target = forecast_frame[TARGET_COLUMN].shift(1)
    forecast_frame["rolling_mean_24"] = past_target.rolling(window=24).mean()
    forecast_frame["rolling_std_24"] = past_target.rolling(window=24).std()

    return forecast_frame


def build_forecasting_frame(clean_frame: pd.DataFrame) -> pd.DataFrame:
    fleet_frame = build_fleet_hourly_frame(clean_frame)
    return add_forecasting_features(fleet_frame)
