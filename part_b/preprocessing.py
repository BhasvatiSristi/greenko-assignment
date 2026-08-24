from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import CLEAN_TELEMETRY_PATH, load_raw_telemetry


DEFAULT_CLEANED_PATH = CLEAN_TELEMETRY_PATH

T05_SCALE_FACTOR = 2.239306387561419
T05_START = pd.Timestamp("2026-03-26 00:00:00")
T05_END = pd.Timestamp("2026-04-04 23:00:00")

T03_START = pd.Timestamp("2026-05-04 00:00:00")
T03_END = pd.Timestamp("2026-05-11 23:00:00")


def apply_validated_cleaning(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(
            f"Invalid timestamps found in {telemetry_path}"
        )

    frame["turbine_id"] = frame["turbine_id"].astype(str).str.upper().str.strip()

    frame = frame.sort_values(["timestamp", "turbine_id"]).reset_index(drop=True)

    frame["wind_speed_clean"] = frame["wind_speed"]
    frame["power_anomaly_flag"] = 0

    t05_mask = (
        (frame["turbine_id"] == "T05")
        & (frame["timestamp"] >= T05_START)
        & (frame["timestamp"] <= T05_END)
    )
    frame.loc[t05_mask, "wind_speed_clean"] = (
        frame.loc[t05_mask, "wind_speed"] / T05_SCALE_FACTOR
    )

    t03_mask = (
        (frame["turbine_id"] == "T03")
        & (frame["timestamp"] >= T03_START)
        & (frame["timestamp"] <= T03_END)
    )
    frame.loc[t03_mask, "power_anomaly_flag"] = 1

    return frame


def build_clean_telemetry(
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    raw_frame = load_raw_telemetry(path=raw_path)
    clean_frame = apply_validated_cleaning(raw_frame)

    save_path = Path(output_path) if output_path is not None else DEFAULT_CLEANED_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clean_frame.to_csv(save_path, index=False)
    return clean_frame
