from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TELEMETRY_PATH = ROOT / "data_pack" / "raw" / "turbine_telemetry_90day.csv"
CLEAN_TELEMETRY_PATH = ROOT / "data_pack" / "turbine_90_cleaned.csv"

REQUIRED_RAW_COLUMNS = (
    "timestamp",
    "turbine_id",
    "wind_speed",
    "power_kw",
    "availability",
)


def _resolve_path(path: str | Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path
    return Path(path)


def load_telemetry(
    path: str | Path | None = None,
    required_columns: Sequence[str] = REQUIRED_RAW_COLUMNS,
) -> pd.DataFrame:
    telemetry_path = _resolve_path(path, RAW_TELEMETRY_PATH)
    frame = pd.read_csv(telemetry_path)

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {telemetry_path}: {missing_columns}"
        )

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"Invalid timestamps found in {telemetry_path}")

    frame["turbine_id"] = (
        frame["turbine_id"].astype(str).str.upper().str.strip()
    )

    sort_columns = [column for column in ("timestamp", "turbine_id") if column in frame.columns]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame


def load_raw_telemetry(path: str | Path | None = None) -> pd.DataFrame:
    return load_telemetry(path=path, required_columns=REQUIRED_RAW_COLUMNS)


def load_cleaned_telemetry(path: str | Path | None = None) -> pd.DataFrame:
    cleaned_path = _resolve_path(path, CLEAN_TELEMETRY_PATH)
    return load_telemetry(path=cleaned_path, required_columns=REQUIRED_RAW_COLUMNS)
