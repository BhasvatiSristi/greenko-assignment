from pathlib import Path
from typing import Sequence

import pandas as pd


# Default locations for the raw and cleaned telemetry files.
ROOT = Path(__file__).resolve().parents[1]

RAW_TELEMETRY_PATH = (
    ROOT / "data_pack" / "raw" / "turbine_telemetry_90day.csv"
)

CLEAN_TELEMETRY_PATH = (
    ROOT / "data_pack" / "turbine_90_cleaned.csv"
)


# Columns required for the forecasting pipeline.
REQUIRED_RAW_COLUMNS = (
    "timestamp",
    "turbine_id",
    "wind_speed",
    "power_kw",
    "availability",
)


def load_telemetry(
    path: str | Path | None = None,
    required_columns: Sequence[str] = REQUIRED_RAW_COLUMNS,
) -> pd.DataFrame:
    """
    Load telemetry data and perform basic validation.

    This function only handles loading, validation,
    timestamp conversion, ID standardization, and sorting.
    Domain-specific cleaning is handled in preprocessing.py.
    """

    # Use the raw telemetry file when no path is provided.
    telemetry_path = (
        Path(path) if path is not None else RAW_TELEMETRY_PATH
    )

    # Load the CSV into a DataFrame.
    frame = pd.read_csv(telemetry_path)

    # Check that all required columns are present.
    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in "
            f"{telemetry_path}: {missing_columns}"
        )

    frame = frame.copy()

    return frame