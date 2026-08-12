from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_TURBINES = {"T01", "T02", "T03", "T04", "T05"}
RATED_CAPACITY_KW = 2000.0


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_telemetry_path() -> Path:
    return _workspace_root() / "data_pack" / "raw" / "turbine_telemetry_14day_sample.csv"


def _default_dam_path() -> Path:
    return _workspace_root() / "data_pack" / "raw" / "dam_price_14day_sample.csv"


@dataclass
class QueryResult:
    ok: bool
    message: str
    data: dict | None = None


class DataQueryTool:

    def __init__(self, telemetry_path: str | Path | None = None, dam_path: str | Path | None = None):

        self.telemetry_path = Path(telemetry_path or _default_telemetry_path())
        self.dam_path = Path(dam_path or _default_dam_path())

        self.telemetry = pd.read_csv(self.telemetry_path)
        self.dam = pd.read_csv(self.dam_path)

        self.telemetry["timestamp"] = pd.to_datetime(self.telemetry["timestamp"], errors="coerce")
        self.dam["timestamp"] = pd.to_datetime(self.dam["timestamp"], errors="coerce")
        self.telemetry["turbine_id"] = self.telemetry["turbine_id"].astype(str).str.upper().str.strip()

    def _validate_turbine_id(self, turbine_id: str | None) -> str | None:
        if turbine_id is None:
            return None
        cleaned = str(turbine_id).strip().upper()
        if cleaned not in EXPECTED_TURBINES:
            return None
        return cleaned

    def _format_empty(self, message: str) -> dict:
        return {"ok": False, "message": message, "data": None}

    def turbine_summary(self, turbine_id: str | None = None) -> dict:

        df = self.telemetry.copy()

        if turbine_id:
            cleaned = self._validate_turbine_id(turbine_id)
            if cleaned is None:
                return self._format_empty(f"Invalid or unknown turbine_id: {turbine_id}")
            df = df[df["turbine_id"] == cleaned]
            if df.empty:
                return self._format_empty(f"No telemetry rows found for turbine {cleaned}")

        return {
            "ok": True,
            "message": "Turbine summary computed successfully.",
            "data": {
                "rows": int(len(df)),
                "average_power_kw": float(df["power_kw"].mean()),
                "maximum_power_kw": float(df["power_kw"].max()),
                "minimum_power_kw": float(df["power_kw"].min()),
                "average_wind_speed": float(df["wind_speed"].mean()),
                "available_hours": int((df["availability"] == 1).sum()),
            },
        }

    def turbine_time_window_summary(self, start: str, end: str, turbine_id: str | None = None) -> dict:
        start_ts = pd.to_datetime(start, errors="coerce")
        end_ts = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            return self._format_empty("Invalid start or end timestamp")

        df = self.telemetry[(self.telemetry["timestamp"] >= start_ts) & (self.telemetry["timestamp"] < end_ts)].copy()
        if turbine_id:
            cleaned = self._validate_turbine_id(turbine_id)
            if cleaned is None:
                return self._format_empty(f"Invalid or unknown turbine_id: {turbine_id}")
            df = df[df["turbine_id"] == cleaned]

        if df.empty:
            return self._format_empty("No telemetry rows found in the requested time window")

        return {
            "ok": True,
            "message": "Window summary computed successfully.",
            "data": {
                "rows": int(len(df)),
                "average_power_kw": float(df["power_kw"].mean()),
                "average_wind_speed": float(df["wind_speed"].mean()),
                "available_hours": int((df["availability"] == 1).sum()),
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
            },
        }

    def telemetry_for_timestamps(self, timestamps: Iterable[pd.Timestamp], turbine_id: str | None = None) -> dict:
        timestamp_list = list(pd.to_datetime(list(timestamps), errors="coerce"))
        timestamp_list = [ts for ts in timestamp_list if not pd.isna(ts)]
        if not timestamp_list:
            return self._format_empty("No valid timestamps supplied")

        df = self.telemetry[self.telemetry["timestamp"].isin(timestamp_list)].copy()
        if turbine_id:
            cleaned = self._validate_turbine_id(turbine_id)
            if cleaned is None:
                return self._format_empty(f"Invalid or unknown turbine_id: {turbine_id}")
            df = df[df["turbine_id"] == cleaned]

        if df.empty:
            return self._format_empty("No telemetry rows found for the supplied timestamps")

        return {
            "ok": True,
            "message": "Telemetry rows retrieved successfully.",
            "data": df.sort_values(["timestamp", "turbine_id"]).to_dict(orient="records"),
        }

    def dam_summary(self) -> dict:

        if self.dam.empty:
            return self._format_empty("DAM price data is empty")

        return {
            "ok": True,
            "message": "DAM summary computed successfully.",
            "data": {
                "rows": int(len(self.dam)),
                "average_price": float(self.dam["dam_price_inr_per_kwh"].mean()),
                "maximum_price": float(self.dam["dam_price_inr_per_kwh"].max()),
                "minimum_price": float(self.dam["dam_price_inr_per_kwh"].min()),
                "start": self.dam["timestamp"].min().isoformat() if self.dam["timestamp"].notna().any() else None,
                "end": self.dam["timestamp"].max().isoformat() if self.dam["timestamp"].notna().any() else None,
            },
        }

    def dam_above_threshold(self, threshold: float) -> dict:
        if self.dam.empty:
            return self._format_empty("DAM price data is empty")
        filtered = self.dam[self.dam["dam_price_inr_per_kwh"] > float(threshold)].copy()
        if filtered.empty:
            return self._format_empty(f"No DAM rows found above threshold {threshold}")
        return {
            "ok": True,
            "message": "Filtered DAM rows retrieved successfully.",
            "data": {
                "rows": int(len(filtered)),
                "average_price": float(filtered["dam_price_inr_per_kwh"].mean()),
                "timestamps": filtered["timestamp"].sort_values().dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            },
        }

    def period_average_power(self, start: str | None = None, end: str | None = None, turbine_id: str | None = None) -> dict:
        df = self.telemetry.copy()
        if start is not None:
            start_ts = pd.to_datetime(start, errors="coerce")
            if pd.isna(start_ts):
                return self._format_empty("Invalid start timestamp")
            df = df[df["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.to_datetime(end, errors="coerce")
            if pd.isna(end_ts):
                return self._format_empty("Invalid end timestamp")
            df = df[df["timestamp"] < end_ts]
        if turbine_id:
            cleaned = self._validate_turbine_id(turbine_id)
            if cleaned is None:
                return self._format_empty(f"Invalid or unknown turbine_id: {turbine_id}")
            df = df[df["turbine_id"] == cleaned]
        if df.empty:
            return self._format_empty("No telemetry rows found for the requested period")
        return {
            "ok": True,
            "message": "Period average computed successfully.",
            "data": {"average_power_kw": float(df["power_kw"].mean()), "rows": int(len(df))},
        }
