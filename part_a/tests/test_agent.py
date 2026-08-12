from __future__ import annotations

from pathlib import Path

from part_a.agent import ask_agent
from part_a.tools.data_tools import DataQueryTool


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY = ROOT / "data_pack" / "raw" / "turbine_telemetry_14day_sample.csv"
DAM = ROOT / "data_pack" / "raw" / "dam_price_14day_sample.csv"


def test_assignment_question_1_local_calculation():
    tool = DataQueryTool(TELEMETRY, DAM)
    week1 = tool.period_average_power("2026-03-01 00:00:00", "2026-03-08 00:00:00")
    week2 = tool.period_average_power("2026-03-08 00:00:00", "2026-03-15 00:00:00")
    assert week1["ok"] is True
    assert week2["ok"] is True
    assert round(week1["data"]["average_power_kw"] / 2000, 6) == 0.210009
    assert round(week2["data"]["average_power_kw"] / 2000, 6) == 0.235819


def test_assignment_question_2_local_calculation():
    tool = DataQueryTool(TELEMETRY, DAM)
    high_price = tool.dam_above_threshold(4.0)
    telemetry = tool.telemetry_for_timestamps(high_price["data"]["timestamps"])
    assert high_price["ok"] is True
    assert telemetry["ok"] is True
    avg_high = sum(row["power_kw"] for row in telemetry["data"]) / len(telemetry["data"])
    overall = tool.period_average_power()
    assert round(avg_high, 6) == 473.54861
    assert avg_high > overall["data"]["average_power_kw"]


def test_agent_handles_empty_query():
    assert ask_agent("") == "Please ask a non-empty question."
