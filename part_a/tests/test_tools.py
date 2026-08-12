from __future__ import annotations

from pathlib import Path

import pandas as pd

from part_a.tools.calculation_tools import CalculationTool
from part_a.tools.data_tools import DataQueryTool, RATED_CAPACITY_KW
from part_a.tools.rulebook_tool import RulebookTool


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY = ROOT / "data_pack" / "raw" / "turbine_telemetry_14day_sample.csv"
DAM = ROOT / "data_pack" / "raw" / "dam_price_14day_sample.csv"


def test_turbine_summary_t01():
    tool = DataQueryTool(TELEMETRY, DAM)
    result = tool.turbine_summary("T01")
    assert result["ok"] is True
    assert result["data"]["rows"] == 336
    assert round(result["data"]["average_power_kw"], 6) == 428.083631


def test_turbine_capacity_factor_t05():
    tool = DataQueryTool(TELEMETRY, DAM)
    calculator = CalculationTool()
    summary = tool.turbine_summary("T05")
    assert summary["ok"] is True
    cf = calculator.capacity_factor(summary["data"]["average_power_kw"], RATED_CAPACITY_KW)
    assert round(cf, 6) == 0.213538


def test_dam_summary():
    tool = DataQueryTool(TELEMETRY, DAM)
    result = tool.dam_summary()
    assert result["ok"] is True
    assert round(result["data"]["average_price"], 6) == 5.104399


def test_rulebook_lookup_temporal_correlation():
    rulebook = RulebookTool(ROOT / "data_pack" / "compliance_rulebook.docx")
    result = rulebook.lookup("temporal correlation")
    assert result["ok"] is True
    assert any("Rule 1" in line for line in result["matches"])


def test_rulebook_lookup_unsupported_topic():
    rulebook = RulebookTool(ROOT / "data_pack" / "compliance_rulebook.docx")
    result = rulebook.lookup("battery-storage requirement")
    assert result["ok"] is False
    assert "No relevant rule" in result["message"]


def test_unknown_turbine_returns_gracefully():
    tool = DataQueryTool(TELEMETRY, DAM)
    result = tool.turbine_summary("T99")
    assert result["ok"] is False
    assert "Invalid or unknown turbine_id" in result["message"]


def test_missing_timestamp_returns_gracefully():
    tool = DataQueryTool(TELEMETRY, DAM)
    result = tool.telemetry_for_timestamps([pd.NaT])
    assert result["ok"] is False


def test_empty_query_not_ok():
    from part_a.agent import ask_agent

    assert ask_agent("") == "Please ask a non-empty question."
