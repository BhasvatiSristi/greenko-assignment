from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from langchain.tools import tool

from .calculation_tools import CalculationTool
from .data_tools import DataQueryTool, RATED_CAPACITY_KW
from .rulebook_tool import RulebookTool


DATA_TOOL = DataQueryTool()
CALCULATOR = CalculationTool()
RULEBOOK = RulebookTool()


def _format_result(result: dict, tool_name: str) -> str:
    prefix = f"Tool used: {tool_name}\n"
    if not result.get("ok"):
        return prefix + result.get("message", "Unable to complete request.")
    data = result.get("data")
    return prefix + pd.Series(data).to_json() if isinstance(data, dict) else prefix + str(data)


@tool
def get_turbine_summary(turbine_id: str) -> str:
    """Return deterministic summary statistics for a single turbine."""
    result = DATA_TOOL.turbine_summary(turbine_id)
    if not result["ok"]:
        return f"Tool used: get_turbine_summary\n{result['message']}"
    data = result["data"]
    return (
        "Tool used: get_turbine_summary\n"
        f"Turbine: {str(turbine_id).strip().upper()}\n"
        f"Observations: {data['rows']}\n"
        f"Average power: {data['average_power_kw']:.2f} kW\n"
        f"Maximum power: {data['maximum_power_kw']:.2f} kW\n"
        f"Minimum power: {data['minimum_power_kw']:.2f} kW\n"
        f"Average wind speed: {data['average_wind_speed']:.2f} m/s\n"
        f"Available hours: {data['available_hours']}"
    )


@tool
def get_dam_summary() -> str:
    """Return deterministic summary statistics for the DAM price series."""
    result = DATA_TOOL.dam_summary()
    if not result["ok"]:
        return f"Tool used: get_dam_summary\n{result['message']}"
    data = result["data"]
    return (
        "Tool used: get_dam_summary\n"
        f"Average DAM price: ₹{data['average_price']:.3f}/kWh\n"
        f"Maximum DAM price: ₹{data['maximum_price']:.3f}/kWh\n"
        f"Minimum DAM price: ₹{data['minimum_price']:.3f}/kWh"
    )


@tool
def get_turbine_capacity_factor(turbine_id: str) -> str:
    """Return the capacity factor for one turbine using the rated capacity from the assignment."""
    summary = DATA_TOOL.turbine_summary(turbine_id)
    if not summary["ok"]:
        return f"Tool used: get_turbine_capacity_factor\n{summary['message']}"
    average_power = summary["data"]["average_power_kw"]
    capacity_factor = CALCULATOR.capacity_factor(average_power, RATED_CAPACITY_KW)
    turbine = str(turbine_id).strip().upper()
    return (
        "Tool used: get_turbine_capacity_factor\n"
        f"Turbine: {turbine}\n"
        f"Average power: {average_power:.2f} kW\n"
        f"Rated capacity: {RATED_CAPACITY_KW:.0f} kW\n"
        f"Capacity factor: {capacity_factor:.4f} ({capacity_factor:.2%})"
    )


@tool
def lookup_compliance_rule(query: str) -> str:
    """Return the actual relevant rule text from the supplied rulebook."""
    result = RULEBOOK.lookup(query)
    if not result["ok"]:
        return f"Tool used: lookup_compliance_rule\n{result['message']}"
    lines = [
        "Tool used: lookup_compliance_rule",
        f"Topic: {result['topic'] or 'general'}",
        f"Source: {result['source']}",
        "Relevant rule text:",
        *result["matches"],
    ]
    return "\n".join(lines)


@tool
def compare_weekly_capacity_factor() -> str:
    """Compare the first and second week capacity factor using the 14-day telemetry sample."""
    start = DATA_TOOL.telemetry["timestamp"].min()
    if pd.isna(start):
        return "Tool used: compare_weekly_capacity_factor\nNo telemetry timestamps available."
    week1_end = start + timedelta(days=7)
    week2_end = week1_end + timedelta(days=7)
    week1 = DATA_TOOL.period_average_power(str(start), str(week1_end))
    week2 = DATA_TOOL.period_average_power(str(week1_end), str(week2_end))
    if not week1["ok"] or not week2["ok"]:
        message = week1.get("message") if not week1["ok"] else week2.get("message")
        return f"Tool used: compare_weekly_capacity_factor\n{message}"
    week1_cf = CALCULATOR.capacity_factor(week1["data"]["average_power_kw"], RATED_CAPACITY_KW)
    week2_cf = CALCULATOR.capacity_factor(week2["data"]["average_power_kw"], RATED_CAPACITY_KW)
    comparison = CALCULATOR.compare_values(week2_cf, week1_cf, "week2", "week1")
    return (
        "Tool used: compare_weekly_capacity_factor\n"
        f"Week 1 capacity factor: {week1_cf:.4f} ({week1_cf:.2%})\n"
        f"Week 2 capacity factor: {week2_cf:.4f} ({week2_cf:.2%})\n"
        f"Difference: {comparison['data']['difference']:.4f}\n"
        f"Week 2 is {comparison['data']['direction']} Week 1"
    )


@tool
def compare_output_on_high_dam_price(threshold: float = 4.0) -> str:
    """Compare turbine output during DAM hours above a threshold with the overall period average."""
    dam_result = DATA_TOOL.dam_above_threshold(threshold)
    if not dam_result["ok"]:
        return f"Tool used: compare_output_on_high_dam_price\n{dam_result['message']}"
    telemetry_result = DATA_TOOL.telemetry_for_timestamps(pd.to_datetime(dam_result["data"]["timestamps"]))
    if not telemetry_result["ok"]:
        return f"Tool used: compare_output_on_high_dam_price\n{telemetry_result['message']}"
    high_price_avg = CALCULATOR.average([row["power_kw"] for row in telemetry_result["data"]])
    overall_avg = DATA_TOOL.period_average_power()
    comparison = CALCULATOR.compare_values(high_price_avg, overall_avg["data"]["average_power_kw"], "high_price_period", "overall_period")
    return (
        "Tool used: compare_output_on_high_dam_price\n"
        f"DAM threshold: ₹{threshold:.2f}/kWh\n"
        f"High-price telemetry average: {high_price_avg:.2f} kW\n"
        f"Overall telemetry average: {overall_avg['data']['average_power_kw']:.2f} kW\n"
        f"Output is {comparison['data']['direction']} the overall period average"
    )
