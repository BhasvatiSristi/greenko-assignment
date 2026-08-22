from langchain_core.tools import tool
from data_query import get_dam_prices_for_window


# ---------------------------------------------------------
# Capacity Factor
# ---------------------------------------------------------

@tool
def calculate_capacity_factor(
    average_power_kw: float,
    rated_capacity_kw: float = 2000.0
) -> str:
    """
    Calculate turbine capacity factor.

    Formula:

    capacity factor = average power / rated capacity

    Returns both decimal and percentage values.
    """

    if rated_capacity_kw <= 0:
        return "Rated capacity must be greater than zero."

    capacity_factor = (
        average_power_kw / rated_capacity_kw
    )

    percentage = capacity_factor * 100

    return (
        f"Capacity factor: {capacity_factor:.4f}\n"
        f"Capacity factor percentage: {percentage:.2f}%"
    )

# ---------------------------------------------------------
# Average DAM price in a window
# ---------------------------------------------------------

@tool
def calculate_average_dam_price(user_query: str) -> str:
    """
    Calculate the average DAM price for the
    time period specified in the user's question.
    """

    result = get_dam_prices_for_window(user_query)

    rows = result["rows"]

    if not rows:
        return "No DAM price data found for the requested period."

    prices = [float(row[1]) for row in rows]

    average_price = sum(prices) / len(prices)

    return (
        f"Average DAM price: {average_price:.3f}\n"
        f"Period: {result['start_date']} to {result['end_date']}\n"
        f"Hourly prices used: {len(prices)}"
    )

# ---------------------------------------------------------
# Temporal Correlation Coverage
# ---------------------------------------------------------

@tool
def calculate_correlation_coverage(
    correlated_hours: int,
    total_hours: int
) -> str:
    """
    Calculate the percentage of hours satisfying a temporal
    correlation condition.

    Formula:

    correlated hours / total hours * 100
    """

    if total_hours <= 0:
        return "Total hours must be greater than zero."

    percentage = (
        correlated_hours
        / total_hours
        * 100
    )

    return (
        f"Correlated hours: {correlated_hours}\n"
        f"Total hours: {total_hours}\n"
        f"Correlation coverage: {percentage:.2f}%"
    )


CALCULATOR_TOOLS = [
    calculate_capacity_factor,
    calculate_correlation_coverage,
    calculate_average_dam_price
]