from langchain_core.tools import tool


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
# Percentage Improvement
# ---------------------------------------------------------

@tool
def calculate_percentage_improvement(
    baseline_value: float,
    model_value: float
) -> str:
    """
    Calculate percentage improvement of a model over a baseline.

    Formula:

    (baseline - model) / baseline * 100

    Lower-is-better metrics such as MAE and RMSE can use this.
    """

    if baseline_value == 0:
        return "Cannot calculate improvement because baseline is zero."

    improvement = (
        (baseline_value - model_value)
        / baseline_value
        * 100
    )

    return f"Percentage improvement: {improvement:.2f}%"


# ---------------------------------------------------------
# Percentage Difference
# ---------------------------------------------------------

@tool
def calculate_percentage_difference(
    value_a: float,
    value_b: float
) -> str:
    """
    Calculate the percentage difference between two values.
    """

    denominator = (abs(value_a) + abs(value_b)) / 2

    if denominator == 0:
        return "Cannot calculate percentage difference."

    difference = (
        abs(value_a - value_b)
        / denominator
        * 100
    )

    return f"Percentage difference: {difference:.2f}%"


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
    calculate_percentage_improvement,
    calculate_percentage_difference,
    calculate_correlation_coverage
]