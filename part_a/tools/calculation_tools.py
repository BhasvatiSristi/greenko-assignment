from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalculationResult:
    ok: bool
    message: str
    data: dict | None = None


class CalculationTool:

    @staticmethod
    def capacity_factor(average_power_kw: float, capacity_kw: float = 2000) -> float:

        if capacity_kw <= 0:
            raise ValueError("Capacity must be greater than zero.")

        return float(average_power_kw) / float(capacity_kw)

    @staticmethod
    def average(values: list[float]) -> float:

        if not values:
            raise ValueError("Cannot calculate average of empty list.")

        return float(sum(values) / len(values))

    @staticmethod
    def percentage_change(new_value: float, old_value: float) -> float:

        if old_value == 0:
            raise ValueError("Cannot calculate percentage change from zero.")

        return float(((new_value - old_value) / old_value) * 100)

    @staticmethod
    def percentage_difference(value_a: float, value_b: float) -> float:

        denominator = (abs(value_a) + abs(value_b)) / 2
        if denominator == 0:
            raise ValueError("Cannot calculate percentage difference when both values are zero.")
        return float(abs(value_a - value_b) / denominator * 100)

    @staticmethod
    def correlation_coverage(total_hours: int, valid_hours: int) -> float:

        if total_hours <= 0:
            raise ValueError("Total hours must be greater than zero.")
        if valid_hours < 0:
            raise ValueError("Valid hours cannot be negative.")
        return float(valid_hours / total_hours * 100)

    @staticmethod
    def compare_values(value_a: float, value_b: float, label_a: str = "A", label_b: str = "B") -> dict:

        difference = float(value_a - value_b)
        if value_a == value_b:
            direction = "equal to"
        elif value_a > value_b:
            direction = f"above {label_b}"
        else:
            direction = f"below {label_b}"
        return {
            "ok": True,
            "message": "Comparison computed successfully.",
            "data": {
                label_a: float(value_a),
                label_b: float(value_b),
                "difference": difference,
                "direction": direction,
            },
        }
