from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import fit_gradient_boosting, make_persistence_prediction


@dataclass(frozen=True)
class ValidationResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame


def mape(y_true, y_pred) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_array)
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / denominator[mask])) * 100
    )


def smape(y_true, y_pred) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_array) + np.abs(y_pred_array)
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(2 * np.abs(y_pred_array[mask] - y_true_array[mask]) / denominator[mask]) * 100
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape(y_true, y_pred)),
        "sMAPE": float(smape(y_true, y_pred)),
    }


def _build_folds(
    forecast_frame: pd.DataFrame,
    evaluation_days: int = 28,
    fold_days: int = 7,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    evaluation_end = forecast_frame["timestamp"].max()
    evaluation_start = evaluation_end.floor("D") - pd.Timedelta(days=evaluation_days - 1)

    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    fold_start = evaluation_start
    while fold_start <= evaluation_end:
        fold_end = min(
            fold_start + pd.Timedelta(days=fold_days) - pd.Timedelta(hours=1),
            evaluation_end,
        )
        folds.append((fold_start, fold_end))
        fold_start = fold_end + pd.Timedelta(hours=1)
    return folds


def run_walk_forward_validation(
    forecast_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "total_power_kw",
    baseline_column: str = "lag_24",
    model_factory: Callable[[], object] = fit_gradient_boosting,
) -> ValidationResult:
    folds = _build_folds(forecast_frame)
    prediction_frames: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, float | int]] = []

    for fold_number, (test_start, test_end) in enumerate(folds, start=1):
        train_mask = forecast_frame["timestamp"] < test_start
        test_mask = (forecast_frame["timestamp"] >= test_start) & (
            forecast_frame["timestamp"] <= test_end
        )

        train_data = forecast_frame.loc[train_mask].dropna(
            subset=[target_column, *feature_columns]
        )
        test_data = forecast_frame.loc[test_mask].dropna(
            subset=[target_column, *feature_columns, baseline_column]
        ).copy()

        if train_data.empty or test_data.empty:
            continue

        model = model_factory(train_data, feature_columns, target_column)
        model_predictions = model.predict(test_data[feature_columns])
        persistence_predictions = make_persistence_prediction(test_data, baseline_column)

        fold_predictions = pd.DataFrame(
            {
                "timestamp": test_data["timestamp"].values,
                "actual_power_kw": test_data[target_column].values,
                "persistence_prediction": persistence_predictions.values,
                "gradient_boosting_prediction": model_predictions,
                "fold": fold_number,
            }
        )
        prediction_frames.append(fold_predictions)

        fold_summary_rows.append(
            {
                "fold": fold_number,
                "test_start": test_start,
                "test_end": test_end,
                "rows": int(len(test_data)),
                "persistence_mae": regression_metrics(
                    test_data[target_column], persistence_predictions
                )["MAE"],
                "persistence_rmse": regression_metrics(
                    test_data[target_column], persistence_predictions
                )["RMSE"],
                "gradient_boosting_mae": regression_metrics(
                    test_data[target_column], model_predictions
                )["MAE"],
                "gradient_boosting_rmse": regression_metrics(
                    test_data[target_column], model_predictions
                )["RMSE"],
            }
        )

    if not prediction_frames:
        raise ValueError("Walk-forward validation produced no predictions.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_summary_rows)

    overall_metrics = pd.DataFrame(
        [
            {"Model": "24-hour Persistence", **regression_metrics(
                predictions["actual_power_kw"], predictions["persistence_prediction"]
            )},
            {"Model": "Gradient Boosting", **regression_metrics(
                predictions["actual_power_kw"], predictions["gradient_boosting_prediction"]
            )},
        ]
    )

    return ValidationResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
    )
