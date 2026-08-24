from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import fit_gradient_boosting, make_persistence_prediction


@dataclass
class ValidationResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame


def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE and sMAPE."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ignore zero actual values for percentage-based metrics.
    non_zero = y_true != 0

    if non_zero.any():
        mape = np.mean(
            np.abs(
                (y_true[non_zero] - y_pred[non_zero])
                / y_true[non_zero]
            )
        ) * 100

        smape = np.mean(
            2 * np.abs(y_pred[non_zero] - y_true[non_zero])
            / (
                np.abs(y_true[non_zero])
                + np.abs(y_pred[non_zero])
            )
        ) * 100
    else:
        mape = np.nan
        smape = np.nan

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape,
        "sMAPE": smape,
    }


def build_folds(
    forecast_frame,
    evaluation_days=28,
    fold_days=7,
):
    """
    Create 7-day test folds from the final 28 days.
    """

    evaluation_end = forecast_frame["timestamp"].max()

    evaluation_start = (
        evaluation_end.floor("D")
        - pd.Timedelta(days=evaluation_days - 1)
    )

    folds = []

    fold_start = evaluation_start

    while fold_start <= evaluation_end:

        fold_end = min(
            fold_start
            + pd.Timedelta(days=fold_days)
            - pd.Timedelta(hours=1),
            evaluation_end,
        )

        folds.append((fold_start, fold_end))

        fold_start = fold_end + pd.Timedelta(hours=1)

    return folds


def run_walk_forward_validation(
    forecast_frame,
    feature_columns,
    target_column="total_power_kw",
    baseline_column="lag_24",
):
    """
    Train on past data and test on future data
    using 4 weekly walk-forward folds.
    """

    folds = build_folds(forecast_frame)

    all_predictions = []
    all_fold_metrics = []

    for fold_number, (test_start, test_end) in enumerate(
        folds, start=1
    ):

        # Everything before the test period is training data.
        train_mask = (
            forecast_frame["timestamp"] < test_start
        )

        # Current 7-day period is test data.
        test_mask = (
            (forecast_frame["timestamp"] >= test_start)
            & (forecast_frame["timestamp"] <= test_end)
        )

        train_data = forecast_frame.loc[
            train_mask
        ].dropna(
            subset=[target_column] + feature_columns
        )

        test_data = forecast_frame.loc[
            test_mask
        ].dropna(
            subset=[target_column]
            + feature_columns
            + [baseline_column]
        )

        if train_data.empty or test_data.empty:
            continue

        # Train Gradient Boosting only on past data.
        model = fit_gradient_boosting(
            train_data,
            feature_columns,
            target_column,
        )

        # Gradient Boosting predictions.
        gb_predictions = model.predict(
            test_data[feature_columns]
        )

        # Simple persistence predictions.
        persistence_predictions = (
            make_persistence_prediction(
                test_data,
                baseline_column,
            )
        )

        # Calculate metrics once for each model.
        persistence_metrics = calculate_metrics(
            test_data[target_column],
            persistence_predictions,
        )

        gb_metrics = calculate_metrics(
            test_data[target_column],
            gb_predictions,
        )

        # Save predictions.
        fold_predictions = pd.DataFrame({
            "timestamp": test_data["timestamp"].values,
            "actual_power_kw": test_data[target_column].values,
            "persistence_prediction": persistence_predictions.values,
            "gradient_boosting_prediction": gb_predictions,
            "fold": fold_number,
        })

        all_predictions.append(fold_predictions)

        # Save metrics for this fold.
        all_fold_metrics.append({
            "fold": fold_number,
            "test_start": test_start,
            "test_end": test_end,

            "persistence_mae": persistence_metrics["MAE"],
            "persistence_rmse": persistence_metrics["RMSE"],

            "gradient_boosting_mae": gb_metrics["MAE"],
            "gradient_boosting_rmse": gb_metrics["RMSE"],
        })

    if not all_predictions:
        raise ValueError(
            "Walk-forward validation produced no predictions."
        )

    # Combine predictions from all folds.
    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_metrics = pd.DataFrame(all_fold_metrics)

    # Calculate final metrics using all test predictions together.
    persistence_overall = calculate_metrics(
        predictions["actual_power_kw"],
        predictions["persistence_prediction"],
    )

    gb_overall = calculate_metrics(
        predictions["actual_power_kw"],
        predictions["gradient_boosting_prediction"],
    )

    overall_metrics = pd.DataFrame([
        {
            "Model": "24-hour Persistence",
            **persistence_overall,
        },
        {
            "Model": "Gradient Boosting",
            **gb_overall,
        },
    ])

    return ValidationResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
    )