from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def make_persistence_prediction(frame: pd.DataFrame, baseline_column: str = "lag_24") -> pd.Series:
    return frame[baseline_column].copy()


def make_gradient_boosting_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def fit_gradient_boosting(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "total_power_kw",
) -> GradientBoostingRegressor:
    model = make_gradient_boosting_model()
    model.fit(train_frame[feature_columns], train_frame[target_column])
    return model
