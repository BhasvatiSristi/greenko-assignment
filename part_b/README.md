# Part B Renewable Generation Forecasting

## Problem

This part of the assignment uses the approximately 90-day turbine telemetry dataset to build a short-horizon next-day generation forecast. The forecasting target is the fleet-level hourly `total_power_kw` aggregated from the cleaned turbine telemetry.

## Dataset

- Raw telemetry: `data_pack/raw/turbine_telemetry_90day.csv`
- Cleaned telemetry: `data_pack/turbine_90_cleaned.csv`
- DAM price data exists in `data_pack/raw/dam_price_90day.csv`, but it is not used in the Part B forecasting workflow.

## Why EDA Is Kept In A Notebook

EDA is retained as a notebook because it documents the exploratory reasoning and evidence used to identify the data-quality issues. Reusable preprocessing, feature engineering, modelling, and evaluation logic are implemented in Python so the forecasting workflow can be reproduced without manually executing notebook cells.

## Data-Quality Findings

The EDA notebook preserves the discovery path rather than starting from the known anomalies. The main findings are:

- The telemetry history contains missing hourly turbine combinations, so the raw history is not perfectly complete.
- T05 has a wind-speed scaling issue during 2026-03-26 through 2026-04-04 23:00.
- T03 shows an anomalous power period during 2026-05-04 through 2026-05-11 23:00.
- The corrections were validated in the notebook before being copied into reusable preprocessing code.

## Cleaning Decisions

The forecasting pipeline uses only the validated cleaning logic from the EDA:

- Sort the raw telemetry chronologically.
- Preserve the raw dataset unchanged.
- Add `wind_speed_clean`, `power_anomaly_flag`, and `power_kw_clean` as derived columns.
- Apply the validated T05 wind-speed correction only in the flagged window.
- Flag the T03 anomalous power window without overwriting the measured power values.
- Save the cleaned copy to `data_pack/turbine_90_cleaned.csv`.

## Forecasting Target

The target is `total_power_kw`, which is the hourly sum of `power_kw_clean` across all turbines.

## Features

The forecasting notebook used these exact features:

- `lag_1`
- `lag_24`
- `lag_48`
- `lag_168`
- `rolling_mean_24`
- `rolling_std_24`
- `hour`
- `day_of_week`

These are created from past information only. No future values are introduced.

## Baseline

The baseline is 24-hour persistence, implemented as `prediction(t) = actual(t - 24 hours)`.

## Stronger Model

The stronger model is `GradientBoostingRegressor` with the notebook hyperparameters:

- `n_estimators=100`
- `learning_rate=0.05`
- `max_depth=3`
- `random_state=42`

## Walk-Forward Validation

The evaluation is time-aware. The final 28 complete days are split into four non-overlapping 7-day folds. Each fold trains on all earlier observations and predicts the current test window. Random splitting is not used because it would let future observations leak into training.

## Metrics

The pipeline reports:

- MAE
- RMSE
- MAPE
- sMAPE

The metric definitions follow the notebook convention, including the zero-denominator mask for percentage error calculations.

## Results

Current notebook reference results from the walk-forward comparison are:

| Model | MAE | RMSE | MAPE | sMAPE |
| --- | ---: | ---: | ---: | ---: |
| 24-hour Persistence | 846.2997023809525 | 1136.7739577077841 | 165.15437163122942 | 60.245313363514796 |
| Gradient Boosting | 666.2118633708275 | 874.415872423686 | 130.6448320787672 | 47.91270142873522 |

The stronger model beats the persistence baseline across all reported metrics.

## Limitations

- The telemetry history still contains missing hourly turbine combinations in the raw source.
- The pipeline follows the notebook-derived validation window and model choice rather than redesigning the forecast problem.
- DAM prices are not part of the Part B solution.

## How To Run

From the repository root:

```bash
python -m part_b.run_forecasting
```

This will:

1. Load the raw telemetry.
2. Apply the validated preprocessing.
3. Build the forecasting features.
4. Run the persistence baseline and gradient boosting model.
5. Perform walk-forward validation.
6. Save the cleaned data and comparison outputs into `data_pack/`.
