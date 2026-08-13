from __future__ import annotations

from pathlib import Path

from data_loader import RAW_TELEMETRY_PATH
from evaluation import run_walk_forward_validation
from features import FEATURE_COLUMNS, TARGET_COLUMN, build_forecasting_frame
from preprocessing import build_clean_telemetry


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON_PATH = ROOT / "data_pack" / "model_comparison.csv"
FORECAST_RESULTS_PATH = ROOT / "data_pack" / "forecast_results.csv"
FOLD_METRICS_PATH = ROOT / "data_pack" / "fold_metrics.csv"


def main() -> None:
    clean_frame = build_clean_telemetry(raw_path=RAW_TELEMETRY_PATH)
    forecast_frame = build_forecasting_frame(clean_frame)

    result = run_walk_forward_validation(
        forecast_frame=forecast_frame,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )

    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(FORECAST_RESULTS_PATH, index=False)
    result.fold_metrics.to_csv(FOLD_METRICS_PATH, index=False)
    result.overall_metrics.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("Model comparison")
    print(result.overall_metrics.to_string(index=False))
    print()
    print("Fold metrics")
    print(result.fold_metrics.to_string(index=False))
    print()
    print(f"Saved cleaned data to: {Path(clean_frame.attrs.get('saved_path', ROOT / 'data_pack' / 'turbine_90_cleaned.csv'))}")
    print(f"Saved forecast results to: {FORECAST_RESULTS_PATH}")
    print(f"Saved fold metrics to: {FOLD_METRICS_PATH}")
    print(f"Saved model comparison to: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
