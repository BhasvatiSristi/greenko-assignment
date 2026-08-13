# AI Engineering Take-Home Assignment

## Agent Development & Renewable Energy Forecasting

**Greenko — Power-to-X Renewable Energy Operations**

This submission covers two connected engineering tasks built from the supplied repository artifacts. Part A is a grounded operational assistant over turbine telemetry, DAM prices, and a short compliance rulebook. Part B is a next-day renewable generation forecasting pipeline built from the 90-day telemetry data after notebook-based data-quality investigation and validated cleaning.

## 1. Problem Framing

The core problem is to make reliable decisions from imperfect renewable-energy data. In Part A, the assistant answers operator questions about turbine behaviour, market conditions, and RFNBO-style compliance using deterministic data and rulebook tools rather than free-form guessing. In Part B, the workflow turns the cleaned telemetry into a short-horizon next-day forecast so that generation planning can account for daily seasonality, recent history, and the effect of data-quality issues on downstream modelling. The two parts share the same theme: keep the reasoning grounded in the source data and make the final result easy to defend in an interview.

## 2. Conceptual Framework

### Part A

Part A uses deterministic routing to send each question to the smallest tool that can answer it, then uses Groq only to phrase the tool output as a concise response when available. That keeps the system debuggable and reduces the risk of hallucinated calculations or compliance claims.

```text
User Question
      ↓
Deterministic routing in agent.py
      ↓
┌────────────┬──────────────┬──────────────┐
│ Telemetry  │ Calculation  │  Compliance  │
│   tools    │    tools     │   rulebook   │
└────────────┴──────────────┴──────────────┘
      ↓
Evidence-based answer
(Groq synthesis if enabled)
```

### Part B

Part B keeps the EDA in notebooks because the notebook captures the exploratory reasoning chain: what was observed first, what was investigated next, why a turbine or period was suspected, and what evidence justified the correction. The reusable pieces were moved into Python so the forecast can be rerun without manually executing notebook cells.

```text
Raw telemetry
      ↓
EDA / data-quality investigation
      ↓
Validated cleaning
      ↓
Feature engineering
      ↓
Time-aware forecasting
      ↓
Baseline + stronger model
      ↓
Walk-forward evaluation
      ↓
Metrics + model comparison
```

## 3. Part A: Agentic System

The Part A implementation is intentionally small. The data layer reads the supplied 14-day turbine telemetry CSV and the 14-day DAM CSV, parses timestamps, and validates turbine IDs against the expected T01–T05 set. The agent layer in `agent.py` routes questions with explicit keyword and pattern checks instead of asking the LLM to choose tools freely.

The final response is synthesized by ChatGroq (`llama-3.3-70b-versatile`, temperature 0) from deterministic tool output, or returned directly if no Groq key is available.

### Tools

| Question type | Tool path |
|---|---|
| Average power for T01 | `get_turbine_summary` → `DataQueryTool.turbine_summary` |
| Capacity factor for T05 | `get_turbine_capacity_factor` → deterministic capacity-factor calculation |
| Average DAM price | `get_dam_summary` → `DataQueryTool.dam_summary` |
| First vs second week comparison | `compare_weekly_capacity_factor` |
| High-DAM-price output comparison | `compare_output_on_high_dam_price` |
| RFNBO / temporal correlation question | `lookup_compliance_rule` → `RulebookTool.lookup` |

### Failure Handling

Failure handling is explicit. Unknown turbine IDs return a clear invalid/unknown message, empty or unsupported questions are rejected, and unsupported compliance topics return no relevant rule found. The supplied rulebook is the source of truth; the agent does not invent regulatory requirements when the document does not provide them.

The tests in `part_a/tests/test_tools.py` confirm the intended behaviour, including the T01 summary, T05 capacity factor, DAM summary, and graceful handling of unsupported rulebook topics.

## 4. Part B: Data Quality and EDA

The EDA notebook follows a discovery-first sequence rather than starting from the known anomalies. It begins with basic data quality checks and temporal completeness, then moves through distributions, wind-speed versus power relationships, and cross-turbine comparisons before narrowing into temporal anomaly detection and correction validation. That order matters because it shows how the issue was discovered from the data itself.

| Issue | Evidence | Likely cause | Treatment |
|---|---|---|---|
| Missing hourly turbine combinations | 90-day telemetry has 10,725 rows rather than a fully complete 10,800-row hourly panel | Incomplete hourly coverage in the supplied history | Leave the raw data unchanged; do not fabricate values |
| T05 wind-speed anomaly | T05 distribution and power-curve scatter showed a shifted wind-speed relationship; the notebook estimated a multiplicative factor of about 2.24 over 2026-03-26 to 2026-04-04 23:00 | Sensor scaling / unit issue | Create `wind_speed_clean` by scaling only the affected window in the processed copy |
| T03 power anomaly | A distinct anomalous period appeared in temporal and physical-relationship checks over 2026-05-04 to 2026-05-11 23:00 | Power-measurement anomaly that could not be safely rescaled | Flag the period with `power_anomaly_flag` and keep the measured power values |

The final cleaned dataset is saved as `data_pack/turbine_90_cleaned.csv` and preserves the original raw columns plus `wind_speed_clean`, `power_anomaly_flag`, and `power_kw_clean`. That separation keeps the investigation reproducible while preventing the raw telemetry from being overwritten.

## 5. Part B: Forecasting

The forecasting target is hourly fleet `total_power_kw`, built by summing `power_kw_clean` across turbines. The horizon is next-day generation in a time-aware sense: the model is evaluated on hourly steps, with the evaluation window organised into four consecutive 7-day walk-forward folds across the last 28 complete days.

This is a forecasting problem, so random train/test splitting would leak future information into training and would not reflect how the model would be used operationally.

### Features

- `lag_1`
- `lag_24`
- `lag_48`
- `lag_168`
- `rolling_mean_24`
- `rolling_std_24`
- `hour`
- `day_of_week`

The calendar and lag features are safe because they use only past information available at forecast time.

### Models

**24-hour Persistence Baseline**

The baseline predicts each hour using the previous day's value for the same hour.

**Gradient Boosting**

`GradientBoostingRegressor` with:

- `n_estimators=100`
- `learning_rate=0.05`
- `max_depth=3`
- `random_state=42`

### Results

| Model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|
| 24-hour Persistence | 846.30 | 1136.77 | 165.15% | 60.25% |
| Gradient Boosting | 666.21 | 874.42 | 130.64% | 47.91% |

The stronger model beats the persistence baseline on every reported metric. The absolute errors are still large enough to show that the problem is noisy and sensitive to data-quality issues, but the improvement is meaningful and stable across the walk-forward folds.

The pipeline saves the cleaned data, fold metrics, forecast results, and comparison table into `data_pack/` so the run can be reproduced from the repository root with:

```bash
python -m part_b.run_forecasting
```

## 6. Brief Prior-Art Scan

| Source | Why it matters here |
|---|---|
| Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., OTexts, 2021; online edition updated 2026) | Background for time-series EDA, naive baselines, and practical forecasting workflow. |
| scikit-learn documentation for `TimeSeriesSplit` | Supports time-ordered validation and expanding training sets for forecasting. |
| Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2023) | Background for tool-using language-model systems that interleave reasoning with external actions. |

## 7. Assumptions and Limitations

- Part A is limited to the supplied 14-day telemetry, DAM series, and short rulebook.
- The agent uses deterministic routing rather than open-ended tool planning.
- Part B is limited to the supplied 90-day history and does not address long-term seasonality.
- Missing hourly combinations are not inferred or replaced.
- The model family is intentionally simple and is not a full production forecasting stack.

## 8. What I Would Do With More Time

- Add richer routing tests and conversation traces for Part A.
- Expand the compliance layer with structured document parsing if the rulebook grows.
- Add probabilistic forecasts or prediction intervals.
- Add automated data-quality monitoring and drift checks.
- Compare Gradient Boosting with alternative time-series models using the same walk-forward evaluation.

## 9. Conclusion

Part A shows grounded agentic reasoning over operational, market, and compliance data, with deterministic tools providing the numerical and rule-based evidence. Part B shows the traditional forecasting workflow: inspect the data carefully, fix only validated issues, build safe lagged features, and evaluate with time-aware splits.

Across both parts, the main engineering theme is the same: **make decisions from imperfect operational data without losing traceability.**
