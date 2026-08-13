# AI Engineering Take-Home Assignment

## Agent Development & Renewable Energy Forecasting

**Greenko — Power-to-X Renewable Energy Operations**

This project contains two connected parts:

- **Part A:** An AI-powered renewable-energy operations assistant that answers questions about turbine telemetry, DAM prices, and RFNBO compliance.
- **Part B:** A renewable-generation forecasting pipeline that first investigates the 90-day telemetry for data-quality problems, applies only validated corrections, engineers time-series features, and evaluates forecasting models using walk-forward validation.

The overall engineering objective is to make operational decisions from imperfect renewable-energy data while keeping the reasoning **grounded, reproducible, and explainable**.

---

# 1. Project Structure

```text
GREENKO-ASSIGNMENT/
│
├── data_pack/
│   ├── raw/
│   │   └── turbine_telemetry_90day.csv
│   │
│   ├── compliance_rulebook.docx
│   ├── fold_metrics.csv
│   ├── forecast_results.csv
│   ├── greenko_part_a.db
│   ├── model_comparison.csv
│   ├── README_DATA.docx
│   └── turbine_90_cleaned.csv
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── part_a_agent.ipynb
│   ├── part_b_eda.ipynb
│   └── part_b_forecasting.ipynb
│
├── part_a/
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── data_query.py
│   │   └── rulebook.py
│   │
│   └── agent.py
│
├── part_b/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── preprocessing.py
│   └── run_forecasting.py
│
├── venv/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py
└── WRITEUP.md
```

## Role of the main files

### Part A

- `part_a/agent.py` — main operations-agent orchestration layer.
- `part_a/tools/data_query.py` — converts natural-language structured-data questions into SQL and executes the SQL against the SQLite database.
- `part_a/tools/calculator.py` — deterministic numerical tools such as capacity factor, percentage improvement, percentage difference, and temporal-correlation coverage.
- `part_a/tools/rulebook.py` — loads the supplied RFNBO rulebook and answers compliance questions using only that document.
- `data_pack/greenko_part_a.db` — structured SQLite data used by the operations assistant.
- `data_pack/compliance_rulebook.docx` — source document for compliance questions.

### Part B

- `part_b/data_loader.py` — loads and validates the raw/cleaned telemetry.
- `part_b/preprocessing.py` — applies the validated T05 wind-speed correction and flags the T03 power anomaly.
- `part_b/features.py` — aggregates turbine data into hourly fleet generation and creates forecasting features.
- `part_b/models.py` — defines the persistence baseline and Gradient Boosting model.
- `part_b/evaluation.py` — performs walk-forward validation and calculates MAE, RMSE, MAPE, and sMAPE.
- `part_b/run_forecasting.py` — runs the complete forecasting pipeline from raw data to saved results.
- `notebooks/part_b_eda.ipynb` — exploratory data-quality investigation.
- `notebooks/part_b_forecasting.ipynb` — forecasting experiments and evaluation.
- `data_pack/turbine_90_cleaned.csv` — processed telemetry.
- `data_pack/fold_metrics.csv` — fold-level validation results.
- `data_pack/forecast_results.csv` — predictions from the validation process.
- `data_pack/model_comparison.csv` — overall model comparison.

### Frontend

`streamlit_app.py` provides a single interface with:

- **Operations Agent**
- **Generation Forecast**

The application presents the project as one renewable-energy operations system rather than as two unrelated notebooks.

---

# 2. Problem Framing

The project addresses two related operational problems.

### Part A — Operations Intelligence

An operator may ask questions such as:

- What is the average power output of a turbine?
- What is its capacity factor?
- What is the average DAM price?
- How does generation compare across periods?
- What happens during high-DAM-price hours?
- What does the supplied RFNBO rulebook say about temporal correlation or matching?

The challenge is not simply generating a natural-language answer. The answer needs to be backed by the supplied data and rulebook.

Therefore, Part A separates:

**data retrieval → calculation → rulebook grounding → natural-language response**

rather than allowing the LLM to freely invent numerical or regulatory information.

### Part B — Renewable Generation Forecasting

The second problem is to forecast the next 24 hours of renewable generation from historical turbine telemetry.

Before modelling, the telemetry must be investigated because forecasting a corrupted signal can produce misleading results.

The workflow is therefore:

```text
Raw telemetry
      ↓
Data-quality investigation
      ↓
Identify abnormalities from the data
      ↓
Validate the suspected issues
      ↓
Apply targeted cleaning
      ↓
Aggregate fleet generation
      ↓
Create time-series features
      ↓
Persistence baseline
      ↓
Gradient Boosting
      ↓
Walk-forward validation
      ↓
Model comparison
      ↓
24-hour forecast
```

---

# 3. Part A — AI Operations Agent

## 3.1 Architecture

The current Part A implementation uses three main categories of tools:

```text
                     User Question
                          │
                          ▼
                    agent.py
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
     Data Query       Calculator       Rulebook
        Tool             Tools            Tool
          │               │                │
          ▼               ▼                ▼
     SQLite DB       Python maths     Rulebook DOCX
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                    Evidence-based
                       response
```

The important design principle is that the LLM is **not the source of truth**.

For structured data questions, the LLM generates SQL, but that SQL is executed against the actual SQLite database. For calculations, dedicated Python tools perform the arithmetic. For compliance, the supplied rulebook is loaded and passed as the only source to the compliance model.

---

## 3.2 Structured Data Querying

`data_query.py` provides the `query_data` LangChain tool.

It is designed for questions involving:

- turbine power
- wind speed
- availability
- fleet generation
- averages
- minimum/maximum values
- counts
- DAM prices
- dates and periods
- comparisons between telemetry and DAM prices

The process is:

```text
Natural-language question
          ↓
Get SQLite database schema
          ↓
Groq generates SQLite SQL
          ↓
Validate generated SQL
          ↓
Execute against SQLite in read-only mode
          ↓
Return generated SQL + database result
```

The SQL generation prompt restricts the model to `SELECT` or `WITH` queries and instructs it not to invent tables or columns.

The execution layer adds another safety boundary:

- `INSERT` is rejected.
- `UPDATE` is rejected.
- `DELETE` is rejected.
- `DROP` is rejected.
- `ALTER` is rejected.
- `CREATE` is rejected.
- `REPLACE` is rejected.
- `TRUNCATE` is rejected.
- `ATTACH` and `DETACH` are rejected.

The SQLite database is also opened in read-only mode.

This is important because the LLM is allowed to generate SQL, but it is **not allowed to modify the underlying data**.

---

## 3.3 Calculation Tools

`calculator.py` contains deterministic calculation tools.

### Capacity factor

```text
Capacity Factor =
Average Power / Rated Capacity
```

The default rated capacity in the tool is 2000 kW.

The tool returns both the decimal and percentage representation.

### Percentage improvement

For lower-is-better metrics such as MAE and RMSE:

```text
Improvement =
(Baseline - Model) / Baseline × 100
```

### Percentage difference

The tool uses the average absolute magnitude of the two values as the denominator.

### Temporal correlation coverage

```text
Correlation Coverage =
Correlated Hours / Total Hours × 100
```

These calculations are intentionally kept outside the LLM so that derived numerical values are reproducible.

---

# 4. Part A — RFNBO Compliance Tool

`rulebook.py` handles compliance questions.

The supplied document is:

```text
data_pack/compliance_rulebook.docx
```

The rulebook is loaded into memory and supplied to a dedicated Groq model.

The compliance prompt explicitly requires:

- use only the supplied rulebook;
- do not use outside knowledge;
- do not invent requirements;
- explicitly say when the rulebook does not contain enough information.

The tool is intended for questions involving:

- temporal correlation;
- hourly matching;
- monthly matching;
- electrolyzer requirements;
- renewable-generation correlation;
- RFNBO compliance requirements explicitly stated in the supplied document.

This is especially important for regulatory questions because a plausible-sounding answer is not sufficient. The system must distinguish between:

**"the rulebook says this"**

and

**"this is something I know from outside the rulebook."**

---

# 5. Part A — Why the Architecture Is Grounded

The main engineering decision in Part A is to separate **reasoning from evidence retrieval**.

For example:

```text
Question:
"What is the average power of T03?"

        ↓

Data Query Tool

        ↓

LLM generates SQL

        ↓

SQLite executes SQL

        ↓

Actual database result

        ↓

Agent produces response
```

For:

```text
"What is the capacity factor?"

        ↓

Relevant power value
        +
Rated capacity

        ↓

Calculator tool

        ↓

Capacity factor
```

For:

```text
"What does the RFNBO rulebook say about
temporal correlation?"

        ↓

Rulebook DOCX

        ↓

Compliance LLM
restricted to supplied document

        ↓

Rulebook-grounded response
```

This makes the system easier to debug and defend during an interview.

---

# 6. Part B — EDA and Data-Quality Investigation

The EDA was deliberately performed **before modelling**.

Instead of starting with a list of known abnormalities and immediately correcting them, the analysis follows a discovery-first approach:

```text
Basic data inspection
        ↓
Temporal completeness
        ↓
Distributions
        ↓
Wind-speed / power relationship
        ↓
Cross-turbine comparison
        ↓
Temporal anomaly investigation
        ↓
Validate suspicious periods
        ↓
Apply only justified corrections
```

This is important because the purpose of EDA was not simply to produce plots. It was to determine:

1. whether the data was structurally complete;
2. whether individual turbines behaved differently;
3. whether physical relationships looked reasonable;
4. whether suspicious periods could be explained;
5. whether an abnormal observation should be corrected, flagged, or left untouched.

---

# 7. EDA Findings

## 7.1 Missing hourly turbine combinations

The 90-day dataset contains:

```text
Expected:
90 days × 24 hours × 5 turbines
= 10,800 turbine-hour observations

Observed:
10,725 observations
```

Therefore, the supplied history contains **75 fewer turbine-hour combinations than a complete hourly 5-turbine panel**.

This was treated as a data-availability issue rather than an instruction to manufacture values.

### Decision

The raw data was not artificially filled.

The reasoning was:

> Missing telemetry is not automatically equivalent to zero generation.

Filling missing observations without knowing why they are missing could introduce artificial generation patterns and bias the forecasting model.

---

# 8. T05 Wind-Speed Anomaly

The EDA showed that T05 behaved differently from the other turbines in its wind-speed relationship.

The issue became visible through:

- turbine-level wind-speed distributions;
- wind-speed versus power relationships;
- comparison with the behaviour of other turbines;
- temporal inspection of the suspicious period.

The affected period was:

```text
T05
2026-03-26 00:00
to
2026-04-04 23:00
```

The validated scaling factor was:

```text
2.239306387561419
```

The corrected wind speed is therefore effectively:

```text
wind_speed_clean = wind_speed / 2.239306387561419
```

for the affected T05 period only.

### Why this was treated as a scaling issue

The anomaly appeared as a shifted wind-speed relationship rather than simply random noise.

The correction was therefore applied to the wind-speed measurement, not to power generation.

### Implementation

The preprocessing pipeline:

- preserves the original `wind_speed`;
- creates `wind_speed_clean`;
- applies the correction only to T05;
- applies it only inside the validated date window.

This keeps the raw observation available for auditability.

---

# 9. T03 Power Anomaly

A separate abnormal period was identified for T03.

The affected period was:

```text
T03
2026-05-04 00:00
to
2026-05-11 23:00
```

The EDA showed a distinct abnormal power pattern through temporal and physical-relationship checks.

Unlike the T05 wind-speed problem, there was no sufficiently safe scaling transformation that could be justified for the T03 power values.

### Decision

The T03 values were **not rescaled**.

Instead, the pipeline creates:

```text
power_anomaly_flag
```

and marks the affected observations.

The measured:

```text
power_kw
```

is retained through:

```text
power_kw_clean
```

rather than replacing it with an invented value.

This is a deliberate conservative treatment.

---

# 10. Final Cleaning Strategy

The cleaned dataset preserves the original observations and adds derived columns:

```text
Original columns
      +
wind_speed_clean
      +
power_anomaly_flag
      +
power_kw_clean
```

The output is:

```text
data_pack/turbine_90_cleaned.csv
```

The cleaning implementation therefore follows three principles:

### 1. Do not overwrite raw data

The original measurement remains available.

### 2. Correct only validated issues

The T05 correction is limited to the identified turbine and time window.

### 3. Flag what cannot safely be corrected

The T03 power anomaly is flagged rather than artificially repaired.

---

# 11. Part B — Forecasting Dataset

After cleaning, the individual turbine observations are aggregated into an hourly fleet-level dataset.

The target is:

```text
total_power_kw
```

which is calculated as:

```text
sum of power_kw_clean across turbines
```

The feature-building stage also creates:

```text
mean_wind_speed
available_turbines
```

and reindexes the fleet data to an hourly timeline.

---

# 12. Forecasting Features

The model uses a deliberately compact feature set:

| Feature | Purpose |
|---|---|
| `lag_1` | Previous hour generation |
| `lag_24` | Same hour on the previous day |
| `lag_48` | Same hour two days earlier |
| `lag_168` | Same hour one week earlier |
| `rolling_mean_24` | Recent 24-hour generation level |
| `rolling_std_24` | Recent 24-hour variability |
| `hour` | Daily seasonality |
| `day_of_week` | Weekly seasonality |

The rolling features are calculated from shifted target values so that the model does not accidentally use the current or future target while constructing features.

This is important for preventing temporal leakage.

---

# 13. Forecasting Models

## 13.1 24-hour Persistence Baseline

The first model is intentionally simple.

For each hour:

```text
Prediction(t) = Actual generation at t - 24 hours
```

This gives a meaningful baseline because renewable generation often contains daily patterns.

A stronger machine-learning model should demonstrate improvement over this baseline.

---

## 13.2 Gradient Boosting

The second model is:

```text
GradientBoostingRegressor
```

with:

```text
n_estimators = 100
learning_rate = 0.05
max_depth = 3
random_state = 42
```

The model is trained only on historical data available before each validation period.

---

# 14. Time-Aware Validation

A random train/test split would not be appropriate here.

For forecasting:

```text
Past → Training
Future → Testing
```

must be respected.

The evaluation therefore uses **walk-forward validation** over the final 28 days, organised into consecutive 7-day folds.

Conceptually:

```text
Fold 1:
Train → historical data
Test  → Week 1

Fold 2:
Train → everything before Week 2
Test  → Week 2

Fold 3:
Train → everything before Week 3
Test  → Week 3

Fold 4:
Train → everything before Week 4
Test  → Week 4
```

The implementation trains a fresh Gradient Boosting model on the historical portion of each fold and compares it with the 24-hour persistence baseline.

---

# 15. Evaluation Metrics

The pipeline reports:

### MAE

Mean Absolute Error.

It gives the average magnitude of the prediction error in kW.

### RMSE

Root Mean Squared Error.

It penalises larger errors more heavily than MAE.

### MAPE

Mean Absolute Percentage Error.

The implementation excludes observations where the actual value is zero from the MAPE calculation.

### sMAPE

Symmetric Mean Absolute Percentage Error.

This provides a more symmetric percentage-based error measure.

---

# 16. Forecasting Results

| Model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|
| 24-hour Persistence | 846.30 | 1136.77 | 165.15% | 60.25% |
| Gradient Boosting | 666.21 | 874.42 | 130.64% | 47.91% |

Gradient Boosting improves on the persistence baseline across all four reported metrics.

The improvement is particularly meaningful because the comparison is performed using time-aware validation rather than a random split.

At the same time, the remaining error shows that the forecasting problem is noisy and that data quality has a meaningful effect on the achievable accuracy.

---

# 17. End-to-End Forecasting Pipeline

The reusable forecasting pipeline is implemented in:

```text
part_b/run_forecasting.py
```

The sequence is:

```text
Load raw telemetry
        ↓
Apply validated cleaning
        ↓
Build hourly fleet frame
        ↓
Create lag / rolling / calendar features
        ↓
Run walk-forward validation
        ↓
Train persistence + Gradient Boosting
        ↓
Calculate metrics
        ↓
Save outputs
```

The generated outputs are:

```text
data_pack/turbine_90_cleaned.csv
data_pack/forecast_results.csv
data_pack/fold_metrics.csv
data_pack/model_comparison.csv
```

---

# 18. Streamlit Application

`streamlit_app.py` provides the final user-facing interface.

The application contains two modules:

```text
Greenko Renewable Energy Operations

├── 🤖 Operations Agent
│     ├── Structured telemetry / DAM questions
│     ├── Calculations
│     └── RFNBO compliance questions
│
└── 📈 Generation Forecast
      ├── 24-hour forecast
      ├── hourly forecast table
      └── model metrics
```

The application therefore connects the two technical parts into one operational interface.

---

# 19. Reproducibility

The project separates:

### Raw data

```text
data_pack/raw/
```

### Processed data

```text
data_pack/turbine_90_cleaned.csv
```

### Exploratory notebooks

```text
notebooks/
```

### Reusable Python pipeline

```text
part_b/
```

### Final results

```text
data_pack/
```

This separation makes it possible to distinguish between:

- exploration;
- validated decisions;
- reusable production-style code;
- generated outputs.

---

# 20. Assumptions and Limitations

## Part A

- The operations assistant is limited to the supplied structured data and compliance rulebook.
- SQL generation is performed by an LLM, so SQL validation is necessary.
- The structured database is intentionally read-only.
- Compliance answers are restricted to the supplied rulebook.
- The system is not intended to replace a production compliance or regulatory review process.
- The current system is designed around the specific question types required by the assignment rather than being a general enterprise agent.

## Part B

- The available historical dataset covers approximately 90 days.
- Missing turbine-hour combinations are not artificially filled.
- The T05 correction is based on the validated scaling relationship found during EDA.
- The T03 anomaly is flagged rather than reconstructed.
- The feature set is intentionally compact.
- Only a Gradient Boosting model and a persistence baseline are compared.
- The model does not explicitly use external weather forecasts.
- Long-term seasonal effects are difficult to estimate from the available history.
- The reported percentage errors should be interpreted carefully because renewable generation can approach low values, which can make percentage-based metrics unstable.

---

## 21. What I Would Do With More Time

### Part A — Agent

1. **Reduce document-lookup latency**  
   The current document lookup takes noticeable time to return an answer. I would explore faster approaches such as retrieving only the relevant rule sections instead of processing the full document for every question.

2. **Improve tool performance and reliability**  
   I would add more testing around tool selection, edge cases, and response latency to make the agent more reliable for a wider range of operational questions.

### Part B — Forecasting

3. **Compare more forecasting models**  
   I would train at least two additional models and compare them with the current persistence baseline and Gradient Boosting model. I would then select the best-performing model based on the same walk-forward evaluation metrics.

4. **Tune the selected model**  
   After identifying the strongest model, I would tune its hyperparameters using time-aware validation rather than standard random cross-validation.

5. **Add better forecasting inputs**  
   I would explore additional useful features, such as weather forecasts, if they were available, since future wind conditions can provide information that historical generation alone cannot capture.

There are several directions I would take if this were being developed beyond the take-home assignment.

## Part A — Agent

### 1. Better intent routing

Instead of relying only on a fixed set of expected question patterns, I would introduce a more explicit intent classification layer.

For example:

```text
Question
   ↓
Intent classification
   ↓
Structured data / Calculation / Compliance
   ↓
Relevant tool
```

This would make the system more robust to differently worded operator questions.

### 2. SQL validation and query testing

I would add a stronger SQL validation layer using:

- SQL parsing;
- schema validation;
- query complexity limits;
- execution time limits;
- additional tests for generated queries.

### 3. Tool observability

I would log:

```text
User question
→ selected tool
→ generated SQL / calculation
→ tool result
→ final answer
```

This would make debugging and auditing easier.

### 4. Compliance retrieval

If the rulebook became larger, I would replace the current whole-document prompting approach with structured document retrieval or RAG.

This would allow:

```text
Question
   ↓
Relevant rule sections
   ↓
LLM
   ↓
Answer + source section
```

rather than passing the entire document every time.

---

# 22. Part B — Further Improvements

## 1. More sophisticated anomaly detection

The current EDA uses distributions, relationships, cross-turbine comparisons, and temporal investigation.

With more time, I would add automated monitoring using methods such as:

- rolling z-scores;
- Isolation Forest;
- robust statistical thresholds;
- change-point detection;
- turbine-specific control limits.

The important principle would remain:

> Automated detection should identify candidates; domain/physical validation should decide whether they are actually abnormal.

---

## 2. Better treatment of missing data

The current approach intentionally avoids fabricating missing observations.

A production system could distinguish between:

```text
sensor outage
maintenance
communication failure
true zero generation
```

and then apply different treatment depending on the reason.

---

## 3. Weather information

The forecasting model currently depends heavily on historical generation patterns.

A major improvement would be adding external weather forecasts:

```text
Wind speed forecast
Wind direction
Temperature
Air pressure
Weather uncertainty
        ↓
Generation forecast
```

For wind generation, this could provide information about future conditions that historical lags alone cannot know.

---

## 4. More forecasting models

I would compare the current Gradient Boosting model with:

- Random Forest;
- XGBoost / LightGBM;
- Random Forest-style ensemble baselines;
- SARIMAX;
- other time-series approaches;
- potentially sequence models if the dataset were large enough.

The models should still be evaluated using the same time-aware validation strategy.

---

## 5. Hyperparameter tuning

The current model uses a deliberately small fixed configuration.

With more time, I would tune:

```text
n_estimators
learning_rate
max_depth
min_samples_split
min_samples_leaf
subsample
```

using time-aware cross-validation rather than ordinary random cross-validation.

---

## 6. Prediction intervals

A production forecasting system should ideally provide more than:

```text
Expected generation = X kW
```

It should provide something like:

```text
Expected generation = X kW

Likely range = [lower, upper]
```

This would allow operators to understand forecast uncertainty.

---

## 7. Forecast monitoring

After deployment, I would monitor:

- MAE over time;
- forecast bias;
- prediction drift;
- actual-vs-predicted generation;
- turbine-level data-quality issues;
- model degradation.

This would turn the forecasting model into a continuously monitored system rather than a one-time prediction script.

---

# 23. Key Engineering Decisions

The most important decisions in this project were:

### Part A

**Do not let the LLM invent evidence.**

Use tools and the supplied sources for actual data and rules.

### Part B

**Do not start modelling before understanding the data.**

The EDA identifies structural and physical abnormalities first.

### Cleaning

**Do not automatically "fix" every anomaly.**

Correct T05 because there is a defensible scaling relationship.

Flag T03 because a safe correction could not be justified.

### Forecasting

**Do not randomly split time-series data.**

Use walk-forward validation so that the evaluation resembles real deployment.

### Reproducibility

**Keep raw, cleaned, exploratory, reusable, and generated artifacts separate.**

This makes the reasoning easier to audit and explain.

---

# 24. Conclusion

This project combines an evidence-grounded AI operations assistant with a data-quality-aware renewable-generation forecasting pipeline.

Part A demonstrates how an LLM can interact with structured operational data and a compliance document while keeping the actual evidence retrieval and calculations controlled.

Part B demonstrates a complete forecasting workflow:

```text
EDA
→ anomaly identification
→ validated cleaning
→ feature engineering
→ baseline
→ machine learning
→ time-aware validation
→ model comparison
→ forecast
```

The central engineering lesson across both parts is:

> **Use AI to assist reasoning and communication, but keep the underlying evidence, calculations, and data-quality decisions traceable and defensible.**
