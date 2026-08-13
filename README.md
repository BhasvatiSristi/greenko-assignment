# Greenko AI Engineering Take-Home Assignment

This repository contains my completed Greenko take-home assignment, split into two parts:

- **Part A**: a renewable-energy operations agent that answers questions over turbine telemetry, DAM prices, and a compliance rulebook.
- **Part B**: a next-day renewable generation forecasting workflow built from the 90-day telemetry dataset.
- **Streamlit demo**: a single UI that presents Part A and Part B together for review.

## Project Structure

- `part_a/` - Part A agent, tools, and tests
- `part_b/` - Part B preprocessing, forecasting, and evaluation code
- `notebooks/` - EDA and forecasting notebooks used for the assignment write-up
- `data_pack/` - raw data, cleaned data, and model outputs
- `models/` - saved forecasting model artifact
- `streamlit_app.py` - root Streamlit application entry point
- `WRITEUP.md` - final written project summary

## How To Run

### 1. Install dependencies

Use the provided virtual environment if available, then install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
### 2. Run the Streamlit demo

The Streamlit app combines Part A and Part B in one interface:

```bash
streamlit run streamlit_app.py
```

## What Part A Does

Part A is a small LangChain + Groq agent. It uses structured tools for telemetry and DAM queries, deterministic calculator logic for derived metrics like capacity factor, and a rulebook lookup for compliance questions. The goal is grounded, evidence-based answers rather than free-form guessing.

## What Part B Does

Part B uses the 90-day turbine telemetry data to build a short-horizon next-day generation forecast. The EDA notebook documents the data-quality investigation and the validated cleaning decisions, while the Python modules handle reusable preprocessing, feature engineering, walk-forward evaluation, and model comparison.

## Notes

- The raw data in `data_pack/raw/` is left untouched.
- The cleaned telemetry used for forecasting is saved to `data_pack/turbine_90_cleaned.csv`.
- The saved forecasting model is `models/gradient_boosting_model.pkl`.
- The main Streamlit UI is intentionally simple and is meant for demonstration, not production deployment.