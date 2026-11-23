# kse-load-forecast

Day-ahead hourly electricity demand forecasting for the Polish power system (KSE), benchmarked against
the official forecast published by the transmission system operator (PSE).

**Status:** data pipeline, exploratory analysis and model selection with time-series cross-validation.
Planned: forecast-weather inputs, experiment tracking (MLflow), orchestration (Airflow), serving (FastAPI),
GCP (GCS + BigQuery), Docker.

## Problem

Every day, predict the hourly national demand [MW] for the next day (24 values; 23/25 on DST change days)
and compare against PSE's own day-ahead forecast (`load_fcst`).

## Data

| Source | What | Resolution |
|---|---|---|
| [PSE API](https://api.raporty.pse.pl/api/kse-load) (`kse-load`) | actual demand (target) and PSE forecast (benchmark), since 2024-06-14 | 15 min |
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | temperature, apparent temperature, solar radiation, cloud cover, wind, precipitation for 8 largest cities | 1 h |

Weather is aggregated into one national series, weighted by city population.

## Method

- **Features** (one row per UTC hour): calendar in Polish local time (hour, weekday, month, public holidays,
  bridge days, Christmas period), weather and heating/cooling degrees, demand lags of **48 h or more**
  (nothing that is unknown at forecast time).
- **Time handling:** everything is computed in UTC (no DST gaps or duplicates); calendar features are
  derived in `Europe/Warsaw`.
- **Validation:** expanding-window cross-validation with 14 monthly folds (2025-07 to 2026-08); each model
  is retrained on all data before the test month. Data from 2026-09-01 is a **holdout**, excluded from EDA
  and model selection, reserved for the final evaluation.
- **Metric:** MAE [MW]; RMSE and bias reported alongside.

## Results (cross-validation, 14 folds)

| Model | MAE [MW] | Bias [MW] | Months better than PSE |
|---|---|---|---|
| PSE official forecast | 433 | −157 | – |
| **Ensemble** (LightGBM ratio target + LightGBM linear trees) | **496** | +108 | 6 / 14 |
| LightGBM | 535 | +130 | 2 / 14 |
| Ridge regression | 666 | +163 | 1 / 14 |
| Naive (same hour last week) | 990 | +5 | 0 / 14 |

Key findings:
- The model does not beat PSE yet. About 60 % of the gap comes from winter: the 2025/26 winter was much colder
  than anything in the training history (down to −15.6 °C vs −9.8 °C), and tree models cannot extrapolate.
  Models with linear components reduce the error in frost but still underestimate demand.
- Outside winter the ensemble is within ~30 MW of PSE.
- Bridge-day and Christmas-period features (motivated by EDA) cut the error on those days substantially.
- **Caveat:** features use *observed* weather, while PSE only had *forecast* weather, so these results are
  optimistic. Switching to archived weather forecasts is the next step.

## Project structure

```
src/kse_forecast/
    config.py        paths, time zone, holdout and CV settings
    pse.py           PSE API client, incremental update, data quality checks
    weather.py       Open-Meteo client, population-weighted national average
    features.py      feature table (1 row = 1 hour)
    validation.py    holdout split and monthly expanding-window CV folds
    models.py        candidate models (sklearn-compatible)
    evaluation.py    metrics and cross-validation runner
    __main__.py      command line interface
tests/               pytest: DST handling, no leakage in lags and folds, data contracts
notebooks/
    01_eda.ipynb              exploratory analysis (development data only)
    02_model_selection.ipynb  cross-validation and model choice
data/                not in git; recreated by the pipeline
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .

python -m kse_forecast update-data     # download / incrementally update PSE and weather data
python -m kse_forecast build-features  # build data/processed/features.parquet
python -m kse_forecast cv              # cross-validate all candidate models
pytest                                 # run tests
ruff check .                           # lint
```

The first `update-data` downloads the full history (around 900 API calls, several minutes);
later runs only fetch the missing days.
