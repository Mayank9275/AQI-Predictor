# AQI Predictor

This student project uses hourly CPCB measurements for Chandni Chowk, Delhi, to estimate AQI one calendar hour ahead. It compares machine learning with a persistence baseline, which assumes the next hour's AQI will equal the current hour's AQI.

The `AQI` column is treated as the project's existing provisional target. Its calculation has not been independently validated against every official CPCB sub-index rule, and this workflow does not recalculate it.

## Data

The training script reads `Data/air_quality_ml_ready.csv`. This file keeps the AQI labels and the pollutant and weather measurements before the final full-data missing-value fill. Missing feature values are imputed inside each model pipeline using medians learned from the training partition only. Missing AQI targets are not filled.

`Data/Final_Air_quality.csv` is a separate processed file with 27 columns. The training script does not write to it. Raw data in `Data/Raw Data/` is also left untouched.

## Prediction setup

For a forecast made at time *t*, the target is AQI at exactly *t + 1 hour*. Rows are evaluated only when the next calendar hour has a valid AQI label. A regular hourly index is used to build features, so a missing hour breaks lag and rolling history rather than allowing a feature to reach across a sensor outage.

Inputs include current-hour pollutant and weather readings, current AQI, calendar hour/day/month, AQI lags at 1, 3, 6, and 24 hours, and rolling AQI mean and standard deviation over 3, 6, and 24 hours. Rolling windows use only current and earlier observations and require every hour in the window to be present.

Rows are kept in time order and split into 70% training, 15% validation, and 15% test. Random Forest and Extra Trees settings are selected using validation MAE. The median imputer is fitted only when the candidate pipeline is trained on the training partition. Test metrics are then reported for persistence and the best validation-selected setting from each model family.

## Metrics

- **MAE** (mean absolute error) is the average absolute AQI prediction error. Lower is better.
- **RMSE** (root mean squared error) penalizes large errors more heavily. Lower is better.
- **R²** compares explained variation with a constant-mean prediction. Higher is generally better, but a high R² alone does not mean a model beats persistence.

The metrics report also compares model and persistence error on hours when AQI changes and when it stays the same. Persistence can be hard to beat because AQI often changes slowly. A model should only be described as better than persistence if its test results support that conclusion.

## Limitations

- AQI is provisional and has not been independently audited against all CPCB rules.
- The input file was formed from the project preprocessing stages, so any upstream label-generation choices remain part of the data's limitations.
- Hourly gaps are preserved as gaps; this reduces available examples and can leave some lag or rolling features missing. The training-only median imputer handles remaining missing predictors.
- A chronological holdout tests later observations from this station and period. It does not establish performance at another monitoring station or in future conditions unlike this dataset.
- This is a one-hour-ahead prediction experiment, not a health advisory system.

## Run the project

Open PowerShell in the project root and activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
python train_model.py
```

Or run directly without activation:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The project uses pandas, NumPy, scikit-learn, matplotlib, and joblib. If an import is missing, install the project's dependencies into `.venv` with:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib joblib
```

## Generated outputs

Running `train_model.py` writes the following files under `models/`:

- `hourly_aqi_model.joblib` — selected pipeline and its feature list
- `test_predictions.csv` — target timestamp, forecast origin, actual AQI, selected model prediction, and persistence prediction
- `actual_vs_predicted.png` — actual AQI, model prediction, and baseline
- `prediction_error.png` — actual minus model prediction
- `metrics_report.txt` and `metrics_report.json` — split details, model settings, test metrics, and AQI-change diagnostics

The printed metrics and saved report are generated from the current data when the script runs; no results are hard-coded in this README.
