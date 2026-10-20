"""Train and evaluate one-hour-ahead AQI models.

Run this file from the project root with the project's virtual environment.
The target AQI is provisional and is used as supplied; this script does not
recalculate it.
"""

from pathlib import Path
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Save figures reliably without opening a window on Windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path("Data/air_quality_ml_ready.csv")
OUTPUT_DIR = Path("models")
RANDOM_STATE = 42
OUTPUT_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_hourly_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load rows with AQI labels and build features on an hourly time grid."""
    df = pd.read_csv(path)
    if "Timestamp" not in df or "AQI" not in df:
        raise ValueError("Input data must contain Timestamp and AQI columns.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "AQI"])
    # When duplicate timestamps exist, retain the last source row deterministically.
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    df = df.set_index("Timestamp").sort_index()
    if df.empty:
        raise ValueError("No rows with both a valid timestamp and AQI were found.")

    # Reindexing inserts missing hours as NaN. Shifts and rolling windows therefore
    # cannot accidentally reach across an outage.
    full_hours = pd.date_range(df.index.min(), df.index.max(), freq="h")
    hourly = df.reindex(full_hours)
    hourly.index.name = "Timestamp"

    feature_frame = pd.DataFrame(index=hourly.index)
    feature_frame["AQI"] = hourly["AQI"]

    for lag in (1, 3, 6, 24):
        feature_frame[f"AQI_lag_{lag}h"] = hourly["AQI"].shift(lag)

    # Past/current values only. Full windows are required, so a missing hour
    # invalidates that window instead of being silently skipped.
    for window in (3, 6, 24):
        rolling = hourly["AQI"].rolling(window=window, min_periods=window)
        feature_frame[f"AQI_rolling_{window}h_mean"] = rolling.mean()
        feature_frame[f"AQI_rolling_{window}h_std"] = rolling.std()

    feature_frame["Hour"] = hourly.index.hour
    feature_frame["DayOfWeek"] = hourly.index.dayofweek
    feature_frame["Month"] = hourly.index.month

    # Only pollutant and weather measurements at prediction time are included.
    # Explicitly omit identifiers, categories, and any possible target columns.
    excluded = {"Timestamp", "AQI", "AQI_Category", "Target_Next_Hour_AQI"}
    measurement_columns = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    for column in measurement_columns:
        feature_frame[column] = hourly[column]

    next_aqi = hourly["AQI"].shift(-1)
    next_timestamp = pd.Series(hourly.index, index=hourly.index).shift(-1)
    is_next_calendar_hour = (next_timestamp - pd.Series(hourly.index, index=hourly.index)) == pd.Timedelta(hours=1)
    target = next_aqi.where(is_next_calendar_hour)

    data = feature_frame.copy()
    data["Target_Next_Hour_AQI"] = target
    data["Target_Timestamp"] = pd.Series(hourly.index, index=hourly.index).shift(-1)
    data = data.dropna(subset=["AQI", "Target_Next_Hour_AQI"])
    features = feature_frame.columns.tolist()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    return data, features


def score(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": float(r2_score(y_true, predictions)),
    }


def accuracy_within_5(y_true: pd.Series, predictions: np.ndarray) -> float:
    """Return the percent of predictions within 5 AQI points of the actual value."""
    return float(np.mean(np.abs(np.asarray(y_true) - predictions) <= 5) * 100)


def make_pipeline(estimator) -> Pipeline:
    # This imputer is fit by Pipeline.fit on training rows only.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def main() -> None:
    data, features = load_hourly_data(DATA_PATH)
    X = data[features]
    y = data["Target_Next_Hour_AQI"]
    n = len(data)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_validation, y_validation = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]
    X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]

    # Hyperparameters are selected using validation MAE only.
    candidates = []
    for model_name, model_type in (("RandomForest", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)):
        for depth in (10, 20, None):
            for leaf_size in (1, 3):
                estimator = model_type(
                    n_estimators=100,
                    max_depth=depth,
                    min_samples_leaf=leaf_size,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                pipeline = make_pipeline(estimator)
                pipeline.fit(X_train, y_train)
                val_prediction = pipeline.predict(X_validation)
                val_mae = mean_absolute_error(y_validation, val_prediction)
                candidates.append((val_mae, model_name, depth, leaf_size, pipeline))

    candidates.sort(key=lambda item: item[0])
    best_val_mae, best_name, best_depth, best_leaf, best_model = candidates[0]

    # Evaluate each model family at its best validation-selected setting, and
    # evaluate the overall selected candidate exactly once on the test set.
    best_by_family = {}
    for candidate in candidates:
        if candidate[1] not in best_by_family:
            best_by_family[candidate[1]] = candidate

    baseline_predictions = X_test["AQI"].to_numpy()
    baseline_metrics = score(y_test, baseline_predictions)

    family_metrics = {}
    family_predictions = {}
    for family, (_, _, depth, leaf_size, model) in best_by_family.items():
        prediction = model.predict(X_test)
        family_predictions[family] = prediction
        family_metrics[family] = {
            "selected_on_validation": {"max_depth": depth, "min_samples_leaf": leaf_size},
            **score(y_test, prediction),
        }

    # Reuse the selected family's test prediction for the saved outputs.
    test_predictions = family_predictions[best_name]
    selected_metrics = {key: value for key, value in family_metrics[best_name].items() if key in {"MAE", "RMSE", "R2"}}

    train_predictions = best_model.predict(X_train)
    validation_predictions = best_model.predict(X_validation)
    train_mae = mean_absolute_error(y_train, train_predictions)
    validation_mae = mean_absolute_error(y_validation, validation_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_accuracy = accuracy_within_5(y_train, train_predictions)
    validation_accuracy = accuracy_within_5(y_validation, validation_predictions)
    test_accuracy = accuracy_within_5(y_test, test_predictions)

    # Explain where the old result can lose to persistence: compare both models
    # on stable hours and on hours where AQI actually changes.
    changed = y_test.to_numpy() != baseline_predictions
    diagnostics = {
        "test_hours_with_AQI_change": int(changed.sum()),
        "test_hours_without_AQI_change": int((~changed).sum()),
        "persistence_MAE_on_change_hours": float(mean_absolute_error(y_test.to_numpy()[changed], baseline_predictions[changed])) if changed.any() else None,
        "selected_model_MAE_on_change_hours": float(mean_absolute_error(y_test.to_numpy()[changed], test_predictions[changed])) if changed.any() else None,
        "persistence_MAE_on_stable_hours": float(mean_absolute_error(y_test.to_numpy()[~changed], baseline_predictions[~changed])) if (~changed).any() else None,
        "selected_model_MAE_on_stable_hours": float(mean_absolute_error(y_test.to_numpy()[~changed], test_predictions[~changed])) if (~changed).any() else None,
        "interpretation": "Persistence is difficult to beat when AQI changes slowly. Compare change-hour errors to see whether the selected model helps during actual AQI changes.",
    }

    print(f"Train:      {len(X_train)} rows ({len(X_train) / n * 100:.1f}%)")
    print(f"Validation: {len(X_validation)} rows ({len(X_validation) / n * 100:.1f}%)")
    print(f"Test:       {len(X_test)} rows ({len(X_test) / n * 100:.1f}%)")
    print("\n========== FINAL TEST RESULTS ==========")
    print(f"MAE:          {selected_metrics['MAE']:.3f}")
    print(f"RMSE:         {selected_metrics['RMSE']:.3f}")
    print(f"R²:           {selected_metrics['R2']:.3f}")
    print(f"Baseline MAE: {baseline_metrics['MAE']:.3f}")
    print("\nTraining Accuracy (within ±5 AQI points):    "
          f"{train_accuracy:.2f}%")
    print(f"Training Error (MAE):                       {train_mae:.3f}")
    print("Validation Accuracy (within ±5 AQI points): "
          f"{validation_accuracy:.2f}%")
    print(f"Validation Error (MAE):                     {validation_mae:.3f}")
    print("Testing Accuracy (within ±5 AQI points):     "
          f"{test_accuracy:.2f}%")
    print(f"Testing Error (MAE):                        {test_mae:.3f}")

    result = pd.DataFrame({
        "Timestamp": data["Target_Timestamp"].iloc[validation_end:].to_numpy(),
        "Forecast_Origin_Timestamp": data.index[validation_end:],
        "Actual_AQI": y_test.to_numpy(),
        "Predicted_AQI": test_predictions,
        "Baseline_AQI": baseline_predictions,
    })
    result.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    joblib.dump(
        {"model": best_model, "features": features, "target": "AQI at exactly t + 1 hour", "selected_model": best_name},
        OUTPUT_DIR / "hourly_aqi_model.joblib",
    )

    metrics_report = {
        "input_file": str(DATA_PATH),
        "target": "AQI at exactly one calendar hour after the forecast origin",
        "target_note": "Existing provisional AQI; CPCB sub-index calculation has not been independently validated here.",
        "valid_examples": n,
        "split_rows": {"train": len(X_train), "validation": len(X_validation), "test": len(X_test)},
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "selected_model": best_name,
        "selected_parameters": {"max_depth": best_depth, "min_samples_leaf": best_leaf},
        "selected_validation_MAE": float(best_val_mae),
        "test_metrics": {"Persistence": baseline_metrics, **family_metrics, "Selected": selected_metrics},
        "test_diagnostics": diagnostics,
        "feature_columns": features,
        "preprocessing": "Median imputer is part of each model pipeline and is fit only on the chronological training partition.",
        "selection_rule": "Lowest validation MAE among the tested Random Forest and Extra Trees settings.",
    }
    (OUTPUT_DIR / "metrics_report.json").write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "metrics_report.txt").write_text(
        "Hourly AQI prediction metrics\n"
        "=============================\n"
        f"Selected model: {best_name} (depth={best_depth}, min_samples_leaf={best_leaf})\n"
        f"Validation MAE: {best_val_mae:.3f}\n"
        f"Rows (train/validation/test): {len(X_train)}/{len(X_validation)}/{len(X_test)}\n\n"
        "Final test metrics (MAE and RMSE: lower is better; R² is not sufficient alone)\n"
        + "\n".join(
            f"{name}: MAE={values['MAE']:.3f}, RMSE={values['RMSE']:.3f}, R²={values['R2']:.3f}"
            for name, values in [("Persistence", baseline_metrics), *family_metrics.items(), ("Selected", selected_metrics)]
        )
        + "\n\nAQI-change diagnostic\n"
        + json.dumps(diagnostics, indent=2)
        + "\n",
        encoding="utf-8",
    )

    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        result["Timestamp"],
        result["Actual_AQI"],
        label="Actual AQI",
        color="blue",
        linewidth=1.5
    )

    ax.plot(
        result["Timestamp"],
        result["Predicted_AQI"],
        label="Predicted AQI",
        color="green",
        linewidth=1.2,
        alpha=0.9
    )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result["Timestamp"], result["Actual_AQI"] - result["Predicted_AQI"], linewidth=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title="Prediction error (actual − predicted)", xlabel="Target hour", ylabel="AQI error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
