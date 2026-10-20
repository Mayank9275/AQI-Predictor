
import pandas as pd
from pathlib import Path

# Load dataset containing provisional AQI labels
input_path = Path("Data") / "air_quality_with_aqi.csv"
output_path = Path("Data") / "air_quality_ml_ready.csv"

df = pd.read_csv(input_path)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Remove rows where AQI could not be calculated
df = df.dropna(subset=["Timestamp", "AQI", "AQI_Category"])

# Create time-based features for ML
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["Month"] = df["Timestamp"].dt.month

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Save a separate dataset
df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Dataset shape:", df.shape)
print("\nAQI category counts:")
print(df["AQI_Category"].value_counts())
print("\nMissing values in target AQI:", df["AQI"].isna().sum())
