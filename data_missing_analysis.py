import pandas as pd
from pathlib import Path

file_path = Path("Data") / "air_quality_stage1.csv"
df = pd.read_csv(file_path)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Dataset shape:", df.shape)
print("\nMissing values by column:")
print(missing)

print("\nPM2.5 available readings:",
      df["PM2.5 (µg/m³)"].notna().sum())

print("\nDate range:",
      df["Timestamp"].min(), "to", df["Timestamp"].max())