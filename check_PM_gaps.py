
import pandas as pd
from pathlib import Path

file_path = Path("Data") / "air_quality_stage1.csv"
df = pd.read_csv(file_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

pollutants = ["PM2.5 (µg/m³)", "PM10 (µg/m³)"]

for pollutant in pollutants:
    missing = df[pollutant].isna()
    groups = missing.ne(missing.shift()).cumsum()

    gaps = (
        df.loc[missing]
        .groupby(groups[missing])
        .agg(
            start=("Timestamp", "min"),
            end=("Timestamp", "max"),
            missing_hours=("Timestamp", "size")
        )
        .sort_values("missing_hours", ascending=False)
    )

    print(f"\n{'=' * 45}")
    print(f"Pollutant: {pollutant}")
    print(f"Total missing readings: {missing.sum()}")
    print(f"Missing periods: {len(gaps)}")
    print("10 longest gaps:")
    print(gaps.head(10).to_string(index=False))
