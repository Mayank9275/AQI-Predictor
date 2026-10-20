
import pandas as pd
from pathlib import Path

file_path = Path("Data") / "air_quality_stage1.csv"
df = pd.read_csv(file_path)

aqi_pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]

print("AQI pollutant data availability:\n")

for col in aqi_pollutants:
    valid = df[col].notna().sum()
    missing = df[col].isna().sum()
    print(
        f"{col}: {valid:,} valid, "
        f"{missing:,} missing "
        f"({missing / len(df) * 100:.2f}%)"
    )
