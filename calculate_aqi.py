
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Load hourly data
# --------------------------------------------------
input_path = Path("Data") / "air_quality_stage1.csv"
output_path = Path("Data") / "air_quality_with_aqi.csv"

df = pd.read_csv(input_path)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"])
df = df.sort_values("Timestamp")
df = df.drop_duplicates(subset="Timestamp").set_index("Timestamp")

# --------------------------------------------------
# 2. CPCB breakpoint bands from your Excel workbook
# Format: (concentration low, high, AQI low, high)
# --------------------------------------------------
breakpoints = {
    "PM2.5 (µg/m³)": [
        (0, 30, 0, 50), (31, 60, 51, 100),
        (61, 90, 101, 200), (91, 120, 201, 300),
        (121, 250, 301, 400), (251, 380, 401, 500)
    ],
    "PM10 (µg/m³)": [
        (0, 50, 0, 50), (51, 100, 51, 100),
        (101, 250, 101, 200), (251, 350, 201, 300),
        (351, 430, 301, 400), (431, 510, 401, 500)
    ],
    "NO2 (µg/m³)": [
        (0, 40, 0, 50), (41, 80, 51, 100),
        (81, 180, 101, 200), (181, 280, 201, 300),
        (281, 400, 301, 400), (401, 520, 401, 500)
    ],
    "SO2 (µg/m³)": [
        (0, 40, 0, 50), (41, 80, 51, 100),
        (81, 380, 101, 200), (381, 800, 201, 300),
        (801, 1600, 301, 400), (1601, 2620, 401, 500)
    ],
    "CO (mg/m³)": [
        (0, 1.0, 0, 50), (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400), (34.1, 50.0, 401, 500)
    ],
    "Ozone (µg/m³)": [
        (0, 50, 0, 50), (51, 100, 51, 100),
        (101, 168, 101, 200), (169, 208, 201, 300),
        (209, 748, 301, 400), (749, 1000, 401, 500)
    ],
    "NH3 (µg/m³)": [
        (0, 200, 0, 50), (201, 400, 51, 100),
        (401, 800, 101, 200), (801, 1200, 201, 300),
        (1201, 1800, 301, 400), (1801, 2400, 401, 500)
    ]
}

# --------------------------------------------------
# 3. Calculate pollutant sub-index
# --------------------------------------------------
def calculate_subindex(value, bands):
    if pd.isna(value) or value < 0:
        return np.nan

    # Truncate concentration before selecting its band.
    # CO is expressed to one decimal place in the workbook.
    if bands is breakpoints["CO (mg/m³)"]:
        value = np.floor(value * 10) / 10
    else:
        value = np.floor(value)

    for c_low, c_high, i_low, i_high in bands:
        if c_low <= value <= c_high:
            if c_high == c_low:
                return float(i_high)

            index = (
                (i_high - i_low) / (c_high - c_low)
            ) * (value - c_low) + i_low

            return float(np.floor(index + 0.5))

    # Outside the workbook's defined concentration bands:
    # leave unclassified rather than inventing a value.
    return np.nan

# --------------------------------------------------
# 4. Calculate rolling pollutant averages
# --------------------------------------------------
pollutant_averages = {}

for pollutant in breakpoints:
    df[pollutant] = pd.to_numeric(df[pollutant], errors="coerce")

    if pollutant in ["CO (mg/m³)", "Ozone (µg/m³)"]:
        window_hours = 8
        minimum_readings = 6
    else:
        window_hours = 24
        minimum_readings = 16

    # Require actual readings within the time window.
    average = df[pollutant].rolling(
        f"{window_hours}h",
        min_periods=minimum_readings
    ).mean()

    pollutant_averages[pollutant] = average

# --------------------------------------------------
# 5. Calculate each pollutant's sub-index
# --------------------------------------------------
subindices = pd.DataFrame(index=df.index)

for pollutant, bands in breakpoints.items():
    subindices[pollutant] = pollutant_averages[pollutant].apply(
        lambda value: calculate_subindex(value, bands)
    )

# CPCB PDF exception:
# If 8-hour ozone average exceeds 208 µg/m³,
# use the 1-hour average for the ozone sub-index.
ozone_1h = df["Ozone (µg/m³)"].rolling(
    "1h", min_periods=1
).mean()

ozone_exception = pollutant_averages["Ozone (µg/m³)"] > 208

subindices.loc[ozone_exception, "Ozone (µg/m³)"] = (
    ozone_1h.loc[ozone_exception].apply(
        lambda value: calculate_subindex(
            value, breakpoints["Ozone (µg/m³)"]
        )
    )
)

# --------------------------------------------------
# 6. Calculate overall AQI
# --------------------------------------------------
valid_count = subindices.notna().sum(axis=1)

has_pm = subindices[
    ["PM2.5 (µg/m³)", "PM10 (µg/m³)"]
].notna().any(axis=1)

valid_aqi = (valid_count >= 3) & has_pm

df["AQI"] = subindices.max(axis=1).where(valid_aqi).round().astype("Int64")

def get_category(aqi):
    if pd.isna(aqi):
        return pd.NA
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    if aqi <= 500:
        return "Severe"
    return pd.NA

df["AQI_Category"] = df["AQI"].apply(get_category)

# --------------------------------------------------
# 7. Save results without changing original data
# --------------------------------------------------
df.reset_index().to_csv(output_path, index=False)

print("Saved:", output_path)
print("Dataset shape:", df.shape)
print("\nAQI category counts (including missing):")
print(df["AQI_Category"].value_counts(dropna=False))
print("\nValid AQI rows:", df["AQI"].notna().sum())
print("Insufficient/invalid AQI rows:", df["AQI"].isna().sum())
