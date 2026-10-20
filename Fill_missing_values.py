import pandas as pd
from pathlib import Path

input_file = Path("Data/air_quality_ml_ready.csv")
output_file = Path("Data/Final_Air_quality.csv")

df = pd.read_csv(input_file)
original_columns = df.columns.tolist()

# Fill missing numerical values without creating extra columns
numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols] = df[numeric_cols].interpolate(
    method="linear",
    limit=3,
    limit_area="inside"
)

for col in numeric_cols:
    if col != "AQI":
        df[col] = df[col].fillna(df[col].median())

# Keep exactly the original 27 columns
df = df[original_columns]
df.to_csv(output_file, index=False)

print("Final shape:", df.shape)
print("Column count:", len(df.columns))
print("Columns:", df.columns.tolist())
print("Missing values:", int(df.isna().sum().sum()))