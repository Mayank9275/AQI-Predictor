
import pandas as pd
from pathlib import Path

file_path = Path("Data") / "air_quality_ml_ready.csv"
df = pd.read_csv(file_path)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Dataset shape:", df.shape)
print("\nColumns with missing values:")

if missing.empty:
    print("No missing values found.")
else:
    report = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": (missing / len(df) * 100).round(2)
    })
    print(report)

print("\nTotal missing cells:", int(df.isna().sum().sum()))
