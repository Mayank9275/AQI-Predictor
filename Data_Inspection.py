
import pandas as pd
from pathlib import Path

file_path = Path("Data") / "combined_air_quality.csv"
df = pd.read_csv(file_path)

# 1. Dataset size
print("Dataset shape:", df.shape)

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

# 3. Missing percentage
print("\nMissing percentage:")
print((df.isnull().mean() * 100).round(2).sort_values(ascending=False))

# 4. Duplicate timestamps
print("\nDuplicate timestamps:", df["Timestamp"].duplicated().sum())

# 5. Date range
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
print("\nDate range:", df["Timestamp"].min(), "to", df["Timestamp"].max())

# 6. Available data
print("\nNon-missing values:")
print(df.notna().sum().sort_values(ascending=False))
