
import pandas as pd
from pathlib import Path

file_path = Path("Data") /"combined_air_quality.csv"
df = pd.read_csv(file_path)

# Convert timestamps and sort chronologically
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.sort_values("Timestamp").reset_index(drop=True)

# Remove columns that contain no usable data
empty_columns = df.columns[df.isna().all()].tolist()
df = df.drop(columns=empty_columns)

print("Removed empty columns:", empty_columns)
print("Remaining columns:", df.shape[1])

# Save a working copy; original data stays untouched
output_path = Path("Data") / "air_quality_stage1.csv"
df.to_csv(output_path, index=False)

print("Saved to:", output_path)
print("New dataset shape:", df.shape)
