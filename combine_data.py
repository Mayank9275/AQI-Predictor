
import pandas as pd
from pathlib import Path

# Folder containing the raw CSV files
data_folder = Path("Data") / "Raw Data"

# Read each year's dataset
df_2024 = pd.read_csv(data_folder / "Raw_Data_2024_site_5393_chandni_chowk_delhi_iitm_1Hr.csv")
df_2025 = pd.read_csv(data_folder / "Raw_Data_2025_site_5393_chandni_chowk_delhi_iitm_1Hr.csv")
df_2026 = pd.read_csv(data_folder / "Raw_Data_2026_site_5393_chandni_chowk_delhi_iitm_1Hr.csv")

# Combine all three datasets
df = pd.concat([df_2024, df_2025, df_2026], ignore_index=True)

# Sort by timestamp
df = df.sort_values("Timestamp").reset_index(drop=True)

# Save the combined dataset
output_path = data_folder / "combined_air_quality.csv"
df.to_csv(output_path, index=False)

# Display summary
print("2024 records:", len(df_2024))
print("2025 records:", len(df_2025))
print("2026 records:", len(df_2026))
print("Total records:", len(df))
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nSaved to:", output_path)
