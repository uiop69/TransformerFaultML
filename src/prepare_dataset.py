import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = Path("data/raw/dataset_(589).xlsx")
OUTPUT_FILE = Path("data/processed/transformer_faults.csv")

# Create output directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_excel(INPUT_FILE)

print("Original shape:", df.shape)

# Rename target column
df = df.rename(columns={"故障类型": "fault_type"})

# Check duplicates
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Display class distribution
print("\nClass distribution:")
print(df["fault_type"].value_counts())

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned shape:", df.shape)
print("Saved to:", OUTPUT_FILE)
