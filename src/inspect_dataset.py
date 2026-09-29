import pandas as pd

INPUT_FILE = "data/raw/dataset_(589).xlsx"
OUTPUT_FILE = "data/processed/dga_clean.csv"

# Load dataset
df = pd.read_excel(INPUT_FILE)

# Rename the target column
df = df.rename(columns={
    "故障类型": "Fault"
})

# Translate fault labels
fault_mapping = {
    "局部放电": "Partial Discharge",
    "低能放电": "Low Energy Discharge",
    "高能放电": "High Energy Discharge",
    "低温过热": "Low Temperature Thermal",
    "中温过热": "Medium Temperature Thermal",
    "高温过热": "High Temperature Thermal"
}

df["Fault"] = df["Fault"].map(fault_mapping)

# Remove exact duplicate rows
before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"Rows before removing duplicates: {before}")
print(f"Rows after removing duplicates:  {after}")
print(f"Duplicates removed:              {before - after}")

# Save processed dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)

print("\nFault distribution:")
print(df["Fault"].value_counts())