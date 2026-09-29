import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "data/processed/dga_clean.csv"

df = pd.read_csv(FILE)

gas_columns = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

# -----------------------------
# 1. Fault distribution
# -----------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Fault",
    order=df["Fault"].value_counts().index
)

plt.title("Transformer Fault Class Distribution")
plt.xlabel("Fault Type")
plt.ylabel("Number of Samples")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("results/fault_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 2. Gas distributions
# -----------------------------

for gas in gas_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Fault",
        y=gas
    )

    plt.title(f"{gas} Concentration by Fault Type")
    plt.xlabel("Fault Type")
    plt.ylabel(f"{gas} Concentration")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        f"results/{gas}_distribution.png",
        dpi=300
    )

    plt.show()
