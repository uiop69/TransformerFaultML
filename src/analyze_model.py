import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("data/processed/transformer_faults.csv")

features = [
    "H2",
    "CH4",
    "C2H6",
    "C2H4",
    "C2H2"
]

X = df[features]
y = df["fault_type"]


# ============================================================
# 2. Log transformation
# ============================================================

X = np.log1p(X)


# ============================================================
# 3. Encode target
# ============================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ============================================================
# 4. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 5. Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# 6. Predictions
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 7. Classification report
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)


# ============================================================
# 8. Confusion matrix
# ============================================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Transformer Fault Diagnosis - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "data/processed/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. Feature importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

for feature, value in importance.items():
    print(f"{feature:6s}: {value:.4f}")


# ============================================================
# 10. Feature importance plot
# ============================================================

plt.figure(figsize=(8, 5))

importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Gas")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png",
    dpi=300
)

plt.show()
