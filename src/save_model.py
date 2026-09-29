import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# ============================================================
# 1. Load dataset
# ============================================================

DATA_FILE = "data/processed/transformer_faults.csv"

df = pd.read_csv(DATA_FILE)

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
# 2. Encode target
# ============================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ============================================================
# 3. Build final model
# ============================================================

model = Pipeline([
    (
        "log",
        FunctionTransformer(
            np.log1p,
            validate=False
        )
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 4. Train on complete dataset
# ============================================================

print("Training final model...")

model.fit(X, y_encoded)


# ============================================================
# 5. Save model and label encoder
# ============================================================

joblib.dump(
    model,
    "models/transformer_fault_model.joblib"
)

joblib.dump(
    encoder,
    "models/label_encoder.joblib"
)


print("\nModel saved successfully.")

print("\nModel classes:")

for number, name in enumerate(encoder.classes_):
    print(f"{number}: {name}")
