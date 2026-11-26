"""Train and save the three models used by the desktop application.

Run from the repository root:
    python src/train_all_models.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "transformer_faults.csv"
MODEL_DIR = ROOT / "models"
FEATURES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

df = pd.read_csv(DATA_FILE)
X = df[FEATURES].astype(float)
if (X < 0).any().any():
    raise ValueError("Gas concentrations must be non-negative.")
y = LabelEncoder().fit_transform(df["fault_type"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

log_transform = FunctionTransformer(np.log1p, validate=False)
model_definitions = {
    "random_forest_model.joblib": Pipeline([
        ("log", log_transform),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight="balanced", random_state=42
        )),
    ]),
    "svm_model.joblib": Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=42
        )),
    ]),
    "knn_model.joblib": Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        )),
    ]),
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
encoder = LabelEncoder()
encoder.fit(df["fault_type"])
joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

for filename, model in model_definitions.items():
    print(f"\nEvaluating {filename} on the same stratified 20% hold-out set ...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(classification_report(
        y_test, predictions, target_names=encoder.classes_, zero_division=0
    ))

    # Refit on the complete dataset for deployment after evaluation.
    model.fit(X, y)
    joblib.dump(model, MODEL_DIR / filename)
    print(f"Saved full-data model: {MODEL_DIR / filename}")

# Keep the original filename so older scripts/build configurations continue to work.
joblib.dump(
    model_definitions["random_forest_model.joblib"],
    MODEL_DIR / "transformer_fault_model.joblib"
)
print("All three models trained and saved.")
