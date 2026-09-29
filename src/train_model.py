import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load dataset
# ============================================================

DATA_FILE = "data/processed/transformer_faults.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 2. Separate features and target
# ============================================================

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
# 3. Encode fault labels
# ============================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nFault classes:")
for number, name in enumerate(label_encoder.classes_):
    print(f"{number}: {name}")


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. Log transformation
# ============================================================

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)


# ============================================================
# 6. Models
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}


# ============================================================
# 7. Train and evaluate
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train_log, y_train)

    predictions = model.predict(X_test_log)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_
        )
    )
