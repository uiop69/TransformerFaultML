import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score


# Load dataset
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


# Log transform
X = np.log1p(X)


# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


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


for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": make_scorer(
                f1_score,
                average="macro"
            )
        }
    )

    accuracy = scores["test_accuracy"]
    f1 = scores["test_f1_macro"]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{accuracy.mean():.4f} ± {accuracy.std():.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1.mean():.4f} ± {f1.std():.4f}"
    )
