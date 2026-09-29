import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


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
# 2. Encode target
# ============================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# ============================================================
# 3. Pipeline
# ============================================================

pipeline = Pipeline([
    (
        "log",
        FunctionTransformer(
            np.log1p,
            validate=False
        )
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ============================================================
# 4. Parameter grid
# ============================================================

param_grid = {

    "model__n_estimators": [
        200,
        300,
        500
    ],

    "model__max_depth": [
        None,
        10,
        20,
        30
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


# ============================================================
# 5. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. Grid search
# ============================================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


print("Starting Random Forest hyperparameter tuning...")
print("This may take some time...\n")


grid.fit(X, y)


# ============================================================
# 7. Results
# ============================================================

print("\n" + "=" * 60)
print("BEST RANDOM FOREST")
print("=" * 60)

print("\nBest Macro F1:")
print(f"{grid.best_score_:.4f}")

print("\nBest Parameters:")

for parameter, value in grid.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 8. Top 10 configurations
# ============================================================

results = pd.DataFrame(grid.cv_results_)

results = results.sort_values(
    "rank_test_score"
)

print("\n" + "=" * 60)
print("TOP 10 CONFIGURATIONS")
print("=" * 60)

columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "param_model__n_estimators",
    "param_model__max_depth",
    "param_model__min_samples_split",
    "param_model__min_samples_leaf",
    "param_model__max_features"
]

print(
    results[columns].head(10).to_string(index=False)
)