import sys
from pathlib import Path

import joblib
import pandas as pd

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILES = {
    "random_forest": "random_forest_model.joblib",
    "svm": "svm_model.joblib",
    "knn": "knn_model.joblib",
}
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine (SVM)",
    "knn": "K-Nearest Neighbors (KNN)",
}

encoder = joblib.load(BASE_DIR / "models" / "label_encoder.joblib")
models = {}
for key, filename in MODEL_FILES.items():
    path = BASE_DIR / "models" / filename
    if path.exists():
        models[key] = joblib.load(path)

# Backward compatibility with the original single-model release.
if "random_forest" not in models:
    old_path = BASE_DIR / "models" / "transformer_fault_model.joblib"
    if old_path.exists():
        models["random_forest"] = joblib.load(old_path)


def _predict(model_key, data):
    if model_key not in models:
        raise FileNotFoundError(
            f"{MODEL_NAMES[model_key]} is not trained yet. "
            "Run: python src/train_all_models.py"
        )
    model = models[model_key]
    encoded_prediction = model.predict(data)[0]
    fault = encoder.inverse_transform([encoded_prediction])[0]
    probabilities = model.predict_proba(data)[0]

    classifier = model.named_steps.get("classifier")
    if classifier is None:
        classifier = model.steps[-1][1]

    probability_dict = {
        encoder.inverse_transform([int(class_id)])[0]: float(probability)
        for class_id, probability in zip(classifier.classes_, probabilities)
    }
    return fault, probability_dict


def _make_input(H2, CH4, C2H6, C2H4, C2H2):
    return pd.DataFrame([{
        "H2": H2, "CH4": CH4, "C2H6": C2H6,
        "C2H4": C2H4, "C2H2": C2H2,
    }])


def predict_fault(H2, CH4, C2H6, C2H4, C2H2, model_name="random_forest"):
    data = _make_input(H2, CH4, C2H6, C2H4, C2H2)
    return _predict(model_name, data)


def compare_models(H2, CH4, C2H6, C2H4, C2H2):
    data = _make_input(H2, CH4, C2H6, C2H4, C2H2)
    return {
        key: _predict(key, data)
        for key in MODEL_FILES
    }


if __name__ == "__main__":
    fault, probabilities = predict_fault(500, 300, 100, 400, 20)
    print("Predicted fault:", fault)
    for name, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {probability:.2%}")
