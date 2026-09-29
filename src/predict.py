import sys
from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# Locate bundled/project files
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Load model
# ============================================================

model = joblib.load(
    BASE_DIR / "models" / "transformer_fault_model.joblib"
)

encoder = joblib.load(
    BASE_DIR / "models" / "label_encoder.joblib"
)





# ============================================================
# Prediction function
# ============================================================

def predict_fault(H2, CH4, C2H6, C2H4, C2H2):

    data = pd.DataFrame([{
        "H2": H2,
        "CH4": CH4,
        "C2H6": C2H6,
        "C2H4": C2H4,
        "C2H2": C2H2
    }])

    # Prediction
    prediction = model.predict(data)[0]

    # Convert encoded label to original label
    fault = encoder.inverse_transform(
        [prediction]
    )[0]

    # Probabilities
    probabilities = model.predict_proba(data)[0]

    probability_dict = {}

    for class_number, probability in zip(
        model.named_steps["classifier"].classes_,
        probabilities
    ):
        class_name = encoder.inverse_transform(
            [class_number]
        )[0]

        probability_dict[class_name] = probability

    return fault, probability_dict


# ============================================================
# Test prediction
# ============================================================

if __name__ == "__main__":

    fault, probabilities = predict_fault(
        H2=500,
        CH4=300,
        C2H6=100,
        C2H4=400,
        C2H2=20
    )

    print("\n" + "=" * 50)
    print("TRANSFORMER FAULT DIAGNOSIS")
    print("=" * 50)

    print("\nPredicted fault:")
    print(fault)

    print("\nProbabilities:")

    for name, probability in sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{name:15s}: "
            f"{probability * 100:.2f}%"
        )
