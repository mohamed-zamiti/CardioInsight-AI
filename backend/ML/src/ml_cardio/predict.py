import json
from functools import lru_cache

import joblib
import pandas as pd
import shap

from ml_cardio.config import MODELS_DIR
from ml_cardio.preprocessing.prep import FEATURES

THRESHOLD = 0.5   # decision cutoff; to be chosen with clinical input later

# What the doctor enters (BMI is computed here, not entered)
RANGES = {                      # physiological limits, same idea as clean_data
    "age_years": (18, 120),
    "height": (120, 220),       # cm
    "weight": (30, 200),        # kg
    "ap_hi": (70, 250),         # systolic, mmHg
    "ap_lo": (40, 150),         # diastolic, mmHg
}
CHOICES = {
    "is_male": {0, 1},
    "smoke": {0, 1},
    "alco": {0, 1},
    "active": {0, 1},
    "cholesterol": {1, 2, 3},   # 1 normal, 2 above normal, 3 well above normal
    "gluc": {1, 2, 3},
}
REQUIRED = list(RANGES) + list(CHOICES)


class PatientDataError(ValueError):
    """Raised when patient input is missing or outside valid ranges."""
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@lru_cache(maxsize=1)
def _load():
    """Load model, metadata and SHAP explainer once, then reuse."""
    model_path = MODELS_DIR / "cardio_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run scripts/run_train.py first.")
    pipe = joblib.load(model_path)
    with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    return pipe, prep, shap.TreeExplainer(clf), names, meta


def validate_patient(patient: dict) -> None:
    errors = []
    for key in REQUIRED:
        if key not in patient or patient[key] is None:
            errors.append(f"missing field: {key}")
    if errors:
        raise PatientDataError(errors)

    for key, (lo, hi) in RANGES.items():
        if not lo <= patient[key] <= hi:
            errors.append(f"{key}={patient[key]} outside valid range {lo}-{hi}")
    for key, allowed in CHOICES.items():
        if patient[key] not in allowed:
            errors.append(f"{key}={patient[key]} must be one of {sorted(allowed)}")
    if patient["ap_lo"] >= patient["ap_hi"]:
        errors.append("diastolic (ap_lo) must be lower than systolic (ap_hi)")
    if errors:
        raise PatientDataError(errors)


def predict_patient(patient: dict, top_k: int = 5) -> dict:
    """Predict cardiovascular disease for one patient.

    patient keys: age_years, is_male, height, weight, ap_hi, ap_lo,
                  cholesterol, gluc, smoke, alco, active
    """
    validate_patient(patient)
    pipe, prep, explainer, names, meta = _load()

    row = {k: patient[k] for k in REQUIRED}
    row["bmi"] = row["weight"] / (row["height"] / 100) ** 2
    X = pd.DataFrame([row])[FEATURES]            # same columns and order as training

    probability = float(pipe.predict_proba(X)[0, 1])
    shap_values = explainer.shap_values(prep.transform(X))[0]   # log-odds contributions

    contributions = sorted(
        (
            {
                "feature": name,
                "value": round(float(X.iloc[0][name]), 2),
                "shap_value": round(float(sv), 4),
                "effect": "increases risk" if sv > 0 else "decreases risk",
            }
            for name, sv in zip(names, shap_values)
        ),
        key=lambda c: abs(c["shap_value"]),
        reverse=True,
    )[:top_k]

    return {
        "predicted_class": int(probability >= THRESHOLD),
        "probability": round(probability, 4),
        "threshold": THRESHOLD,
        "model_version": meta["model_version"],
        "explanation": contributions,
        "disclaimer": "Decision-support estimate from a prototype model, not a diagnosis.",
    }