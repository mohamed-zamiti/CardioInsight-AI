import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from ml_cardio.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from ml_cardio.preprocessing.prep import FEATURES
from ml_cardio.preprocessing.split import load_splits
from ml_cardio.training.models import get_models

MODEL_NAME = "xgboost"
MODEL_VERSION = "v0.1.0"


def train_and_save(model_name: str = MODEL_NAME, version: str = MODEL_VERSION):
    X_train, _, y_train, _ = load_splits()          # test set stays untouched

    pipe = get_models()[model_name]
    pipe.fit(X_train, y_train)                      # scaler (if any) + model, train only

    cv = pd.read_csv(REPORTS_DIR / "cv_comparison.csv", index_col="model").loc[model_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "cardio_model.joblib")

    metadata = {
        "model_version": version,
        "algorithm": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(X_train)),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "target": "cardio (1 = cardiovascular disease present)",
        "cv_val_roc_auc": float(cv["val_roc_auc"]),
        "cv_val_roc_auc_std": float(cv["val_roc_auc_std"]),
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {model_name} {version} -> {MODELS_DIR}")
    return pipe