import json

from ml_cardio.config import MODELS_DIR
from ml_cardio.evaluation.metrics import evaluate_on_test

if __name__ == "__main__":
    m = evaluate_on_test()
    cv_auc = json.load(open(MODELS_DIR / "model_metadata.json"))["cv_val_roc_auc"]
    for k, v in m.items():
        print(f"{k:20s} {v}")
    print(f"\nCV AUC {cv_auc} vs TEST AUC {m['roc_auc']}")