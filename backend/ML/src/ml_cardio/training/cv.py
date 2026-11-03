import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from ml_cardio.config import RANDOM_STATE

N_SPLITS = 5
SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def compare_models(models: dict, X, y) -> pd.DataFrame:
    """Cross-validate every model on the training set and return a comparison table."""
    rows = []
    for name, pipe in models.items():
        res = cross_validate(
            pipe, X, y, cv=get_cv(), scoring=SCORING,
            return_train_score=True, n_jobs=1,   # models parallelize internally
        )
        rows.append({
            "model": name,
            "val_roc_auc": res["test_roc_auc"].mean(),
            "val_roc_auc_std": res["test_roc_auc"].std(),
            "train_roc_auc": res["train_roc_auc"].mean(),
            "accuracy": res["test_accuracy"].mean(),
            "precision": res["test_precision"].mean(),
            "recall": res["test_recall"].mean(),
            "f1": res["test_f1"].mean(),
            "fit_time_s": res["fit_time"].mean(),
        })
        print(f"done: {name}")
    out = pd.DataFrame(rows).set_index("model")
    out["overfit_gap"] = out["train_roc_auc"] - out["val_roc_auc"]
    return out.sort_values("val_roc_auc", ascending=False).round(4)