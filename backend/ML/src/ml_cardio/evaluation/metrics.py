import json

import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)

from ml_cardio.config import MODELS_DIR, REPORTS_DIR
from ml_cardio.preprocessing.split import load_splits


def evaluate_on_test(threshold: float = 0.5) -> dict:
    _, X_test, _, y_test = load_splits()
    pipe = joblib.load(MODELS_DIR / "cardio_model.joblib")

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall_sensitivity": recall_score(y_test, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, pred),
        "brier_score": brier_score_loss(y_test, proba),   # lower = better calibrated
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    metrics = {k: round(float(v), 4) if isinstance(v, float) else v for k, v in metrics.items()}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No disease", "Disease"], cmap="Blues",
        colorbar=False, ax=axes[0])
    axes[0].set_title(f"Confusion matrix (threshold {threshold})")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_title("ROC curve")
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=axes[2])
    axes[2].set_title("Calibration")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "test_evaluation.png", dpi=130)
    plt.close()

    return metrics