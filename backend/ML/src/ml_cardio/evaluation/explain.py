import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from ml_cardio.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from ml_cardio.preprocessing.split import load_splits


def _clean(names):
    # "num__bmi" -> "bmi", "remainder__smoke" -> "smoke"
    return [n.split("__", 1)[-1] for n in names]


def global_explanation(n_samples: int = 2000):
    model_path = MODELS_DIR / "cardio_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run scripts/run_train.py first.")

    pipe = joblib.load(model_path)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]

    X_train, _, _, _ = load_splits()
    X = X_train.sample(min(n_samples, len(X_train)), random_state=RANDOM_STATE)
    X_t = pd.DataFrame(prep.transform(X), columns=_clean(prep.get_feature_names_out()))

    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_t)                       # values are in log-odds units

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(sv, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close()

    shap.plots.bar(sv, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=130, bbox_inches="tight")
    plt.close()