
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml_cardio.config import RANDOM_STATE
from ml_cardio.preprocessing.prep import build_preprocessor


def get_models() -> dict[str, Pipeline]:
    """Baseline models. Only logistic regression needs scaling; tree models don't."""
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "decision_tree": (
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
        
        "xgboost": (
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE), False),
        "lightgbm": (
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           n_jobs=-1, random_state=RANDOM_STATE, verbose=-1), False),
        
    }
    return {
        name: Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", model)])
        for name, (model, scale) in candidates.items()
    }