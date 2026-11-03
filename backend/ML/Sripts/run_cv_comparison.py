from ml_cardio.config import REPORTS_DIR
from ml_cardio.preprocessing.split import load_splits
from ml_cardio.training.cv import compare_models
from ml_cardio.training.models import get_models


def main():
    X_train, _, y_train, _ = load_splits()   # test set deliberately ignored
    print(f"Train set: {X_train.shape}")

    results = compare_models(get_models(), X_train, y_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "cv_comparison.csv")

    print("\n", results.to_string())
    print(f"\nSaved to {REPORTS_DIR / 'cv_comparison.csv'}")


if __name__ == "__main__":
    main()