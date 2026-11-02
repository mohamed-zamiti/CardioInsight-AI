import pandas as pd
from sklearn.model_selection import train_test_split

from ml_cardio.config import DATA_PREPROCESSED, RANDOM_STATE, TEST_SIZE, TARGET
from ml_cardio.preprocessing.prep import FEATURES

def split_data(df: pd.DataFrame):
    """Split a cleaned dataframe into train and test sets."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,                 # keep the same class ratio in both sets
        random_state=RANDOM_STATE,  # reproducible split
    )

def save_splits(df_clean, X_train, X_test, y_train, y_test):
    DATA_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(DATA_PREPROCESSED / "cardio_clean.csv", index=False)
    X_train.to_csv(DATA_PREPROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PREPROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PREPROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PREPROCESSED / "y_test.csv", index=False)

def load_splits():
    """Read the saved splits back, so training doesn't redo cleaning."""
    p = DATA_PREPROCESSED
    return (pd.read_csv(p / "X_train.csv"), pd.read_csv(p / "X_test.csv"),
            pd.read_csv(p / "y_train.csv").squeeze(), pd.read_csv(p / "y_test.csv").squeeze())