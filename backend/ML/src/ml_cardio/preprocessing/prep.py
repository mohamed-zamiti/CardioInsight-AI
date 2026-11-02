import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TARGET = "cardio"
NUMERICAL = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
BINARY = ["is_male", "smoke", "alco", "active"]
ORDINAL = ["cholesterol", "gluc"]
FEATURES = NUMERICAL + BINARY + ORDINAL

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns="id", errors="ignore").drop_duplicates().copy()
    df["age_years"] = df["age"] / 365.25
    df["is_male"] = (df["gender"] == 2).astype(int)
    df = df.drop(columns=["age", "gender"])
    valid = (
        df["ap_hi"].between(70, 250)
        & df["ap_lo"].between(40, 150)
        & (df["ap_lo"] < df["ap_hi"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
    )
    df = df[valid].copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df = df[df["bmi"].between(12, 60)]
    return df[FEATURES + [TARGET]].reset_index(drop=True)
def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Fit on the training set only (call inside a Pipeline)."""
    return ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", NUMERICAL)],
        remainder="passthrough",
    )