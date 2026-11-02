import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardio_train.csv"
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")