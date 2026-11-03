from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]   # .../backend/ML
DATA_RAW = ML_ROOT / "data" / "raw"
DATA_PREPROCESSED = ML_ROOT / "data" / "preprocessed"
MODELS_DIR = ML_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "cardio"
REPORTS_DIR = ML_ROOT / "reports"