from pathlib import Path
import joblib
import numpy as np

from src.features import FEATURES

from train import MODEL_PATH

MODEL_PATH = Path("models/model.joblib")
CLASSES = ["H", "D", "A"]
_model = None

def _load():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model