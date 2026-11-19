from pathlib import Path
import joblib
import numpy as np

from src.features import FEATURES

from train import MODEL_PATH

MODEL_PATH = Path("models/model.joblib")
CLASSES = ["H", "D", "A"]
_model = None

