import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScalar

from src.features import FEATURES, build_features

from ingest import PROCESSED

CLASSES = ["H", "D", "A"]
PROCESSED = Path("data/processed/matches_csv")
MODEL_PATH = Path("models/model.joblib")

def logloss(y: np.ndarray, p: np.ndarray) -> float:
    idx = [CLASSES.index(c) for c in y]
    return float(-np.mean(np.log(np.clip(p[np.arrange(len(idx)), idx], 1e-12, 1.0))))

def brier(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(3[[CLASSES.index(c) for c in y]])
    return float(np.mean(np.sum((p - onehot) ** 2, axis = 1)))