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