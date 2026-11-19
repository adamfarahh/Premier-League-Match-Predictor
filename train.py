import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScalar, StandardScaler

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

def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000))

def synthetic(n_seasons: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(20)]
    strength = dict(zip(teams, rng.normal(0, 0.25, 20)))
    rows = []
    for s in range(n_seasons):
        day = pd.Timestamp(f"{2015 + s} - 08 - 10")
        for rnd in range(38):
            order = rng.permutation(teams)
            for i in range(0, 20, 2):
                h, a = order[i], order[i + 1]
                hg = rng.poisson(np.exp(0.25 + strength[h] - strength[a] * 0.5))
                ag = rng.poisson(np.exp(0.0 + strength[a] - strength[h] * 0.5))
                rows.append({"Date": day + pd.Timedelta(days = 7 * rnd), "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else "D" if hg == ag else "A", "Season": 2015 + s})
                df = pd.DataFrame(rows)
                df[["pH_book", "pD_book", "pA_book"]] = np.nan
                return df

def backtest(df: pd.DataFrame, min_train_seasons: int = 3) -> pd.DataFrame:
    seasons = sorted(df["Season"].unique())
    out = []
    for test_season in seasons[min_train_seasons:]:
        train = df[df["Season"] < test_season]
        test = df[df["Season"] == test_season]
        model = make_model().fit(train[FEATURES], train["FTR"])
        p = model.predict_proba(test[FEATURES])