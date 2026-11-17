import argparse
import io
from pathlib import Path

import pandas as pd
import numpy as np
import requests

RAW = Path("data/raw")
PROCESSED = Path("data/processed/matches.csv")
seasons = ["2122", "2223", "2324", "2425", "2526"]
frames = []

for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    df = pd.read_csv(url)
    df["Season"] = season
    frames.append(df)

matches = pd.concat(frames, ignore_index=True)

print(matches.shape)
print(matches.head())
