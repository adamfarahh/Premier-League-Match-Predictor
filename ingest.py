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

KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]

def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"

def download_season(start_year: int) -> pd.DataFrame:
    response = requests.get(URL.format(season=season_code(start_year)), timeout = 30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.content.decode("latin-1")))
    df = df[[c for c in KEEP if c in df.columns]].copy()
    