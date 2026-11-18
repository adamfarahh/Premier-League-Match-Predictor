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
    df["Season"] = start_year
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset = ["Date", "HomeTeam", "AwayTeam", "FTR"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, format = "mixed")
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop = True)
    for col in ("B365H", "B365D", "B365A"):
        if col not in df.columns:
            df[col] = np.nan
    return df

def devig(df: pd.DataFrame) -> pd.DataFrame: """Convert decimal odds to probabilities that sum to 1 (removes bookmaker margin)"""
    raw = 1.0 / df[["B365H", "B365D", "B365A"]].to_numpy(dtype=float)
    probs = raw / raw.sum(axis =1, keepdims = True)
    out = df.copy()
    out[["pH_book", "pD_book", "pA_book"]] = probs
    return out

def main(start: int, end: int) -> None:
    RAW.mkdir(parents = True, exist_ok = True)
    frames = []
    for y in range(start, end + 1):
        print(f"downloading {y}/{y + 1}")
        frames.append(download_season(y))
    df = devig(clean(pd.concat(frames, ignore_index = True)))
    PROCESSED.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(PROCESSED)
    print(f"saved {len(df)} matches to {PROCESSED}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default = 2015)
    ap.add_argument("--end", type=int, default = 2025)
    args = parser.parse_args()
    main(args.start, args.end)