import numpy as np
import pandas as pd
from pandas.plotting._matplotlib import hist

ELO_START = 1500.0
ELO_K = 20.0
HOME_ADV = 60.0
WINDOW = 5
FEATURES = ["elo_diff", "form_pts_diff", "gf_diff", "ga_diff", "rest_diff"]

def _expected(elo_h: float, elo_a: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_h + HOME_ADV - elo_a) / 400.0))

def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True).copy()
    elo: dict[str, float] = {}
    hist: dict[str, list] = {}
    rows = []

def form(team: str, date) -> tuple [float, float, float, float]:
    h = hist.get(team, [])[-WINDOW:]
    if not h:
        return 1.3, 1.3, 1.3, np.nan
    pts = np.mean([x[1] for x in h])
    gf = np.mean([x[2] for x in h])
    ga = np.mean([x[3] for x in h])
    rest = min((date - h[-1][0]).days, 14)
    return pts, gf, ga, rest

    for r in df.itertuples():
        h, a = r.HomeTeam, r.AwayTeam
        eh, ea = elo.get(h, ELO_START), elo.get(a, ELO_START)
        hp, hgf, hga, rest = form(h, r.Date)
        ap_, agf, aga, rest = form(a, r.Date)
        res_d

    return pd.concat([df, pd.DataFrame(rows)], axis = 1)
