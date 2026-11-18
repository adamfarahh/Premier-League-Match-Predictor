import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
HOME_ADV = 60.0
WINDOW = 5
FEATURES = ["elo_diff", "form_pts_diff", "gf_diff", "ga_diff", "rest_diff"]

def _expected(elo_h: float, elo_a: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_h + HOME_ADV - elo_a) / 400.0))
