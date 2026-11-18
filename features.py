import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
HOME_ADV = 60.0
WINDOW = 5
FEATURES = ["elo_diff", "form_pts_diff", "gf_diff", "ga_diff", "rest_diff"]