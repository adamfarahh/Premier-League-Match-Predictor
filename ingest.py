import argparse
import io
from pathlib import Path

import pandas as pd
import numpy as np
import requests

RAW = Path("data/raw")
PROCESSED = Path("data/processed/matches.csv")
URL = "https://www.football-data.co.uk/league/league-match-data"