"""Convert pre-match decimal odds into outcome probabilities."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketOdds:
    home: float
    draw: float
    away: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value > 1 for value in (self.home, self.draw, self.away)):
            raise ValueError("Decimal odds must be finite numbers greater than 1.")

    @property
    def probabilities(self) -> tuple[float, float, float]:
        home, draw, away = 1 / self.home, 1 / self.draw, 1 / self.away
        total = home + draw + away
        return home / total, draw / total, away / total
