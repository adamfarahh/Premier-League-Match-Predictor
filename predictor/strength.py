"""Update opponent-adjusted attack and defense ratings after each match."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from .data import Fixture


# Standardize goals, shots on target, and total shots to comparable scales.
SCALES = (1.3, 2.6, 5.3)
HOME_ADVANTAGES = (0.24, 0.25, 0.25)
UPDATE_RATE = 0.04
GOAL_AVERAGE_RATE = 0.003
IDLE_GRACE_DAYS = 25
IDLE_HALF_LIFE_DAYS = 240


@dataclass(slots=True)
class MetricRating:
    attack: float = 0.0
    defense: float = 0.0

    def decay(self, weight: float) -> None:
        self.attack *= weight
        self.defense *= weight


@dataclass(slots=True)
class ClubStrength:
    ratings: tuple[MetricRating, ...] = field(
        default_factory=lambda: tuple(MetricRating() for _ in SCALES)
    )
    last_match: date | None = None

    def weight(self, day: date) -> float:
        if self.last_match is None:
            return 1.0
        inactive_days = max(0, (day - self.last_match).days - IDLE_GRACE_DAYS)
        return 0.5 ** (inactive_days / IDLE_HALF_LIFE_DAYS)


def _update(observed: int | None, average: float, scale: float, attack: float, defense: float) -> float:
    if observed is None:
        return 0.0
    expected = attack - defense
    residual = (observed - average) / scale - expected
    # Limit the influence of an unusually high or low count.
    return UPDATE_RATE * max(-3, min(3, residual))


class LeagueStrength:
    def __init__(self) -> None:
        self.teams: dict[str, ClubStrength] = defaultdict(ClubStrength)
        self.home_goal_average = 1.5
        self.away_goal_average = 1.2

    def features(self, day: date, home: str, away: str) -> tuple[float, ...]:
        home_club = self.teams.get(home) or ClubStrength()
        away_club = self.teams.get(away) or ClubStrength()
        home_weight, away_weight = home_club.weight(day), away_club.weight(day)
        differences, totals, gaps = [], [], []
        for home_metric, away_metric, advantage in zip(
            home_club.ratings, away_club.ratings, HOME_ADVANTAGES, strict=True,
        ):
            home_edge = home_metric.attack * home_weight - away_metric.defense * away_weight
            away_edge = away_metric.attack * away_weight - home_metric.defense * home_weight
            differences.append(home_edge - away_edge)
            totals.append(home_edge + away_edge)
            gaps.append(abs(home_edge - away_edge + advantage))
        return (*differences, *totals, *gaps)

    def record(self, match: Fixture) -> None:
        home, away = self.teams[match.home], self.teams[match.away]
        home_weight, away_weight = home.weight(match.date), away.weight(match.date)
        home.last_match = away.last_match = match.date
        (home_shots, home_target), (away_shots, away_target) = match.stats.training_shots()
        home_counts = (match.home_goals, home_target, home_shots)
        away_counts = (match.away_goals, away_target, away_shots)
        home_averages = (self.home_goal_average, 4.8, 14.0)
        away_averages = (self.away_goal_average, 3.8, 11.0)
        for index, scale in enumerate(SCALES):
            home_metric, away_metric = home.ratings[index], away.ratings[index]
            home_metric.decay(home_weight)
            away_metric.decay(away_weight)
            home_change = _update(
                home_counts[index], home_averages[index], scale, home_metric.attack, away_metric.defense,
            )
            away_change = _update(
                away_counts[index], away_averages[index], scale, away_metric.attack, home_metric.defense,
            )
            home_metric.attack += home_change
            away_metric.defense -= home_change
            away_metric.attack += away_change
            home_metric.defense -= away_change
        self.home_goal_average += GOAL_AVERAGE_RATE * (match.home_goals - self.home_goal_average)
        self.away_goal_average += GOAL_AVERAGE_RATE * (match.away_goals - self.away_goal_average)
