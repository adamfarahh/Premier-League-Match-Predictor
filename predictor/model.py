"""Build pre-match features and fit a three-outcome probability model."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import groupby
from statistics import fmean
from typing import TypeAlias

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import OUTCOMES, Fixture, club_name
from .strength import LeagueStrength


INITIAL_RATING = 1500.0
ELO_K = 24
HOME_ADVANTAGE = 65.0
ELO_IDLE_GRACE_DAYS = 90
ELO_IDLE_HALF_LIFE_DAYS = 365
FORM_WINDOW = 5
FORM_MAX_AGE_DAYS = 365
MIN_TRAINING_FIXTURES = 60
STRENGTH_TRAINING_HALF_LIFE_DAYS = 730
LOGISTIC_C = 0.1
LEGACY_MODEL = "legacy"
STRENGTH_MODEL = "strength"
DEFAULT_MODEL = LEGACY_MODEL
MODEL_CHOICES = (LEGACY_MODEL, STRENGTH_MODEL)
MODEL_LABELS = {LEGACY_MODEL: "Elo + recent form", STRENGTH_MODEL: "Elo + adjusted strength"}
MODEL_REVISION = "1"
ELO_FEATURES = ("elo_gap", "elo_distance_from_even")
FEATURE_NAMES = {
    LEGACY_MODEL: (*ELO_FEATURES, "recent_points_gap", "recent_goal_gap", "recent_shots_on_target_gap"),
    STRENGTH_MODEL: (
        *ELO_FEATURES,
        *(f"{metric}_{measure}" for measure in ("difference", "total", "distance_from_even")
          for metric in ("goals", "shots_on_target", "shots")),
    ),
}

Probabilities: TypeAlias = tuple[float, float, float]
Features: TypeAlias = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TeamResult:
    date: date
    opponent: str
    venue: str
    goals_for: int
    goals_against: int
    on_target_for: int | None
    on_target_against: int | None

    @property
    def result(self) -> str:
        if self.goals_for > self.goals_against:
            return "W"
        if self.goals_for < self.goals_against:
            return "L"
        return "D"


@dataclass(slots=True)
class TeamState:
    rating: float = INITIAL_RATING
    matches_played: int = 0
    recent: deque[TeamResult] = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))

    def rating_on(self, day: date) -> float:
        if not self.recent:
            return INITIAL_RATING
        inactivity = max(0, (day - self.recent[-1].date).days - ELO_IDLE_GRACE_DAYS)
        return INITIAL_RATING + (self.rating - INITIAL_RATING) * 0.5 ** (inactivity / ELO_IDLE_HALF_LIFE_DAYS)

    def form(self, day: date) -> tuple[float, float, float]:
        recent = [match for match in self.recent if (day - match.date).days <= FORM_MAX_AGE_DAYS]
        if not recent:
            return 1.35, 0.0, 0.0
        points = fmean({"W": 3, "D": 1, "L": 0}[match.result] for match in recent)
        goals = fmean(match.goals_for - match.goals_against for match in recent)
        shots = [
            match.on_target_for - match.on_target_against
            for match in recent
            if match.on_target_for is not None and match.on_target_against is not None
        ]
        return points, goals, fmean(shots) if shots else 0.0


class LeagueHistory:
    """Pre-match reads are separate from recording completed results."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.teams: dict[str, TeamState] = defaultdict(TeamState)
        self.strength = LeagueStrength() if model == STRENGTH_MODEL else None
        self.last_date: date | None = None

    def team(self, name: str) -> TeamState:
        return self.teams.get(name) or TeamState()

    def features(self, day: date, home: str, away: str) -> Features:
        if self.last_date is not None and self.last_date >= day:
            raise ValueError("Feature history must end strictly before the forecast date.")
        home_state, away_state = self.team(home), self.team(away)
        rating_diff = home_state.rating_on(day) - away_state.rating_on(day)
        elo = rating_diff, abs(rating_diff + HOME_ADVANTAGE)
        if self.strength is not None:
            return (*elo, *self.strength.features(day, home, away))
        form_diffs = tuple(
            home_value - away_value
            for home_value, away_value in zip(home_state.form(day), away_state.form(day), strict=True)
        )
        return (*elo, *form_diffs)

    def record(self, fixture: Fixture) -> None:
        if self.last_date is not None and fixture.date < self.last_date:
            raise ValueError("Record completed fixtures in date order.")
        self.last_date = fixture.date
        if self.strength is not None:
            self.strength.record(fixture)
        home, away = self.teams[fixture.home], self.teams[fixture.away]
        home_rating, away_rating = home.rating_on(fixture.date), away.rating_on(fixture.date)
        gap = home_rating + HOME_ADVANTAGE - away_rating
        expected = 1 / (1 + 10 ** (-gap / 400))
        actual = {"H": 1, "D": 0.5, "A": 0}[fixture.outcome]
        change = ELO_K * (actual - expected)
        home.rating = home_rating + change
        away.rating = away_rating - change
        home.matches_played += 1
        away.matches_played += 1
        (_, home_target), (_, away_target) = fixture.stats.training_shots()
        home.recent.append(TeamResult(
            fixture.date, fixture.away, "Home", fixture.home_goals, fixture.away_goals,
            home_target, away_target,
        ))
        away.recent.append(TeamResult(
            fixture.date, fixture.home, "Away", fixture.away_goals, fixture.home_goals,
            away_target, home_target,
        ))


@dataclass(frozen=True, slots=True)
class TrainingRow:
    fixture: Fixture
    features: Features


def training_rows(fixtures: tuple[Fixture, ...], model: str = DEFAULT_MODEL) -> tuple[TrainingRow, ...]:
    state = LeagueHistory(model)
    rows = []
    for day, group in groupby(fixtures, key=lambda match: match.date):
        same_day = tuple(group)
        rows.extend(TrainingRow(match, state.features(day, match.home, match.away)) for match in same_day)
        for match in same_day:
            state.record(match)
    return tuple(rows)


def most_likely(probabilities: Probabilities) -> str:
    return OUTCOMES[int(np.argmax(probabilities))]


class OutcomeModel:
    def __init__(self, rows: tuple[TrainingRow, ...], cutoff: date, model: str) -> None:
        if any(row.fixture.date >= cutoff for row in rows):
            raise ValueError("Training features and outcomes must precede the fit cutoff.")
        targets = [row.fixture.outcome for row in rows]
        if len(rows) < MIN_TRAINING_FIXTURES or set(targets) != set(OUTCOMES):
            raise ValueError(
                f"Training requires at least {MIN_TRAINING_FIXTURES} earlier fixtures "
                "with at least one home win, draw, and away win."
            )
        self.estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=LOGISTIC_C, max_iter=1000),
        )
        weights = None
        if model == STRENGTH_MODEL:
            # Keep the penalty scale stable for dates far past the last available result.
            gap = (cutoff - rows[-1].fixture.date).days
            offset = max(0, gap - STRENGTH_TRAINING_HALF_LIFE_DAYS)
            weights = [
                0.5 ** (((cutoff - row.fixture.date).days - offset) / STRENGTH_TRAINING_HALF_LIFE_DAYS)
                for row in rows
            ]
        self.estimator.fit([row.features for row in rows], targets, logisticregression__sample_weight=weights)
        self.order = [list(self.estimator.classes_).index(outcome) for outcome in OUTCOMES]
        self.baseline: Probabilities = tuple(
            (targets.count(outcome) + 1) / (len(targets) + len(OUTCOMES)) for outcome in OUTCOMES
        )

    def predict(self, features: list[Features]) -> list[Probabilities]:
        probabilities = self.estimator.predict_proba(features)[:, self.order]
        return [tuple(map(float, row)) for row in probabilities]


@dataclass(frozen=True, slots=True)
class ClubForm:
    name: str
    rating: int
    matches_played: int
    recent: tuple[TeamResult, ...]


@dataclass(frozen=True, slots=True)
class Forecast:
    date: date
    home: ClubForm
    away: ClubForm
    probabilities: Probabilities
    trained_through: date
    training_matches: int
    actual: Fixture | None

    @property
    def prediction(self) -> str:
        return most_likely(self.probabilities)


def _club_form(name: str, state: TeamState, day: date) -> ClubForm:
    recent = tuple(match for match in state.recent if (day - match.date).days <= FORM_MAX_AGE_DAYS)
    return ClubForm(name, round(state.rating_on(day)), state.matches_played, recent)


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    cutoff: date
    trained_through: date
    training_matches: int
    model: OutcomeModel
    history: LeagueHistory

    def predict(self, pairings: list[tuple[str, str]]) -> list[Probabilities]:
        return self.model.predict([self.history.features(self.cutoff, home, away) for home, away in pairings])


class Predictor:
    def __init__(self, fixtures: tuple[Fixture, ...], *, model: str = DEFAULT_MODEL) -> None:
        if not fixtures:
            raise ValueError("Fixture history is empty.")
        if model not in MODEL_CHOICES:
            raise ValueError(f"Unknown model {model!r}. Choose from {', '.join(MODEL_CHOICES)}.")
        self.model = model
        self.fixtures = tuple(sorted(fixtures, key=lambda fixture: (fixture.date, fixture.home, fixture.away)))
        self.rows = training_rows(self.fixtures, model)
        self.clubs = sorted({team for match in self.fixtures for team in (match.home, match.away)})
        self._names = {team.casefold(): team for team in self.clubs}
        self._results = {(match.date, match.home, match.away): match for match in self.fixtures}
        self.first_forecast = None
        outcomes = set()
        for count, match in enumerate(self.fixtures, 1):
            outcomes.add(match.outcome)
            if count >= MIN_TRAINING_FIXTURES and outcomes == set(OUTCOMES) and match.date < date.max:
                self.first_forecast = match.date + timedelta(days=1)
                break

    def snapshot(self, cutoff: date) -> ForecastSnapshot:
        prior = tuple(match for match in self.fixtures if match.date < cutoff)
        rows = self.rows[:len(prior)]
        model = OutcomeModel(rows, cutoff, self.model)
        state = LeagueHistory(self.model)
        for match in prior:
            state.record(match)
        return ForecastSnapshot(cutoff, prior[-1].date, len(prior), model, state)

    def forecast(self, day: date, home: str, away: str) -> Forecast:
        home, away = club_name(home), club_name(away)
        home = self._names.get(home.casefold(), home)
        away = self._names.get(away.casefold(), away)
        if home.casefold() == away.casefold():
            raise ValueError("Choose two different clubs.")
        snapshot = self.snapshot(day)
        probabilities, = snapshot.predict([(home, away)])
        return Forecast(
            day, _club_form(home, snapshot.history.team(home), day), _club_form(away, snapshot.history.team(away), day),
            probabilities, snapshot.trained_through, snapshot.training_matches, self._results.get((day, home, away)),
        )
