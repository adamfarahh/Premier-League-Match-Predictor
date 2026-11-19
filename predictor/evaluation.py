"""Expanding-window evaluation at monthly refit boundaries."""

import csv
import io
import math
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from statistics import fmean, stdev
from typing import Callable

from .data import OUTCOMES, Fixture
from .model import (
    LEGACY_MODEL, MIN_TRAINING_FIXTURES, STRENGTH_MODEL,
    OutcomeModel, Predictor, Probabilities, TrainingRow, most_likely,
)

DEVELOPMENT_START = date(2016, 8, 1)
DEVELOPMENT_END = date(2025, 5, 31)
RETROSPECTIVE_START = date(2025, 8, 1)


@dataclass(frozen=True, slots=True)
class Prediction:
    fixture: Fixture
    probabilities: Probabilities
    baseline: Probabilities
    trained_through: date
    training_matches: int = 0

    @property
    def prediction(self) -> str:
        return most_likely(self.probabilities)


@dataclass(frozen=True, slots=True)
class Scores:
    accuracy: float
    log_loss: float
    brier: float


@dataclass(frozen=True, slots=True)
class ModelSelection:
    selected: str
    fixtures: int
    simple: Scores
    adjusted: Scores
    adjusted_gain: float
    season_standard_error: float
    seasons: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    outcome: str
    lower: float
    upper: float
    matches: int
    mean_probability: float
    observed_frequency: float


@dataclass(frozen=True, slots=True)
class MarketComparison:
    matches: int
    model_scores: Scores
    market_scores: Scores


@dataclass(frozen=True, slots=True)
class ConfidentPicks:
    matches: int
    scores: Scores


@dataclass(frozen=True, slots=True)
class Backtest:
    since: date
    until: date
    predictions: tuple[Prediction, ...]

    @property
    def model_scores(self) -> Scores:
        return _scores(self.predictions, lambda row: row.probabilities)

    @property
    def baseline_scores(self) -> Scores:
        return _scores(self.predictions, lambda row: row.baseline)

    @property
    def confusion(self) -> list[list[int]]:
        counts = [[0 for _ in OUTCOMES] for _ in OUTCOMES]
        for row in self.predictions:
            actual, predicted = OUTCOMES.index(row.fixture.outcome), OUTCOMES.index(row.prediction)
            counts[actual][predicted] += 1
        return counts


def _scores(rows: tuple[Prediction, ...], probabilities: Callable[[Prediction], Probabilities]) -> Scores:
    actual = [OUTCOMES.index(row.fixture.outcome) for row in rows]
    distribution = [probabilities(row) for row in rows]
    return Scores(
        fmean(most_likely(probs) == row.fixture.outcome for row, probs in zip(rows, distribution, strict=True)),
        fmean(log_loss(probs, row.fixture.outcome) for row, probs in zip(rows, distribution, strict=True)),
        fmean(
            sum((prob - (index == target)) ** 2 for index, prob in enumerate(probs))
            for target, probs in zip(actual, distribution, strict=True)
        ),
    )


def log_loss(probabilities: Probabilities, outcome: str) -> float:
    return -math.log(max(probabilities[OUTCOMES.index(outcome)], 1e-15))


def _market_probabilities(row: Prediction) -> Probabilities:
    odds = row.fixture.market_odds
    assert odds is not None
    return odds.probabilities


def _model_probabilities(row: Prediction) -> Probabilities:
    return row.probabilities


def compare_market(report: Backtest) -> MarketComparison | None:
    """Compare both methods on the same fixtures with a pre-match market quote."""
    rows = tuple(row for row in report.predictions if row.fixture.market_odds is not None)
    if not rows:
        return None
    return MarketComparison(
        len(rows), _scores(rows, _model_probabilities), _scores(rows, _market_probabilities),
    )


def calibration(report: Backtest, *, market: bool = False, bins: int = 10) -> tuple[CalibrationBin, ...]:
    """One-vs-rest reliability points. Market bins use fixtures with published odds."""
    if bins < 2:
        raise ValueError("Calibration needs at least two bins.")
    grouped: dict[tuple[str, int], list[tuple[float, bool]]] = {}
    for row in report.predictions:
        if market and row.fixture.market_odds is None:
            continue
        probs = _market_probabilities(row) if market else row.probabilities
        for label, probability in zip(OUTCOMES, probs, strict=True):
            index = min(int(probability * bins), bins - 1)
            grouped.setdefault((label, index), []).append((probability, row.fixture.outcome == label))
    return tuple(
        CalibrationBin(label, index / bins, (index + 1) / bins, len(values),
                       fmean(prob for prob, _ in values), fmean(observed for _, observed in values))
        for (label, index), values in sorted(grouped.items(), key=lambda item: (OUTCOMES.index(item[0][0]), item[0][1]))
    )


def score_confident_picks(
    report: Backtest, min_confidence: float, *, market: bool = False,
) -> ConfidentPicks | None:
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")
    probabilities = _market_probabilities if market else _model_probabilities
    rows = tuple(
        row for row in report.predictions
        if (not market or row.fixture.market_odds is not None) and max(probabilities(row)) >= min_confidence
    )
    return ConfidentPicks(len(rows), _scores(rows, probabilities)) if rows else None


def default_backtest_start(predictor: Predictor) -> date:
    return predictor.first_forecast or predictor.fixtures[0].date


def _month_key(row: TrainingRow) -> tuple[int, int]:
    return row.fixture.date.year, row.fixture.date.month


def backtest(predictor: Predictor, since: date | None = None, until: date | None = None) -> Backtest:
    """Evaluate the full eligible archive by default, refitting each month."""
    since = since or predictor.fixtures[0].date
    until = until or predictor.fixtures[-1].date
    if until < since:
        raise ValueError("The backtest end date must be on or after its start date.")
    earliest = predictor.first_forecast
    if earliest is None or until < earliest:
        raise ValueError(
            f"Training requires at least {MIN_TRAINING_FIXTURES} earlier fixtures "
            "with at least one home win, draw, and away win. Choose a later end date."
        )
    since = max(since, earliest)
    eligible = tuple(row for row in predictor.rows if since <= row.fixture.date <= until)
    if not eligible:
        raise ValueError("There are no fixtures in the selected backtest range.")

    predictions = []
    for _, month in groupby(eligible, key=_month_key):
        batch = tuple(month)
        cutoff = batch[0].fixture.date
        training = tuple(row for row in predictor.rows if row.fixture.date < cutoff)
        try:
            model = OutcomeModel(training, cutoff, predictor.model)
        except ValueError as error:
            raise ValueError(f"Cannot start the backtest at {cutoff}. {error} Choose a later start date.") from error
        probabilities = model.predict([row.features for row in batch])
        predictions.extend(
            Prediction(row.fixture, probs, model.baseline, training[-1].fixture.date, len(training))
            for row, probs in zip(batch, probabilities, strict=True)
        )
    return Backtest(since, until, tuple(predictions))


def select_model(
    fixtures: tuple[Fixture, ...], start: date = DEVELOPMENT_START, end: date = DEVELOPMENT_END,
) -> ModelSelection:
    """Prefer the simpler model unless the paired log-loss gain exceeds one season-level SE."""
    development = tuple(match for match in fixtures if match.date <= end)
    simple = backtest(Predictor(development, model=LEGACY_MODEL), start, end)
    adjusted = backtest(Predictor(development, model=STRENGTH_MODEL), start, end)
    season_differences: dict[int, list[float]] = {}
    all_differences = []
    for old, new in zip(simple.predictions, adjusted.predictions, strict=True):
        if old.fixture != new.fixture:
            raise ValueError("Model selection needs paired predictions for the same fixtures.")
        day, outcome = old.fixture.date, old.fixture.outcome
        season = day.year - (day.month < 8)
        gain = log_loss(old.probabilities, outcome) - log_loss(new.probabilities, outcome)
        all_differences.append(gain)
        season_differences.setdefault(season, []).append(gain)
    means = [fmean(differences) for differences in season_differences.values()]
    if len(means) < 2:
        raise ValueError("Model selection needs at least two development seasons.")
    standard_error = stdev(means) / math.sqrt(len(means))
    adjusted_gain = fmean(all_differences)
    selected = STRENGTH_MODEL if adjusted_gain > standard_error else LEGACY_MODEL
    return ModelSelection(
        selected, len(all_differences), simple.model_scores, adjusted.model_scores,
        adjusted_gain, standard_error, len(means),
    )


def _fixture_fields(fixture: Fixture) -> list[str | int | float]:
    return [
        fixture.date.isoformat(), fixture.home, fixture.away, fixture.home_goals, fixture.away_goals,
    ]


def backtest_csv(report: Backtest, *, include_market: bool = False) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    header = [
        "date", "home_team", "away_team", "home_goals", "away_goals", "actual", "prediction",
        "p_home", "p_draw", "p_away", "baseline_p_home", "baseline_p_draw", "baseline_p_away",
        "trained_through",
    ]
    if include_market:
        header.extend(("market_prediction", "market_p_home", "market_p_draw", "market_p_away"))
    writer.writerow(header)
    for row in report.predictions:
        values = [
            *_fixture_fields(row.fixture), row.fixture.outcome, row.prediction,
            *(f"{value:.8f}" for value in (*row.probabilities, *row.baseline)), row.trained_through,
        ]
        if include_market:
            if row.fixture.market_odds is None:
                values.extend([""] * 4)
            else:
                probabilities = _market_probabilities(row)
                values.extend([most_likely(probabilities), *(f"{value:.8f}" for value in probabilities)])
        writer.writerow(values)
    return buffer.getvalue()
