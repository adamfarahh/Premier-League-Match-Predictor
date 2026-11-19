"""Load scheduled fixtures and forecast a season from dated snapshots."""

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import Fixture, club_name, parse_date
from .evaluation import backtest
from .model import Predictor, Probabilities, most_likely
from .network import read_url


LONDON = ZoneInfo("Europe/London")
FPL = "https://fantasy.premierleague.com/api"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ScheduledMatch:
    date: date
    home: str
    away: str

    @property
    def pair(self) -> tuple[str, str]:
        return self.home, self.away


@dataclass(frozen=True, slots=True)
class OfficialMatch:
    fixture_id: int
    gameweek: int | None
    kickoff: datetime | None
    home: str
    away: str
    finished: bool
    score: tuple[int, int] | None

    @property
    def day(self) -> date | None:
        return self.kickoff.astimezone(LONDON).date() if self.kickoff else None


def official_schedule(season: int, source: Path | None = None) -> tuple[OfficialMatch, ...]:
    if source:
        clubs = json.loads((source / "fpl-teams.json").read_bytes())["teams"]
        games = json.loads((source / "fpl-fixtures.json").read_bytes())
    else:
        clubs = json.loads(read_url(f"{FPL}/bootstrap-static/"))["teams"]
        games = json.loads(read_url(f"{FPL}/fixtures/"))
    names = {club["id"]: club_name(club["name"]) for club in clubs}
    if len(names) != 20 or len(games) != 380:
        raise ValueError("The official feed does not contain a complete Premier League schedule.")
    results = []
    for game in games:
        kickoff = datetime.fromisoformat(game["kickoff_time"]) if game["kickoff_time"] else None
        if kickoff is not None and kickoff.utcoffset() is None:
            raise ValueError(f"Official fixture {game['id']} has a kickoff without a timezone.")
        finished = game["finished"]
        goals = game["team_h_score"], game["team_a_score"]
        if finished and (kickoff is None or any(not isinstance(goal, int) or goal < 0 for goal in goals)):
            raise ValueError(f"Official fixture {game['id']} has an incomplete final result.")
        match = OfficialMatch(
            game["id"], game["event"], kickoff, names[game["team_h"]], names[game["team_a"]],
            finished, goals if finished else None,
        )
        if match.day and not date(season, 7, 1) <= match.day < date(season + 1, 7, 1):
            raise ValueError(f"The official feed does not cover the requested {season}/{season + 1} season.")
        results.append(match)
    if len({(match.home, match.away) for match in results}) != 380 or len({match.fixture_id for match in results}) != 380:
        raise ValueError("Duplicate games in the official schedule.")
    return tuple(results)


def official_history(
    season: int, matches: tuple[OfficialMatch, ...], archive: tuple[Fixture, ...], now: datetime,
) -> tuple[dict[int, Fixture], tuple[Fixture, ...]]:
    """Keep current-season scores that the league feed has confirmed."""
    start = date(season, 7, 1)
    archive_by_key = {(row.date, row.home, row.away): row for row in archive}
    confirmed = {}
    for match in matches:
        if not match.finished:
            continue
        if match.day is None or match.kickoff >= now or match.score is None:
            raise ValueError(f"Official fixture {match.fixture_id} reports a result before its kickoff.")
        row = archive_by_key.get((match.day, match.home, match.away))
        if row and match.score != (row.home_goals, row.away_goals):
            raise ValueError(f"Archive and official score disagree for {match.home} vs {match.away}.")
        confirmed[match.fixture_id] = row or Fixture(match.day, match.home, match.away, *match.score)
    known = tuple(row for row in archive if row.date < start) + tuple(confirmed.values())
    return confirmed, tuple(sorted(known, key=lambda row: (row.date, row.home, row.away)))


def load_schedule(path: str | Path) -> tuple[ScheduledMatch, ...]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        matches = tuple(
            ScheduledMatch(parse_date(row["date"]), club_name(row["home_team"]), club_name(row["away_team"]))
            for row in csv.DictReader(file)
        )
    clubs = {club for match in matches for club in match.pair}
    expected = {(home, away) for home in clubs for away in clubs if home != away}
    if len(clubs) != 20 or len(matches) != 380 or {match.pair for match in matches} != expected:
        raise ValueError("The season schedule needs all 380 distinct home and away fixtures for 20 clubs.")
    return tuple(sorted(matches, key=lambda match: (match.date, *match.pair)))


@dataclass(frozen=True, slots=True)
class SeasonPrediction:
    match: ScheduledMatch
    probabilities: Probabilities
    trained_through: date
    training_matches: int
    generated_at: datetime
    status: str
    actual: Fixture | None = None


def forecast_season(
    predictor: Predictor, schedule: tuple[ScheduledMatch, ...], *, as_of: datetime,
) -> tuple[SeasonPrediction, ...]:
    if not schedule or as_of.utcoffset() is None:
        raise ValueError("Provide a season schedule and a timezone-aware forecast timestamp.")
    year = min(match.date for match in schedule).year
    start, end = date(year, 7, 1), date(year + 1, 6, 30)
    if any(not start <= match.date <= end for match in schedule):
        raise ValueError("Scheduled dates must be in one season.")
    today = as_of.astimezone(LONDON).date()
    known = tuple(match for match in predictor.fixtures if match.date <= today)
    if not known:
        raise ValueError("No results available at the forecast timestamp.")
    if len(known) != len(predictor.fixtures):
        predictor = Predictor(known, model=predictor.model)
    latest = known[-1].date
    schedule_pairs = {match.pair for match in schedule}
    replays = {}
    played = backtest(predictor, start, min(end, latest)).predictions if latest >= start else ()
    for row in played:
        fixture = row.fixture
        if (fixture.home, fixture.away) not in schedule_pairs:
            raise ValueError("The archive contains a season matchup missing from the schedule.")
        replays[fixture.home, fixture.away] = SeasonPrediction(
            ScheduledMatch(fixture.date, fixture.home, fixture.away),
            row.probabilities, row.trained_through, row.training_matches, as_of, "Completed", fixture,
        )
    predictions = list(replays.values())
    pending = [match for match in schedule if match.pair not in replays]

    def cutoff(match: ScheduledMatch) -> date:
        return min(match.date, latest + timedelta(days=1))

    for day, group in groupby(sorted(pending, key=cutoff), key=cutoff):
        batch = list(group)
        snapshot = predictor.snapshot(day)
        for match, probs in zip(batch, snapshot.predict([match.pair for match in batch]), strict=True):
            status = "Upcoming" if match.date >= today else "Result missing"
            predictions.append(SeasonPrediction(
                match, probs, snapshot.trained_through, snapshot.training_matches, as_of, status,
            ))
    return tuple(sorted(predictions, key=lambda row: (row.match.date, *row.match.pair)))


def season_csv(predictions: tuple[SeasonPrediction, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([
        "date", "home_team", "away_team", "status", "pick", "prediction", "p_home", "p_draw", "p_away",
        "actual", "home_goals", "away_goals", "trained_through", "training_matches", "generated_at",
    ])
    for row in predictions:
        match, actual = row.match, row.actual
        call = most_likely(row.probabilities)
        writer.writerow([
            match.date, *match.pair, row.status, {"H": match.home, "D": "Draw", "A": match.away}[call], call,
            *(f"{p:.8f}" for p in row.probabilities),
            actual.outcome if actual else None, actual.home_goals if actual else None,
            actual.away_goals if actual else None, row.trained_through, row.training_matches, row.generated_at.isoformat(),
        ])
    return buffer.getvalue()
