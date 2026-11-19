"""Prospective forecast ledger and gameweek scoring, stored in SQLite."""

import csv
import io
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from statistics import fmean
from typing import Callable

from .data import Fixture
from .evaluation import log_loss
from .model import DEFAULT_MODEL, MODEL_REVISION, Predictor
from .season import LONDON, OfficialMatch, official_history, utc_now


DEFAULT_SCOREBOARD = Path("scoreboard.sqlite3")
FORECAST_HORIZON = timedelta(hours=72)
SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_state (
    season INTEGER PRIMARY KEY, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    season INTEGER NOT NULL, fixture_id INTEGER NOT NULL,
    gameweek INTEGER, kickoff TEXT, home TEXT NOT NULL, away TEXT NOT NULL,
    actual TEXT, market TEXT, PRIMARY KEY (season, fixture_id)
);
CREATE TABLE IF NOT EXISTS forecasts (
    season INTEGER NOT NULL, fixture_id INTEGER NOT NULL,
    issued_at TEXT NOT NULL, kickoff_at_issue TEXT NOT NULL, home TEXT NOT NULL,
    away TEXT NOT NULL, model TEXT NOT NULL, model_revision TEXT NOT NULL,
    trained_through TEXT NOT NULL, training_matches INTEGER NOT NULL,
    probabilities TEXT NOT NULL, PRIMARY KEY (season, fixture_id)
);
"""


def _timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("Forecast timestamps must include a timezone.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def sync_scoreboard(
    path: Path, season: int, matches: tuple[OfficialMatch, ...], archive: tuple[Fixture, ...],
    *, clock: Callable[[], datetime] = utc_now, model: str = DEFAULT_MODEL, gameweeks: int | None = None,
) -> int:
    """Issue at most one forecast per fixture and attach official results as they arrive."""
    if gameweeks is not None and gameweeks < 1:
        raise ValueError("The number of upcoming gameweeks must be positive.")
    now = clock()
    _timestamp(now)
    confirmed, known = official_history(season, matches, archive, now)
    with closing(sqlite3.connect(path, timeout=30)) as db:
        db.executescript(SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        with db:
            previous = db.execute("SELECT updated_at FROM sync_state WHERE season = ?", (season,)).fetchone()
            if previous and previous[0] > _timestamp(now):
                raise ValueError("The update timestamp is older than the last scoreboard sync.")
            recorded = {
                row[0] for row in db.execute("SELECT fixture_id FROM forecasts WHERE season = ?", (season,))
            }
            games = []
            for match in matches:
                actual = confirmed.get(match.fixture_id)
                odds = actual.market_odds if actual else None
                games.append((
                    season, match.fixture_id, match.gameweek,
                    _timestamp(match.kickoff) if match.kickoff else None, match.home, match.away,
                    actual.outcome if actual else None, json.dumps(odds.probabilities) if odds else None,
                ))
            db.executemany("""
                INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (season, fixture_id) DO UPDATE SET
                    gameweek = excluded.gameweek, kickoff = excluded.kickoff,
                    home = excluded.home, away = excluded.away,
                    actual = COALESCE(excluded.actual, games.actual),
                    market = COALESCE(excluded.market, games.market)
            """, games)

            upcoming = [
                match for match in matches
                if not match.finished and match.gameweek is not None
                and match.kickoff is not None and now < match.kickoff
            ]
            if gameweeks is None:
                upcoming = [match for match in upcoming if match.kickoff <= now + FORECAST_HORIZON]
            else:
                rounds = dict.fromkeys(
                    match.gameweek for match in sorted(upcoming, key=lambda match: (match.kickoff, match.fixture_id))
                )
                selected = set(list(rounds)[:gameweeks])
                upcoming = [match for match in upcoming if match.gameweek in selected]
            pending = [match for match in upcoming if match.fixture_id not in recorded]
            issued_count = 0
            if pending:
                predictor = Predictor(known, model=model)
                latest = known[-1].date

                def cutoff(match: OfficialMatch) -> date:
                    return min(match.kickoff.astimezone(LONDON).date(), latest + timedelta(days=1))

                for day, group in groupby(sorted(pending, key=cutoff), key=cutoff):
                    batch = list(group)
                    snapshot = predictor.snapshot(day)
                    probabilities = snapshot.predict([(match.home, match.away) for match in batch])
                    issued_at = clock()
                    forecasts = [
                        (season, match.fixture_id, _timestamp(issued_at), _timestamp(match.kickoff),
                         match.home, match.away, model, MODEL_REVISION, snapshot.trained_through.isoformat(),
                         snapshot.training_matches, json.dumps(probs))
                        for match, probs in zip(batch, probabilities, strict=True)
                        if issued_at < match.kickoff
                    ]
                    db.executemany("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", forecasts)
                    issued_count += len(forecasts)
            db.execute("""
                INSERT INTO sync_state VALUES (?, ?)
                ON CONFLICT (season) DO UPDATE SET updated_at = excluded.updated_at
            """, (season, _timestamp(clock())))
    return issued_count


@dataclass(frozen=True, slots=True)
class GameweekScore:
    gameweek: int
    scheduled: int
    completed: int
    forecasted: int
    scored: int
    paired: int
    model_log_loss: float | None
    paired_model_log_loss: float | None
    market_log_loss: float | None


@dataclass(frozen=True, slots=True)
class Scoreboard:
    season: int
    updated_at: str | None
    gameweeks: tuple[GameweekScore, ...]


def logged_forecasts(path: Path, season: int) -> list[dict]:
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT p.*, g.gameweek, g.kickoff AS current_kickoff,
                p.issued_at < g.kickoff AND p.home = g.home AND p.away = g.away AS eligible
            FROM forecasts AS p JOIN games AS g USING (season, fixture_id)
            WHERE p.season = ? ORDER BY g.kickoff, p.fixture_id
        """, (season,)).fetchall()
    records = [dict(row) for row in rows]
    for row in records:
        row["probabilities"] = json.loads(row["probabilities"])
        row["eligible"] = bool(row["eligible"])
    return records


def forecast_log_csv(path: Path) -> str:
    """Export immutable pre-kickoff records, without results, odds, or revised kickoff times."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "season", "fixture_id", "issued_at", "kickoff_at_issue", "home", "away", "model", "model_revision",
        "trained_through", "training_matches", "p_home", "p_draw", "p_away",
    ])
    if path.is_file():
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
            rows = db.execute("SELECT * FROM forecasts ORDER BY issued_at, season, fixture_id").fetchall()
        writer.writerows([*row[:-1], *json.loads(row[-1])] for row in rows)
    return buffer.getvalue()


def read_scoreboard(path: Path, season: int) -> Scoreboard:
    if not path.is_file():
        return Scoreboard(season, None, ())
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        updated = db.execute("SELECT updated_at FROM sync_state WHERE season = ?", (season,)).fetchone()
        rows = db.execute("""
            SELECT g.gameweek, g.actual, g.market, p.probabilities FROM games AS g
            LEFT JOIN forecasts AS p ON g.season = p.season AND g.fixture_id = p.fixture_id
                AND p.issued_at < g.kickoff AND p.home = g.home AND p.away = g.away
            WHERE g.season = ? AND g.gameweek IS NOT NULL
            ORDER BY g.gameweek, g.fixture_id
        """, (season,)).fetchall()
    weeks = []
    for gameweek, group in groupby(rows, key=lambda row: row["gameweek"]):
        batch = list(group)
        logged = [row for row in batch if row["probabilities"] is not None]
        scored = [row for row in logged if row["actual"] is not None]
        matched = [row for row in scored if row["market"] is not None]
        weeks.append(GameweekScore(
            gameweek, len(batch), sum(row["actual"] is not None for row in batch),
            len(logged), len(scored), len(matched),
            fmean(log_loss(json.loads(row["probabilities"]), row["actual"]) for row in scored) if scored else None,
            fmean(log_loss(json.loads(row["probabilities"]), row["actual"]) for row in matched) if matched else None,
            fmean(log_loss(json.loads(row["market"]), row["actual"]) for row in matched) if matched else None,
        ))
    return Scoreboard(season, updated[0] if updated else None, tuple(weeks))


def scoreboard_csv(report: Scoreboard) -> str:
    buffer = io.StringIO(newline="")
    columns = list(GameweekScore.__dataclass_fields__)
    writer = csv.writer(buffer)
    writer.writerow(["season", "updated_at", *columns])
    writer.writerows([report.season, report.updated_at, *(getattr(row, key) for key in columns)]
                     for row in report.gameweeks)
    return buffer.getvalue()
