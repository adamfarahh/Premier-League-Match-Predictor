"""Read paired FBref results or a one-row-per-fixture CSV."""

import csv
import io
import math
from dataclasses import astuple, dataclass, fields, replace
from datetime import date
from pathlib import Path

from .market import MarketOdds


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = PROJECT_DIR / "premier_league.csv"
OUTCOMES = ("H", "D", "A")
ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Cardiff": "Cardiff City",
    "Coventry": "Coventry City",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'ham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "QPR": "Queens Park Rangers",
    "Stoke": "Stoke City",
    "Spurs": "Tottenham Hotspur",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wigan": "Wigan Athletic",
    "Wolves": "Wolverhampton Wanderers",
}
CLUB_NAMES = {name.casefold(): full for name, full in ALIASES.items()}
CLUB_NAMES.update({full.casefold(): full for full in ALIASES.values()})
FIXTURE_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}
TEAM_COLUMNS = {"date", "team", "opponent", "venue", "gf", "ga"}
ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")


def club_name(value: str) -> str:
    name = " ".join(value.replace("’", "'").split())
    if not name or not name[0].isalnum():
        raise ValueError("Club names must start with a letter or number.")
    return CLUB_NAMES.get(name.casefold(), name)


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("Dates must use YYYY-MM-DD.") from None
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return parsed


@dataclass(frozen=True, slots=True)
class MatchStats:
    kickoff: str | None = None
    referee: str | None = None
    home_ht_goals: int | None = None
    away_ht_goals: int | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_fouls: int | None = None
    away_fouls: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_yellows: int | None = None
    away_yellows: int | None = None
    home_reds: int | None = None
    away_reds: int | None = None

    @property
    def inconsistent_shots(self) -> bool:
        return any(_clean_shots(total, target) != (total, target) for total, target in (
            (self.home_shots, self.home_shots_on_target),
            (self.away_shots, self.away_shots_on_target),
        ))

    def training_shots(self) -> tuple[tuple[int | None, int | None], tuple[int | None, int | None]]:
        """Return home and away shot counts with conflicting pairs marked missing."""
        return (
            _clean_shots(self.home_shots, self.home_shots_on_target),
            _clean_shots(self.away_shots, self.away_shots_on_target),
        )


def _clean_shots(total: int | None, target: int | None) -> tuple[int | None, int | None]:
    if total is not None and target is not None and target > total:
        return None, None
    return total, target


TEXT_STATS = ("kickoff", "referee")
COUNT_STATS = tuple(field.name for field in fields(MatchStats) if field.name not in TEXT_STATS)


@dataclass(frozen=True, slots=True)
class Fixture:
    date: date
    home: str
    away: str
    home_goals: int
    away_goals: int
    home_xg: float | None = None
    away_xg: float | None = None
    stats: MatchStats = MatchStats()
    market_odds: MarketOdds | None = None

    @property
    def outcome(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


def _count(value: str, column: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{column} must be a non-negative whole number.")
    return int(value)


def _legacy_count(value: str, column: str) -> int | None:
    # Pandas wrote these legacy columns as decimals even for whole counts.
    return _count(value.removesuffix(".0"), column) if value else None


def _xg(value: str, column: str) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{column} must be a finite, non-negative number.")
    return number


def _row_fixture(row: dict[str, str], paired: bool) -> Fixture:
    day = parse_date(row["date"])
    if not paired:
        stats = MatchStats(
            **{key: row.get(key) or None for key in TEXT_STATS},
            **{key: _count(row[key], key) for key in COUNT_STATS if row.get(key)},
        )
        odds_fields = tuple(row.get(key, "") for key in ODDS_COLUMNS)
        odds = MarketOdds(*map(float, odds_fields)) if any(odds_fields) else None
        return Fixture(
            day, club_name(row["home_team"]), club_name(row["away_team"]),
            _count(row["home_goals"], "home_goals"), _count(row["away_goals"], "away_goals"),
            _xg(row.get("home_xg", ""), "home_xg"), _xg(row.get("away_xg", ""), "away_xg"), stats, odds,
        )

    team, opponent = club_name(row["team"]), club_name(row["opponent"])
    goals_for, goals_against = _count(row["gf"], "gf"), _count(row["ga"], "ga")
    xg_for, xg_against = _xg(row.get("xg", ""), "xg"), _xg(row.get("xga", ""), "xga")
    shots = _legacy_count(row.get("sh", ""), "sh")
    on_target = _legacy_count(row.get("sot", ""), "sot")
    metadata = {"kickoff": row.get("time") or None, "referee": row.get("referee") or None}
    if row["venue"] == "Home":
        stats = MatchStats(**metadata, home_shots=shots, home_shots_on_target=on_target)
        return Fixture(day, team, opponent, goals_for, goals_against, xg_for, xg_against, stats)
    if row["venue"] == "Away":
        stats = MatchStats(**metadata, away_shots=shots, away_shots_on_target=on_target)
        return Fixture(day, opponent, team, goals_against, goals_for, xg_against, xg_for, stats)
    raise ValueError("venue must be Home or Away.")


def _reconcile(home: Fixture, away: Fixture) -> Fixture:
    if (home.home_goals, home.away_goals) != (away.home_goals, away.away_goals):
        raise ValueError(f"{home.date} {home.home} vs {home.away} has conflicting scores.")
    stats = replace(
        home.stats, away_shots=away.stats.away_shots, away_shots_on_target=away.stats.away_shots_on_target,
    )
    return replace(
        home,
        home_xg=home.home_xg if home.home_xg is not None else away.home_xg,
        away_xg=away.away_xg if away.away_xg is not None else home.away_xg,
        stats=stats,
    )


def load_fixtures(path: str | Path = DEFAULT_HISTORY) -> tuple[Fixture, ...]:
    """Validate both sides of paired results and return chronological fixtures."""
    fixtures: dict[tuple[date, str, str], dict[str, Fixture]] = {}
    names: dict[str, str] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        paired = not FIXTURE_COLUMNS <= columns and TEAM_COLUMNS <= columns
        if not paired and not FIXTURE_COLUMNS <= columns:
            required = ", ".join(sorted(FIXTURE_COLUMNS))
            raise ValueError(f"Expected paired FBref results or fixture columns: {required}.")
        for raw in reader:
            try:
                if None in raw or None in raw.values():
                    raise ValueError("CSV row has a different number of fields than the header.")
                row = {key: value.strip() for key, value in raw.items()}
                if paired and (row.get("comp") or "Premier League") != "Premier League":
                    continue
                fixture = _row_fixture(row, paired)
                home = names.setdefault(fixture.home.casefold(), fixture.home)
                away = names.setdefault(fixture.away.casefold(), fixture.away)
                if home == away:
                    raise ValueError("A club cannot play itself.")
                fixture = replace(fixture, home=home, away=away)
                key = fixture.date, home, away
                side = row["venue"] if paired else "Home"
                sides = fixtures.setdefault(key, {})
                if side in sides:
                    raise ValueError(f"Duplicate {side} row for {home} vs {away}.")
                sides[side] = fixture
            except ValueError as error:
                raise ValueError(f"{path}, line {reader.line_num}: {error}") from error

    results = []
    for day, home, away in sorted(fixtures):
        sides = fixtures[day, home, away]
        if paired and len(sides) != 2:
            raise ValueError(f"{day} {home} vs {away} has no matching reciprocal result.")
        result = _reconcile(sides["Home"], sides["Away"]) if paired else sides["Home"]
        results.append(result)
    if not results:
        raise ValueError(f"No completed Premier League fixtures found in {path}.")
    return validate_fixtures(results)


def validate_fixtures(fixtures: list[Fixture]) -> tuple[Fixture, ...]:
    results = tuple(sorted(fixtures, key=lambda match: (match.date, match.home, match.away)))
    appearances = set()
    for match in results:
        day, home, away = match.date, match.home, match.away
        if home == away:
            raise ValueError(f"{day} {home} cannot play itself.")
        if (day, home) in appearances or (day, away) in appearances:
            raise ValueError(f"{day} {home} vs {away} schedules a club twice on one day.")
        appearances.update(((day, home), (day, away)))
    return results


def fixtures_csv(fixtures: tuple[Fixture, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([
        "date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg",
        *(field.name for field in fields(MatchStats)), *ODDS_COLUMNS,
    ])
    for match in fixtures:
        writer.writerow([
            match.date, match.home, match.away, match.home_goals, match.away_goals,
            match.home_xg, match.away_xg, *astuple(match.stats),
            *(astuple(match.market_odds) if match.market_odds is not None else (None, None, None)),
        ])
    return buffer.getvalue()
