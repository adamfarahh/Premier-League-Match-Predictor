"""Import completed Premier League results from Football-Data.co.uk season files."""

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.error import URLError

from .data import OUTCOMES, Fixture, MatchStats, _count, _xg, club_name, validate_fixtures
from .market import MarketOdds
from .network import read_url


START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 12, 31)
CSV_URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"
REQUIRED = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
STAT_COLUMNS = {
    "home_ht_goals": "HTHG", "away_ht_goals": "HTAG",
    "home_shots": "HS", "away_shots": "AS",
    "home_shots_on_target": "HST", "away_shots_on_target": "AST",
    "home_corners": "HC", "away_corners": "AC",
    "home_fouls": "HF", "away_fouls": "AF",
    "home_yellows": "HY", "away_yellows": "AY",
    "home_reds": "HR", "away_reds": "AR",
}
AVERAGE_ODDS_PREFIXES = ("Avg", "BbAv")


def _season_code(season: int) -> str:
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _season_date(value: str, season: int) -> date:
    day, month, year = value.split("/")
    if len(year) == 2:
        if int(year) == season % 100:
            year = str(season)
        elif int(year) == (season + 1) % 100:
            year = str(season + 1)
    match_date = date(int(year), int(month), int(day))
    if match_date.year not in (season, season + 1):
        raise ValueError(f"Unexpected match date {value!r} for the {season}–{season + 1} season.")
    return match_date


def _market_odds(row: dict[str, str]) -> MarketOdds | None:
    for prefix in AVERAGE_ODDS_PREFIXES:
        try:
            values = [float(row.get(prefix + outcome, "")) for outcome in OUTCOMES]
            return MarketOdds(*values)
        except ValueError:
            continue
    return None


def parse_season(content: bytes, season: int) -> list[Fixture]:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = content.decode("cp1252")
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if not REQUIRED <= set(reader.fieldnames or ()):
        raise ValueError(f"Season {_season_code(season)} is missing required Football-Data columns.")

    fixtures = []
    for raw in reader:
        row = {key: (raw.get(key) or "").strip() for key in reader.fieldnames}
        if not any(row.values()):
            continue
        try:
            if row["Div"] != "E0":
                raise ValueError("Expected a Premier League (E0) row.")
            if not any(row[key] for key in ("FTHG", "FTAG", "FTR")):
                continue
            stats = MatchStats(
                kickoff=row.get("Time") or None,
                referee=row.get("Referee") or None,
                **{key: _count(row[column], column) for key, column in STAT_COLUMNS.items() if row.get(column)},
            )
            fixture = Fixture(
                _season_date(row["Date"], season), club_name(row["HomeTeam"]), club_name(row["AwayTeam"]),
                _count(row["FTHG"], "FTHG"), _count(row["FTAG"], "FTAG"),
                _xg(row.get("HxG", ""), "HxG"), _xg(row.get("AxG", ""), "AxG"), stats, _market_odds(row),
            )
            if fixture.outcome != row["FTR"]:
                raise ValueError(f"Full-time result {row['FTR']!r} conflicts with the score.")
            fixtures.append(fixture)
        except ValueError as error:
            raise ValueError(f"Season {_season_code(season)}, line {reader.line_num}: {error}") from error
    if not fixtures:
        raise ValueError(f"Season {_season_code(season)} has no completed matches.")
    return fixtures


def _read_season(season: int, from_dir: Path | None) -> list[Fixture]:
    code = _season_code(season)
    if from_dir is not None:
        return parse_season((from_dir / f"{code}.csv").read_bytes(), season)

    url = CSV_URL.format(code=code)
    for attempt in range(2):
        try:
            return parse_season(read_url(url), season)
        except (TimeoutError, URLError) as error:
            if attempt:
                raise OSError(f"Could not download {url}: {error}") from error
    raise AssertionError("Download retry loop finished without a result.")


def import_history(start: date, end: date, from_dir: Path | None = None) -> tuple[Fixture, ...]:
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if start.year < 1996 or end.year > 2099:
        raise ValueError("This importer supports dates from 1996 through 2099.")
    # Include the prior season for July, when the delayed 2019/20 season finished.
    first_season = start.year - (start.month < 8)
    last_season = end.year - (end.month < 8)
    seasons = range(first_season, last_season + 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(lambda season: _read_season(season, from_dir), seasons))
    for season, fixtures in zip(seasons, files, strict=True):
        if season < last_season and len(fixtures) != 380:
            raise ValueError(f"Season {_season_code(season)} has {len(fixtures)} results instead of 380.")
    results = validate_fixtures([match for file in files for match in file if start <= match.date <= end])
    if not results:
        raise ValueError("No completed matches fall in the requested date range.")
    return results
