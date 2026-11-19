"""Command-line entry point for preparation, forecasting, and evaluation."""

import argparse
import sqlite3
import time
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from .data import DEFAULT_HISTORY, OUTCOMES, fixtures_csv, load_fixtures, parse_date, validate_fixtures
from .football_data import END_DATE, START_DATE, import_history
from .market import MarketOdds
from .model import DEFAULT_MODEL, LEGACY_MODEL, MODEL_CHOICES, MODEL_LABELS, STRENGTH_MODEL, Predictor

if TYPE_CHECKING:
    from .evaluation import Scores


def _save(output: Path, content: str) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(content)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved {output}")


def _write(output: Path, content: str, source: Path) -> None:
    if output.resolve() == source.resolve():
        raise ValueError("Choose an output path different from the input history.")
    _save(output, content)


def _print_scores(comparison: list[tuple[str, "Scores"]]) -> None:
    print(f"{'Method':<24} {'Log loss':>9} {'Brier':>9} {'Accuracy':>9}")
    for label, metrics in comparison:
        print(f"{label:<24} {metrics.log_loss:>9.3f} {metrics.brier:>9.3f} {metrics.accuracy:>9.1%}")


def _print_probabilities(probabilities: tuple[float, float, float]) -> None:
    for label, probability in zip(("Home win", "Draw", "Away win"), probabilities, strict=True):
        print(f"{label:<8} {probability:6.1%}")


def _confidence(value: str) -> float:
    threshold = float(value)
    if not 0 < threshold <= 1:
        raise argparse.ArgumentTypeError("Enter a confidence between 0 and 1, such as 0.65.")
    return threshold


def _latest_archive(path: Path, season: int, matches, *, offline: bool):
    from .season import LONDON, official_history, utc_now

    archive = load_fixtures(path)
    if offline or not any(match.finished for match in matches):
        return archive
    start = date(season, 7, 1)
    end = date(season + 1, 7, 1)
    try:
        current = import_history(start, utc_now().astimezone(LONDON).date())
    except OSError as error:
        print(f"Market archive unavailable. Using locally stored odds. {error}", flush=True)
        return archive
    new_pairs = {(match.home, match.away) for match in current}
    refreshed = validate_fixtures([
        *(match for match in archive
          if not (start <= match.date < end and (match.home, match.away) in new_pairs)),
        *current,
    ])
    official_history(season, matches, refreshed, utc_now())
    if refreshed != archive:
        _save(path, fixtures_csv(refreshed))
    return refreshed


def _run_scoreboard(args: argparse.Namespace) -> None:
    from .git_log import commit_forecast_log
    from .scoreboard import read_scoreboard, scoreboard_csv, sync_scoreboard
    from .season import official_schedule

    if args.interval <= 0:
        raise ValueError("The sync interval must be positive.")
    if args.gameweeks is not None and (args.gameweeks < 1 or not (args.sync or args.watch)):
        raise ValueError("Use a positive --gameweeks count with --sync or --watch.")
    if args.output and args.output.resolve() == args.data.resolve():
        raise ValueError("Choose a scoreboard export path different from the input history.")
    reserved = {path.resolve() for path in (args.data, args.db, args.output) if path is not None}
    if args.git_log and args.git_log.resolve() in reserved:
        raise ValueError("Choose a separate path for the Git forecast log.")
    while True:
        try:
            if args.sync or args.watch:
                matches = official_schedule(args.season, args.feed_dir)
                archive = _latest_archive(
                    args.data, args.season, matches, offline=args.feed_dir is not None,
                )
                issued = sync_scoreboard(
                    args.db, args.season, matches, archive, model=args.model, gameweeks=args.gameweeks,
                )
                print(f"Issued {issued} new pre-kickoff forecasts.", flush=True)
            report = read_scoreboard(args.db, args.season)
            print(f"{args.season}/{str(args.season + 1)[-2:]} scoreboard | Last updated {report.updated_at or 'never'}")
            print(" GW   FT  Logged Scored Paired    Model   Paired   Market")
            for week in report.gameweeks:
                if not (week.completed or week.forecasted):
                    continue
                loss = " ".join(f"{score:>8.3f}" if score is not None else f"{'—':>8}" for score in (
                    week.model_log_loss, week.paired_model_log_loss, week.market_log_loss,
                ))
                print(f"{week.gameweek:>3} {week.completed:>4} {week.forecasted:>7} "
                      f"{week.scored:>6} {week.paired:>6} {loss}")
            if args.output:
                _write(args.output, scoreboard_csv(report), args.db)
            if args.git_log:
                commit = commit_forecast_log(args.db, args.git_log)
                if commit:
                    print(f"Committed forecast log to Git as {commit}.", flush=True)
        except (OSError, ValueError, sqlite3.Error) as error:
            if not args.watch:
                raise
            print(f"Scoreboard sync failed. Retrying in {args.interval} seconds. {error}", flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Premier League match forecasting")
    parser.add_argument("--data", type=Path, default=DEFAULT_HISTORY, help="input match history CSV")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL, help="results model variant")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate and export one row per fixture")
    prepare.add_argument("--output", type=Path, required=True)
    commands.add_parser("audit", help="check fixture history and report missing or inconsistent stats")

    refresh = commands.add_parser("refresh", help="import completed matches from Football-Data.co.uk")
    refresh.add_argument("--start", type=parse_date, default=START_DATE)
    refresh.add_argument("--end", type=parse_date, default=END_DATE)
    refresh.add_argument("--output", type=Path, default=DEFAULT_HISTORY)
    refresh.add_argument("--from-dir", type=Path, help="read season CSVs such as 2526.csv from this folder")

    predict = commands.add_parser("predict", help="forecast a fixture using results before its date")
    predict.add_argument("--home", required=True)
    predict.add_argument("--away", required=True)
    predict.add_argument("--date", required=True, type=parse_date)
    predict.add_argument("--odds", type=float, nargs=3, metavar=("HOME", "DRAW", "AWAY"),
                         help="also show probabilities from pre-match decimal odds")

    evaluate = commands.add_parser("backtest", help="evaluate all eligible fixtures in date order")
    evaluate.add_argument("--since", type=parse_date, help="start date (default: first date after model warm-up)")
    evaluate.add_argument("--until", type=parse_date, help="end date (default: latest result)")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--compare-legacy", action="store_true", help="also score the previous model")
    evaluate.add_argument("--compare-models", action="store_true", help="also score the other feature set")
    evaluate.add_argument("--compare-market", action="store_true", help="compare published pre-match odds")
    evaluate.add_argument("--min-confidence", type=_confidence, metavar="PROBABILITY",
                          help="report accuracy and coverage for picks at or above this probability")

    serve = commands.add_parser("serve", help="start the local dashboard")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--db", type=Path, default=Path("scoreboard.sqlite3"))
    serve.add_argument("--season", type=int, default=2026)
    serve.add_argument("--schedule", type=Path, default=Path("fixtures_2026_27.csv"))

    season = commands.add_parser("season", help="generate dated forecasts for a full season")
    season.add_argument("--schedule", type=Path, default=Path("fixtures_2026_27.csv"))
    season.add_argument("--live", type=int, metavar="START_YEAR", help="refresh using the official current-season feed")
    season.add_argument("--output", type=Path, required=True)

    scoreboard = commands.add_parser("scoreboard", help="show or update the prospective gameweek scoreboard")
    scoreboard.add_argument("--season", type=int, default=2026)
    scoreboard.add_argument("--db", type=Path, default=Path("scoreboard.sqlite3"))
    scoreboard.add_argument("--sync", action="store_true", help="record upcoming forecasts and settle completed games")
    scoreboard.add_argument("--watch", action="store_true", help="sync repeatedly until stopped")
    scoreboard.add_argument("--gameweeks", type=int, metavar="COUNT",
                            help="log the next COUNT gameweeks instead of just the next 72 hours")
    scoreboard.add_argument("--interval", type=int, default=3600, help="seconds between updates in watch mode")
    scoreboard.add_argument("--feed-dir", type=Path, help="use locally saved public FPL JSON instead of downloading")
    scoreboard.add_argument("--output", type=Path)
    scoreboard.add_argument("--git-log", type=Path, help="append issued forecasts to a CSV and commit it locally")
    commands.add_parser("select-model", help="compare development log loss with a simpler-model tie rule")

    args = parser.parse_args()
    try:
        if args.command == "scoreboard":
            _run_scoreboard(args)
            return
        if args.command == "refresh":
            fixtures = import_history(args.start, args.end, args.from_dir)
            _save(args.output, fixtures_csv(fixtures))
            print(f"Imported {len(fixtures):,} fixtures from {fixtures[0].date} through {fixtures[-1].date}.")
            return

        fixtures = load_fixtures(args.data)
        if args.command == "audit":
            print(f"{len(fixtures):,} unique fixtures from {fixtures[0].date} through {fixtures[-1].date}.")
            inconsistent = sum(match.stats.inconsistent_shots for match in fixtures)
            incomplete = sum(
                any(target is None for _, target in match.stats.training_shots()) for match in fixtures
            )
            xg = sum(match.home_xg is not None and match.away_xg is not None for match in fixtures)
            print(f"Fixtures with inconsistent shot counts: {inconsistent:,}")
            print(f"Fixtures missing usable shots on target: {incomplete:,}")
            print(f"Fixtures with expected goals for both clubs: {xg:,}")
            print(f"Fixtures with pre-match market odds: {sum(match.market_odds is not None for match in fixtures):,}")
            return

        if args.command == "prepare":
            _write(args.output, fixtures_csv(fixtures), args.data)
            print(f"Validated {len(fixtures)} fixtures from {fixtures[0].date} through {fixtures[-1].date}.")
            return

        if args.command == "select-model":
            from .evaluation import select_model

            selection = select_model(fixtures)
            _print_scores([(MODEL_LABELS[LEGACY_MODEL], selection.simple),
                           (MODEL_LABELS[STRENGTH_MODEL], selection.adjusted)])
            print(f"Paired log-loss gain {selection.adjusted_gain:.6f} across {selection.fixtures:,} fixtures.")
            print(f"Standard error across {selection.seasons} seasons {selection.season_standard_error:.6f}.")
            print(f"Selected {MODEL_LABELS[selection.selected]}.")
            return

        if args.command == "season":
            from .season import ScheduledMatch, forecast_season, load_schedule, official_history, official_schedule, season_csv, utc_now

            if args.live is not None:
                matches = official_schedule(args.live)
                fixtures = _latest_archive(args.data, args.live, matches, offline=False)
                _, fixtures = official_history(args.live, matches, fixtures, utc_now())
                schedule = tuple(ScheduledMatch(match.day, match.home, match.away) for match in matches if match.day)
                if len(schedule) != len(matches):
                    raise ValueError("Some official fixtures have no scheduled date. Use the scoreboard for dated forecasts.")
            else:
                if args.output.resolve() == args.schedule.resolve():
                    raise ValueError("Choose an export path different from the schedule.")
                schedule = load_schedule(args.schedule)
            predictions = forecast_season(Predictor(fixtures, model=args.model), schedule, as_of=utc_now())
            _write(args.output, season_csv(predictions), args.data)
            print(f"Generated {len(predictions)} fixtures at {predictions[0].generated_at.isoformat()}.")
            return

        predictor = Predictor(fixtures, model=args.model)
        if args.command == "predict":
            odds = MarketOdds(*args.odds) if args.odds is not None else None
            forecast = predictor.forecast(args.date, args.home, args.away)
            print(f"{forecast.date} | {forecast.home.name} vs {forecast.away.name}")
            if odds is not None:
                print("Results model")
            _print_probabilities(forecast.probabilities)
            print(f"Trained on {forecast.training_matches} fixtures through {forecast.trained_through}.")
            for club in (forecast.home, forecast.away):
                if not club.matches_played:
                    print(f"{club.name} has no earlier results. The forecast uses a neutral starting rating.")
            if forecast.actual:
                print(f"Actual score: {forecast.actual.home_goals}-{forecast.actual.away_goals}")
            if odds is not None:
                print("\nPre-match odds")
                _print_probabilities(odds.probabilities)
        elif args.command == "backtest":
            from .evaluation import backtest, backtest_csv, compare_market, score_confident_picks

            report = backtest(predictor, args.since, args.until)
            first, last = report.predictions[0].fixture.date, report.predictions[-1].fixture.date
            print(f"Archive: {len(fixtures):,} fixtures from {fixtures[0].date} through {fixtures[-1].date}.")
            print(f"{len(report.predictions):,} fixtures evaluated from {first} through {last}.")
            comparison = [("Model", report.model_scores)]
            if args.compare_legacy or args.compare_models:
                alternative = STRENGTH_MODEL if args.model == LEGACY_MODEL else LEGACY_MODEL
                previous = backtest(Predictor(fixtures, model=alternative), report.since, report.until)
                comparison.append((MODEL_LABELS[alternative], previous.model_scores))
            comparison.append(("League frequencies", report.baseline_scores))
            _print_scores(comparison)
            print("Confusion matrix (rows actual, columns predicted):")
            print("        " + " ".join(f"{label:>5}" for label in OUTCOMES))
            for label, row in zip(OUTCOMES, report.confusion, strict=True):
                print(f"      {label} " + " ".join(f"{count:>5}" for count in row))
            if args.compare_market:
                market = compare_market(report)
                if market is None:
                    print("\nNo pre-match market odds available in this date range.")
                else:
                    print(f"\nPre-match comparison on {market.matches:,} of {len(report.predictions):,} fixtures.")
                    _print_scores([("Model", market.model_scores), ("Market odds", market.market_scores)])
            if args.min_confidence is not None:
                print(f"\nPicks with at least {args.min_confidence:.0%} estimated probability")
                methods = [("Model", False)]
                if args.compare_market:
                    methods.append(("Market odds", True))
                for label, use_market in methods:
                    picks = score_confident_picks(report, args.min_confidence, market=use_market)
                    total = len(report.predictions)
                    if picks is None:
                        print(f"{label}: 0 of {total:,} fixtures met the threshold.")
                    else:
                        coverage = picks.matches / total
                        print(
                            f"{label}: {picks.scores.accuracy:.1%} accuracy on "
                            f"{picks.matches:,} of {total:,} fixtures ({coverage:.1%} coverage)."
                        )
            if args.output:
                _write(args.output, backtest_csv(report, include_market=args.compare_market), args.data)
        else:
            from .server import DashboardServer

            if not 0 <= args.port <= 65535:
                raise ValueError("Port must be between 0 and 65535.")
            with DashboardServer(
                (args.host, args.port), predictor, history_path=args.data, scoreboard_path=args.db,
                season=args.season, schedule_path=args.schedule,
            ) as server:
                print(f"Premier League API at http://{args.host}:{server.server_port}", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
    except KeyboardInterrupt:
        pass
    except (OSError, UnicodeError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
