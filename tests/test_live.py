import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.request import urlopen

from predictor.data import Fixture, MatchStats, fixtures_csv
from predictor.evaluation import backtest, calibration, log_loss, select_model
from predictor.market import MarketOdds
from predictor.model import LEGACY_MODEL, MODEL_CHOICES, LeagueHistory, OutcomeModel, Predictor
from predictor.scoreboard import logged_forecasts, read_scoreboard, sync_scoreboard
from predictor.season import OfficialMatch, ScheduledMatch, forecast_season
from predictor.server import DashboardServer


UTC = timezone.utc


def history() -> tuple[Fixture, ...]:
    scores = ((2, 1), (0, 0), (1, 3))
    return tuple(
        Fixture(date(2025, 8, 1) + timedelta(days=index), "Arsenal", "Chelsea", *scores[index % 3])
        for index in range(75)
    )


class LeakageTests(unittest.TestCase):
    def test_prediction_is_invariant_to_same_day_and_later_results_in_both_models(self):
        prior = history()
        day = date(2025, 11, 1)
        changed = (
            Fixture(day, "Arsenal", "Chelsea", 8, 0, stats=MatchStats(home_shots=35, home_shots_on_target=29)),
            Fixture(day, "Fulham", "Liverpool", 4, 5),
            Fixture(day + timedelta(days=1), "Arsenal", "Chelsea", 0, 10, market_odds=MarketOdds(1.1, 9, 25)),
        )
        for name in MODEL_CHOICES:
            with self.subTest(model=name):
                original = Predictor(prior, model=name).forecast(day, "Arsenal", "Chelsea")
                predictor = Predictor(prior + changed, model=name)
                forecast = predictor.forecast(day, "Arsenal", "Chelsea")
                self.assertEqual(forecast.probabilities, original.probabilities)
                self.assertEqual(forecast.training_matches, len(prior))
                self.assertEqual(forecast.actual, changed[0])
                self.assertEqual(
                    predictor.rows[len(prior)].features,
                    predictor.snapshot(day).history.features(day, "Arsenal", "Chelsea"),
                )
                with self.assertRaisesRegex(ValueError, "fit cutoff"):
                    OutcomeModel(predictor.rows, day, name)

    def test_feature_read_cannot_use_recorded_same_day_results(self):
        state = LeagueHistory()
        match = history()[0]
        self.assertEqual(state.features(match.date, match.home, match.away), state.features(match.date, match.home, match.away))
        self.assertEqual(len(state.teams), 0)
        state.record(match)
        with self.assertRaisesRegex(ValueError, "strictly before"):
            state.features(match.date, match.home, match.away)

    def test_calibration_counts_every_forecast_once_per_outcome(self):
        report = backtest(Predictor(history()))
        bins = calibration(report)
        for outcome in ("H", "D", "A"):
            self.assertEqual(sum(point.matches for point in bins if point.outcome == outcome), len(report.predictions))
        self.assertTrue(all(point.lower <= point.mean_probability <= point.upper for point in bins))
        self.assertEqual(calibration(report, market=True), ())

    def test_selection_uses_the_simpler_candidate_under_exact_tie(self):
        from unittest.mock import patch

        prior = history()
        initial = backtest(Predictor(prior)).predictions[0]
        matching = replace(initial, fixture=replace(initial.fixture, date=date(2026, 8, 23)))
        report = replace(backtest(Predictor(prior)), predictions=(initial, matching))
        with patch("predictor.evaluation.backtest", return_value=report):
            selection = select_model(prior, end=date(2026, 12, 31))
        self.assertEqual(selection.selected, LEGACY_MODEL)
        self.assertEqual(selection.seasons, 2)


class LiveTests(unittest.TestCase):
    def test_forecast_that_finishes_after_kickoff_is_never_logged(self):
        now = datetime(2026, 8, 12, 12, tzinfo=UTC)
        kickoff = now + timedelta(seconds=1)
        match = OfficialMatch(1, 1, kickoff, "Arsenal", "Chelsea", False, None)
        clock_values = iter([now, kickoff + timedelta(seconds=1), kickoff + timedelta(seconds=2)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.sqlite3"
            self.assertEqual(sync_scoreboard(path, 2026, (match,), history(), clock=lambda: next(clock_values)), 0)
            self.assertEqual(logged_forecasts(path, 2026), [])

    def test_scoreboard_preserves_forecasts_and_scores_on_matched_odds_only(self):
        now = datetime(2026, 8, 12, 12, tzinfo=UTC)
        past = OfficialMatch(1, 1, now - timedelta(days=1), "Fulham", "Everton", True, (2, 0))
        upcoming = OfficialMatch(2, 2, now + timedelta(days=1), "Arsenal", "Chelsea", False, None)
        unpriced = OfficialMatch(3, 2, now + timedelta(days=1), "Fulham", "Everton", False, None)
        later = OfficialMatch(4, 30, now + timedelta(days=30), "Everton", "Fulham", False, None)
        archive = history()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.sqlite3"
            self.assertEqual(sync_scoreboard(path, 2026, (past, upcoming, unpriced, later), archive, clock=lambda: now), 2)
            first = read_scoreboard(path, 2026).gameweeks
            self.assertEqual((first[0].completed, first[0].forecasted), (1, 0))
            self.assertEqual((first[1].forecasted, first[1].scored, first[2].forecasted), (2, 0, 0))
            records = logged_forecasts(path, 2026)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["eligible"] and record["issued_at"] < record["current_kickoff"] for record in records))
            with closing(sqlite3.connect(path)) as db:
                ledger = db.execute("SELECT * FROM forecasts ORDER BY fixture_id").fetchall()
            self.assertEqual(sync_scoreboard(
                path, 2026, (past, upcoming, unpriced, later), archive,
                clock=lambda: now + timedelta(hours=2),
            ), 0)

            completed = replace(upcoming, finished=True, score=(2, 0))
            second_result = replace(unpriced, finished=True, score=(0, 0))
            odds = MarketOdds(2, 4, 4)
            result = Fixture(completed.day, completed.home, completed.away, 2, 0, market_odds=odds)
            self.assertEqual(sync_scoreboard(path, 2026, (past, completed, second_result, later), archive + (result,),
                                             clock=lambda: now + timedelta(days=2)), 0)
            week = read_scoreboard(path, 2026).gameweeks[1]
            self.assertEqual((week.completed, week.scored, week.paired), (2, 2, 1))
            self.assertAlmostEqual(week.market_log_loss, log_loss(odds.probabilities, "H"))
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT * FROM forecasts ORDER BY fixture_id").fetchall(), ledger)

    def test_season_forecasts_refresh_and_show_snapshot_timestamp(self):
        prior = history()
        first = ScheduledMatch(date(2026, 8, 13), "Arsenal", "Chelsea")
        next_match = ScheduledMatch(date(2026, 8, 15), "Chelsea", "Arsenal")
        now = datetime(2026, 8, 12, 12, tzinfo=UTC)
        initial = forecast_season(Predictor(prior), (first, next_match), as_of=now)
        self.assertEqual([row.generated_at for row in initial], [now, now])
        actual = Fixture(first.date, first.home, first.away, 4, 0)
        later = now + timedelta(days=2)
        refreshed = forecast_season(Predictor(prior + (actual,)), (first, next_match), as_of=later)
        self.assertEqual(refreshed[0].actual, actual)
        self.assertEqual(refreshed[1].trained_through, actual.date)
        self.assertEqual(refreshed[1].generated_at, later)
        self.assertNotEqual(initial[1].probabilities, refreshed[1].probabilities)

    def test_api_reloads_archive_and_serves_pre_match_probabilities(self):
        prior = history()
        with TemporaryDirectory() as folder:
            archive = Path(folder) / "results.csv"
            archive.write_text(fixtures_csv(prior), encoding="utf-8", newline="")
            with DashboardServer(("127.0.0.1", 0), Predictor(prior), history_path=archive,
                                 scoreboard_path=Path(folder) / "scoreboard.sqlite3") as server:
                worker = Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    address = f"http://127.0.0.1:{server.server_port}"
                    def get(path):
                        with urlopen(address + path, timeout=5) as response:
                            return json.load(response)

                    url = "/api/forecast?date=2025-11-01&home=Arsenal&away=Chelsea"
                    original = get(url)
                    actual = Fixture(date(2025, 11, 1), "Arsenal", "Chelsea", 10, 0)
                    archive.write_text(fixtures_csv(prior + (actual,)), encoding="utf-8", newline="")
                    forecast = get(url)
                    self.assertEqual(forecast["probabilities"], original["probabilities"])
                    self.assertEqual(forecast["actual"]["home_goals"], 10)
                    self.assertEqual(get("/api/history")["matches"], len(prior) + 1)
                    self.assertIsNone(get("/api/scoreboard")["updated_at"])
                finally:
                    server.shutdown()
                    worker.join()
