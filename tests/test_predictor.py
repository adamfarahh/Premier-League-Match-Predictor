import csv
import io
import unittest
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from predictor.data import DEFAULT_HISTORY, PROJECT_DIR, MatchStats, fixtures_csv, load_fixtures
from predictor.evaluation import backtest, backtest_csv, compare_market, score_confident_picks
from predictor.football_data import parse_season
from predictor.market import MarketOdds
from predictor.model import LEGACY_MODEL, STRENGTH_MODEL, Predictor, training_rows


class HistoryTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_HISTORY.is_file(), "Optional locally downloaded results")
    def test_bundled_archive_covers_every_requested_calendar_year(self):
        fixtures = load_fixtures(DEFAULT_HISTORY)
        counts = Counter(match.date.year for match in fixtures)
        self.assertEqual(sorted(counts), list(range(2010, 2027)))
        self.assertGreaterEqual(len(fixtures), 6364)
        self.assertEqual(fixtures[0].date, date(2010, 1, 5))
        self.assertGreaterEqual(fixtures[-1].date, date(2026, 11, 8))
        self.assertTrue(all(match.stats.home_shots_on_target is not None for match in fixtures))

    @unittest.skipUnless((PROJECT_DIR / "matches.csv").is_file(), "Optional legacy CSV")
    def test_original_paired_rows_round_trip_as_fixtures(self):
        fixtures = load_fixtures(PROJECT_DIR / "matches.csv")
        self.assertEqual(len(fixtures), 760)
        self.assertIn("Nottingham Forest", {match.home for match in fixtures})
        self.assertTrue(all(match.stats.away_shots is not None for match in fixtures))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.csv"
            path.write_text(fixtures_csv(fixtures), encoding="utf-8", newline="")
            self.assertEqual(load_fixtures(path), fixtures)

    def test_supplier_rows_handle_date_versions_and_unplayed_games(self):
        content = (
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HST,AST,HxG,AxG\n"
            "E0,21/08/26,Man United,Nott'm Forest,1,1,D,0,4,0.25,1.34\n"
            "E0,02/01/2027,Man City,Wolves,2,0,H,6,0,,\n"
            "E0,03/01/2027,Tottenham,Newcastle,,,,,,,\n"
            ",,,,,,,,,,\n"
        ).encode()
        first, second = parse_season(content, 2026)
        self.assertEqual((first.home, first.away), ("Manchester United", "Nottingham Forest"))
        self.assertEqual(first.date, date(2026, 8, 21))
        self.assertEqual(second.date, date(2027, 1, 2))
        self.assertEqual(first.stats.home_shots_on_target, 0)
        self.assertEqual(first.home_xg, 0.25)
        self.assertIsNone(second.home_xg)
        with self.assertRaisesRegex(ValueError, "conflicts with the score"):
            parse_season(content.replace(b",1,1,D,", b",1,1,H,"), 2026)

    def test_supplier_odds_use_the_pre_closing_average_columns(self):
        content = (
            b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA\n"
            b"E0,21/08/26,Man United,Wolves,2,0,H,2.0,4.0,4.0,1.12,10,20\n"
            b"E0,22/08/26,Tottenham,Newcastle,0,0,D,,,,1.7,3.4,5.0\n"
        )
        first, second = parse_season(content, 2026)
        self.assertEqual(first.market_odds, MarketOdds(2.0, 4.0, 4.0))
        self.assertEqual(first.market_odds.probabilities, (0.50, 0.25, 0.25))
        self.assertIsNone(second.market_odds)
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            MarketOdds(2.0, float("nan"), 4.0)


@unittest.skipUnless(DEFAULT_HISTORY.is_file(), "Optional locally downloaded results")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        cls.predictor = Predictor(cls.fixtures)

    def test_replay_does_not_use_same_day_or_future_scores(self):
        day = date(2010, 8, 14)
        actual = next(match for match in self.fixtures if match.date == day)
        earlier = tuple(match for match in self.fixtures if match.date < day)
        changed = replace(
            actual, home_goals=20, away_goals=0,
            stats=replace(actual.stats, home_shots=30, home_shots_on_target=25),
            market_odds=MarketOdds(1.1, 10.0, 20.0),
        )
        original = self.predictor.forecast(day, actual.home, actual.away)
        counterfactual = Predictor((*earlier, changed)).forecast(day, actual.home, actual.away)
        self.assertEqual(original.probabilities, counterfactual.probabilities)
        self.assertEqual(original.home, counterfactual.home)
        self.assertEqual(original.away, counterfactual.away)
        self.assertEqual(counterfactual.actual, changed)
        self.assertAlmostEqual(sum(original.probabilities), 1.0)
        self.assertLess(original.trained_through, day)
        self.assertTrue(all(match.date < day for club in (original.home, original.away) for match in club.recent))

    def test_warmup_and_cold_start(self):
        with self.assertRaisesRegex(ValueError, "60 earlier fixtures"):
            self.predictor.forecast(date(2010, 1, 5), "Arsenal", "Liverpool")
        forecast = self.predictor.forecast(date(2026, 11, 18), "Example Club", "Man City")
        self.assertEqual(forecast.home.matches_played, 0)
        self.assertEqual(forecast.home.rating, 1500)
        self.assertEqual(forecast.away.name, "Manchester City")
        self.assertIsNone(forecast.actual)

    def test_backtest_refits_at_new_month_and_exports_each_fixture_once(self):
        start, end = date(2026, 9, 15), date(2026, 11, 1)
        report = backtest(self.predictor, start, end)
        cutoffs = {}
        for row in report.predictions:
            self.assertLess(row.trained_through, row.fixture.date)
            self.assertAlmostEqual(sum(row.probabilities), 1)
            self.assertAlmostEqual(sum(row.baseline), 1)
            cutoff = cutoffs.setdefault(row.fixture.date.month, row.trained_through)
            self.assertEqual(row.trained_through, cutoff)
        self.assertGreaterEqual(len(cutoffs), 2)
        self.assertEqual(list(cutoffs.values()), sorted(set(cutoffs.values())))
        self.assertEqual(sum(map(sum, report.confusion)), len(report.predictions))
        rows = list(csv.DictReader(io.StringIO(backtest_csv(report))))
        keys = {(row["date"], row["home_team"], row["away_team"]) for row in rows}
        self.assertEqual(len(keys), len(rows))
        self.assertEqual(len(rows), len(report.predictions))

    def test_backtest_defaults_to_all_fixtures_after_warmup(self):
        fixtures = self.fixtures[:220]
        predictor = Predictor(fixtures)
        report = backtest(predictor)
        expected = tuple(match for match in fixtures if match.date >= predictor.first_forecast)
        self.assertEqual(tuple(row.fixture for row in report.predictions), expected)
        self.assertEqual(backtest(predictor, since=fixtures[0].date), report)

    def test_conflicting_shot_pair_affects_neither_shot_rating(self):
        match = self.fixtures[0]
        stats = MatchStats(home_shots=8, home_shots_on_target=9, away_shots=12, away_shots_on_target=5)
        inconsistent = replace(match, stats=stats)
        missing = replace(match, stats=replace(stats, home_shots=None, home_shots_on_target=None))
        followup = replace(match, date=date(2010, 1, 6))
        self.assertTrue(stats.inconsistent_shots)
        self.assertEqual(stats.training_shots(), ((None, None), (12, 5)))
        self.assertEqual(
            training_rows((inconsistent, followup))[-1].features,
            training_rows((missing, followup))[-1].features,
        )

    def test_legacy_comparison_covers_the_same_fixtures_and_cutoffs(self):
        start, end = date(2025, 8, 1), date(2025, 9, 30)
        model = backtest(Predictor(self.fixtures, model=STRENGTH_MODEL), start, end)
        previous = backtest(Predictor(self.fixtures, model=LEGACY_MODEL), start, end)
        self.assertEqual(len(model.predictions), len(previous.predictions))
        for current, old in zip(model.predictions, previous.predictions, strict=True):
            self.assertEqual(current.fixture, old.fixture)
            self.assertEqual(current.trained_through, old.trained_through)
            self.assertEqual(current.baseline, old.baseline)
        self.assertNotEqual(model.predictions[0].probabilities, previous.predictions[0].probabilities)

    def test_odds_and_confidence_scores_keep_matched_fixtures_and_preserve_missing_odds(self):
        report = backtest(self.predictor, date(2025, 8, 15), date(2025, 8, 16))
        priced = replace(report.predictions[0], fixture=replace(
            report.predictions[0].fixture, market_odds=MarketOdds(2.0, 4.0, 4.0),
        ))
        unpriced = replace(report.predictions[1], fixture=replace(
            report.predictions[1].fixture, market_odds=None,
        ))
        mixed = replace(report, predictions=(priced, unpriced))
        market = compare_market(mixed)
        self.assertEqual(market.matches, 1)
        self.assertEqual(market.model_scores.accuracy, priced.prediction == priced.fixture.outcome)
        self.assertEqual(score_confident_picks(mixed, 0.50, market=True).matches, 1)
        self.assertIsNone(score_confident_picks(mixed, 0.51, market=True))
        csv_rows = list(csv.DictReader(io.StringIO(backtest_csv(mixed, include_market=True))))
        self.assertEqual(csv_rows[0]["market_p_home"], "0.50000000")
        self.assertEqual(csv_rows[1]["market_p_home"], "")


if __name__ == "__main__":
    unittest.main()
