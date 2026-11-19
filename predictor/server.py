"""Serve the local dashboard and JSON API using Python's HTTP server."""

import json
import sqlite3
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from .data import load_fixtures, parse_date
from .evaluation import backtest, backtest_csv, default_backtest_start
from .model import MODEL_LABELS, MODEL_REVISION, Predictor
from .scoreboard import DEFAULT_SCOREBOARD, logged_forecasts, read_scoreboard, scoreboard_csv
from .season import forecast_season, load_schedule, utc_now


WEB_DIR = Path(__file__).with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], predictor: Predictor, *, history_path: Path | None = None,
        scoreboard_path: Path = DEFAULT_SCOREBOARD, season: int = 2026,
        schedule_path: Path = Path("fixtures_2026_27.csv"),
    ) -> None:
        self.predictor = predictor
        self.history_path = history_path
        self.history_version = self._history_version()
        self.history_lock = Lock()
        self.scoreboard_path = scoreboard_path
        self.season = season
        self.schedule_path = schedule_path
        self.report = lru_cache(maxsize=8)(backtest)
        super().__init__(address, DashboardHandler)

    def _history_version(self) -> tuple[int, int] | None:
        if self.history_path is None:
            return None
        status = self.history_path.stat()
        return status.st_mtime_ns, status.st_size

    def current_predictor(self) -> Predictor:
        with self.history_lock:
            version = self._history_version()
            if version != self.history_version:
                self.predictor = Predictor(load_fixtures(self.history_path), model=self.predictor.model)
                self.history_version = version
                self.report.cache_clear()
            return self.predictor


def _default_fixture(predictor: Predictor) -> dict:
    latest = predictor.fixtures[-1]
    return {"date": latest.date, "home": latest.home, "away": latest.away}


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def _send(self, status: int, content_type: str, body: bytes, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, content: dict, status: int = 200) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(content, default=lambda value: value.isoformat()).encode())

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, "image/x-icon", b"")
            return
        if url.path in ASSETS:
            filename, content_type = ASSETS[url.path]
            self._send(200, content_type, (WEB_DIR / filename).read_bytes())
            return

        try:
            params = parse_qs(url.query, keep_blank_values=True, max_num_fields=12)
            self._api(url.path, {key: values[0] for key, values in params.items()})
        except (KeyError, ValueError) as error:
            message = f"Missing parameter {error.args[0]}." if isinstance(error, KeyError) else str(error)
            self._json({"error": message}, HTTPStatus.BAD_REQUEST)
        except (OSError, sqlite3.Error) as error:
            self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

    def _api(self, path: str, params: dict[str, str]) -> None:
        if path == "/api/scoreboard/forecasts":
            season = int(params.get("season", self.server.season))
            self._json({"season": season, "forecasts": logged_forecasts(self.server.scoreboard_path, season)})
            return
        if path in {"/api/scoreboard", "/api/scoreboard.csv"}:
            report = read_scoreboard(self.server.scoreboard_path, int(params.get("season", self.server.season)))
            if path.endswith(".csv"):
                self._send(200, "text/csv; charset=utf-8", scoreboard_csv(report).encode(), "scoreboard.csv")
            else:
                self._json(asdict(report))
            return

        predictor = self.server.current_predictor()
        if path == "/api/history":
            self._json({
                "first_date": predictor.fixtures[0].date,
                "last_date": predictor.fixtures[-1].date,
                "matches": len(predictor.fixtures),
                "model": MODEL_LABELS[predictor.model],
                "model_revision": MODEL_REVISION,
                "clubs": predictor.clubs,
                "years": sorted({fixture.date.year for fixture in predictor.fixtures}, reverse=True),
                "first_forecast": predictor.first_forecast,
                "example": _default_fixture(predictor),
                "backtest_start": default_backtest_start(predictor),
            })
        elif path == "/api/forecast":
            forecast = predictor.forecast(parse_date(params["date"]), params["home"], params["away"])
            payload = asdict(forecast)
            payload["prediction"] = forecast.prediction
            payload["model"] = MODEL_LABELS[predictor.model]
            payload["model_revision"] = MODEL_REVISION
            for side, club in (("home", forecast.home), ("away", forecast.away)):
                for recent, match in zip(payload[side]["recent"], club.recent, strict=True):
                    recent["result"] = match.result
            self._json(payload)
        elif path == "/api/results":
            year = int(params.get("year", predictor.fixtures[-1].date.year))
            club = params.get("club")
            self._json({
                "fixtures": [
                    asdict(match) for match in reversed(predictor.fixtures)
                    if match.date.year == year and (not club or club in (match.home, match.away))
                ],
            })
        elif path in {"/api/backtest", "/api/backtest.csv"}:
            since = parse_date(params["since"]) if params.get("since") else default_backtest_start(predictor)
            until = parse_date(params["until"]) if params.get("until") else predictor.fixtures[-1].date
            report = self.server.report(predictor, since, until)
            if path.endswith(".csv"):
                self._send(200, "text/csv; charset=utf-8", backtest_csv(report).encode(), "backtest.csv")
                return
            self._json({
                "matches": len(report.predictions),
                "model": asdict(report.model_scores),
                "baseline": asdict(report.baseline_scores),
                "confusion": report.confusion,
                "predictions": [
                    {
                        "date": row.fixture.date,
                        "home": row.fixture.home,
                        "away": row.fixture.away,
                        "home_goals": row.fixture.home_goals,
                        "away_goals": row.fixture.away_goals,
                        "actual": row.fixture.outcome,
                        "prediction": row.prediction,
                        "probabilities": row.probabilities,
                        "trained_through": row.trained_through,
                    }
                    for row in reversed(report.predictions)
                ],
            })
        elif path == "/api/season":
            generated_at = utc_now()
            predictions = forecast_season(predictor, load_schedule(self.server.schedule_path), as_of=generated_at)
            self._json({
                "generated_at": generated_at,
                "results_through": predictor.fixtures[-1].date,
                "predictions": [asdict(row) for row in predictions],
            })
        else:
            self._json({"error": "Page not found."}, HTTPStatus.NOT_FOUND)
