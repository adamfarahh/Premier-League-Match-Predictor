"""Keep a local Git history of the scoreboard's append-only forecast log."""

import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from .scoreboard import forecast_log_csv


def _git(folder: Path, *args: str, allowed: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(folder), *args], capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if result.returncode not in allowed:
        raise ValueError(result.stderr.strip() or f"git {args[0]} exited with code {result.returncode}.")
    return result


def commit_forecast_log(db_path: Path, target: Path) -> str | None:
    target = target.resolve()
    root = Path(_git(target.parent, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    relative = target.relative_to(root).as_posix()
    content = forecast_log_csv(db_path)
    if len(content.splitlines()) < 2:
        return None
    committed = _git(root, "show", f"HEAD:{relative}", allowed=(0, 128))
    existing = [target.read_text(encoding="utf-8")] if target.is_file() else []
    if committed.returncode == 0:
        existing.append(committed.stdout)
    if any(not old or not content.startswith(old) for old in existing):
        raise ValueError("The current forecast database would rewrite existing forecast log rows.")
    if committed.returncode == 0 and committed.stdout == content and target.is_file():
        return None

    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    _git(root, "add", "--", relative)
    if _git(root, "diff", "--cached", "--quiet", "--", relative, allowed=(0, 1)).returncode == 0:
        return None
    _git(
        root, "-c", "user.name=Premier League Forecast Ledger", "-c", "user.email=forecast-ledger@example.invalid",
        "commit", "--only", "-m", "Record pre-kickoff Premier League forecasts", "--", relative,
    )
    return _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
