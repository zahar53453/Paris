from __future__ import annotations

from datetime import UTC, datetime

from paris_today_bot.config import config


def log_runtime(message: str) -> None:
    timestamped = f"{datetime.now(UTC).isoformat()} {message}"
    print(timestamped, flush=True)
    path = config.runtime_log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")
