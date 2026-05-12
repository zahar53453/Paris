from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from paris_today_bot.models import Position


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_positions(self) -> list[Position]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [Position(**item) for item in payload.get("positions", [])]

    def save_positions(self, positions: list[Position]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"positions": [asdict(position) for position in positions]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

