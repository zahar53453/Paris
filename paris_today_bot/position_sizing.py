from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_position_sizing_rules() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "position_sizing_rules.json"
    return json.loads(path.read_text(encoding="utf-8"))

