from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(slots=True)
class CityProfile:
    slug: str
    path: Path
    payload: dict[str, Any]

    @property
    def city_name(self) -> str:
        return str(self.payload["city_name"])

    @property
    def icao(self) -> str:
        return str(self.payload["icao"])

    @property
    def latitude(self) -> float:
        return float(self.payload["station"]["latitude"])

    @property
    def longitude(self) -> float:
        return float(self.payload["station"]["longitude"])

    @property
    def timezone_name(self) -> str:
        return str(self.payload["timezone"])

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def trading_rules(self) -> dict[str, Any]:
        return dict(self.payload.get("trading_rules", {}))

    @property
    def models(self) -> dict[str, Any]:
        return dict(self.payload.get("models", {}))


def load_profile(profile_name: str) -> CityProfile:
    base_dir = Path(__file__).resolve().parent / "profiles"
    path = Path(profile_name)
    if not path.is_absolute():
        path = base_dir / profile_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CityProfile(
        slug=path.stem,
        path=path,
        payload=payload,
    )


def list_profiles() -> list[CityProfile]:
    base_dir = Path(__file__).resolve().parent / "profiles"
    profiles: list[CityProfile] = []
    for path in sorted(base_dir.glob("*.json")):
        profiles.append(load_profile(str(path)))
    return profiles
