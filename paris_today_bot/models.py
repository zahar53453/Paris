from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MetarObservation:
    observed_at: datetime
    temp_c: int
    raw: str
    cover: str
    wx: str


@dataclass(slots=True)
class WeatherSnapshot:
    now_utc: datetime
    observations: list[MetarObservation]
    obs_current: int | None
    obs_max_so_far: int | None
    max_by_10utc: int | None
    max_by_noon: int | None
    day_regime: str
    morning_clear_count: int
    morning_cloud_count: int
    morning_rain_count: int
    morning_fog_count: int
    rain_count_8_14: int
    gfs_hourly: dict[str, float]
    ecmwf_hourly: dict[str, float]
    ensemble_hourly: dict[str, float] = field(default_factory=dict)
    gfs_daily_max: float | None = None
    ecmwf_daily_max: float | None = None
    ensemble_daily_max: float | None = None
    ensemble_daily_spread: float | None = None
    model_consensus_max: float | None = None
    upper_realistic_max: float | None = None
    model_agreement_spread: float | None = None
    gfs_10utc: float | None = None
    ecmwf_10utc: float | None = None
    model_consensus_10utc: float | None = None


@dataclass(slots=True)
class BucketMarket:
    market_id: str
    question: str
    slug: str
    token_id: str
    no_token_id: str | None
    best_ask: float | None
    best_bid: float | None
    midpoint: float | None
    temperature_c: int | None
    no_best_ask: float | None = None
    no_best_bid: float | None = None
    no_midpoint: float | None = None
    tail: str = "exact"  # exact / or_higher / or_lower


@dataclass(slots=True)
class MarketSnapshot:
    event_title: str
    event_slug: str
    target_date: str
    markets: list[BucketMarket]


@dataclass(slots=True)
class StrategyDecision:
    projected_max: int
    base_max: int
    adjustment: float
    fair_values: dict[str, float]
    reasons: list[str]


@dataclass(slots=True)
class Position:
    market_id: str
    question: str
    token_id: str
    side: str  # YES / NO
    entry_price: float
    size_usd: float
    opened_at: str


@dataclass(slots=True)
class TradeAction:
    action: str  # BUY_YES / BUY_NO / SELL / HOLD / SKIP
    market_id: str
    question: str
    token_id: str
    side: str
    price: float | None
    size_usd: float
    edge: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
