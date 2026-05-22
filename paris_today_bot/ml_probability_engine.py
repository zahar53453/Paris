from __future__ import annotations

import asyncio
import json
import math
import re
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from metar import Metar
from pvlib import location, solarposition
from zoneinfo import ZoneInfo

from paris_today_bot.models import BucketMarket
from paris_today_bot.profile_loader import CityProfile

AVIATION_WEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 120
CONTINUOUS_DISTRIBUTION_GRID_STEP_C = 0.1
TEMPERATURE_BIN_STEP_C = 1.0
TEMPERATURE_BIN_EXTENSION_C = 20.0

PRESENT_WEATHER_CATEGORIES = [
    "rain",
    "snow",
    "fog",
    "thunder",
    "drizzle",
    "freezing",
    "hail",
    "mist",
    "showers",
]

SEASON_BY_MONTH = {
    12: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
    9: 4,
    10: 4,
    11: 4,
}

CLOUD_FRACTION_MAP = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NSC": 0.0,
    "NCD": 0.0,
    "FEW": 0.2,
    "SCT": 0.45,
    "BKN": 0.75,
    "OVC": 1.0,
    "VV": 1.0,
}

WEATHER_TOKEN_MAP = {
    "rain": ("RA", "SHRA", "VCSH"),
    "snow": ("SN", "SG", "PL", "GS"),
    "fog": ("FG", "FZFG"),
    "thunder": ("TS", "TSRA", "VCTS"),
    "drizzle": ("DZ",),
    "freezing": ("FZ", "FZDZ", "FZRA"),
    "hail": ("GR", "GS", "PL"),
    "mist": ("BR", "HZ"),
    "showers": ("SH",),
}

MISSING_MARKERS = {"", "M", "NA", "NAN", "NONE"}

NWP_HOURLY_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
]
NWP_DAILY_VARIABLES = ["temperature_2m_max"]
NWP_HIGH_CONFIDENCE_SPREAD_C = 1.0
NWP_MEDIUM_CONFIDENCE_SPREAD_C = 2.5
NWP_MODELS = [
    {"name": "icon", "provider": "DWD ICON", "models_param": "icon_seamless"},
    {"name": "gfs", "provider": "NOAA GFS", "models_param": "gfs_seamless"},
    {"name": "ecmwf", "provider": "ECMWF IFS", "models_param": "ecmwf_ifs025"},
]
OPEN_METEO_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
OPEN_METEO_MAX_RETRIES = 4
OPEN_METEO_RETRY_BASE_DELAY_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class ModelProjectConfig:
    profile_slug: str
    model_dir: Path
    state_path: Path
    nwp_snapshot_path: Path

    @property
    def expected_model_path(self) -> Path:
        return self.model_dir / "expected_max_model.pkl"

    @property
    def probability_model_path(self) -> Path:
        return self.model_dir / "probability_model.pkl"


@dataclass(slots=True)
class ParsedRealtimeMetar:
    station: str
    valid: pd.Timestamp
    raw_metar: str
    temp_c: float | None
    dewpoint_c: float | None
    wind_speed_kt: float | None
    wind_gust_kt: float | None
    wind_dir_deg: float | None
    visibility_sm: float | None
    cloud_cover_fraction: float | None
    lowest_cloud_base_ft: float | None
    lowest_bkn_ovc_base_ft: float | None
    low_cloud_fraction: float | None
    mid_cloud_fraction: float | None
    high_cloud_fraction: float | None
    has_low_ceiling: int
    fog_or_low_stratus_flag: int
    altimeter_inhg: float | None
    weather_flags: dict[str, int]


@dataclass(slots=True)
class ContinuousResidualDistribution:
    quantile_levels: np.ndarray
    quantile_values_c: np.ndarray
    support_max_c: float


@dataclass(slots=True)
class ProbabilityRecord:
    temperature_c: float
    probability: float
    probability_percent: float


@dataclass(slots=True)
class ProbabilityAnalysis:
    expected_max_c: float
    current_max_so_far: float
    valid_utc: str
    valid_local: str
    raw_metar: str
    probability_table: list[ProbabilityRecord]
    bucket_probabilities: dict[int, float]
    distribution_mean_max_c: float | None = None
    nwp_ensemble: dict[str, Any] | None = None


@dataclass(frozen=True)
class NwpModelSummary:
    name: str
    provider: str
    model_param: str
    fetched_at_utc: str
    local_date: str
    today_max_c: float | None
    remaining_max_c: float | None
    current_hour_temp_c: float | None
    current_hour_dewpoint_c: float | None
    current_hour_relative_humidity_pct: float | None
    current_hour_cloud_cover_pct: float | None
    current_hour_cloud_cover_low_pct: float | None
    current_hour_cloud_cover_mid_pct: float | None
    current_hour_cloud_cover_high_pct: float | None
    current_hour_wind_speed_kt: float | None
    daytime_mean_cloud_cover_pct: float | None
    remaining_mean_cloud_cover_pct: float | None


@dataclass(frozen=True)
class NwpEnsembleSummary:
    local_date: str
    valid_local: str
    available_models: int
    tmax_mean_c: float | None
    tmax_median_c: float | None
    tmax_min_c: float | None
    tmax_max_c: float | None
    tmax_spread_c: float | None
    tmax_std_c: float | None
    remaining_max_mean_c: float | None
    remaining_max_spread_c: float | None
    current_t2m_mean_c: float | None
    current_t2m_spread_c: float | None
    current_dewpoint_mean_c: float | None
    current_dewpoint_spread_c: float | None
    current_relative_humidity_mean_pct: float | None
    current_relative_humidity_spread_pct: float | None
    current_cloud_cover_mean_pct: float | None
    current_cloud_cover_spread_pct: float | None
    current_low_cloud_mean_pct: float | None
    current_low_cloud_spread_pct: float | None
    current_mid_cloud_mean_pct: float | None
    current_mid_cloud_spread_pct: float | None
    current_high_cloud_mean_pct: float | None
    current_high_cloud_spread_pct: float | None
    current_wind_speed_mean_kt: float | None
    current_wind_speed_spread_kt: float | None
    remaining_cloud_mean_pct: float | None
    remaining_cloud_spread_pct: float | None
    consensus_count_within_1c: int
    disagreement_flag: int
    confidence_flag: str
    models: list[dict[str, Any]]


def _project_map() -> dict[str, ModelProjectConfig]:
    base_dir = Path(__file__).resolve().parent
    model_base = base_dir / "ml_models"
    state_base = Path("data") / "ml_state"
    nwp_base = Path("data") / "ml_nwp"
    return {
        "london_eglc_rules": ModelProjectConfig(
            "london_eglc_rules",
            model_base / "eglc_max_temp_forecast",
            state_base / "eglc_max_temp_forecast_state.json",
            nwp_base / "eglc_latest_nwp_snapshot.json",
        ),
        "madrid_lemd_rules": ModelProjectConfig(
            "madrid_lemd_rules",
            model_base / "lemd_max_temp_forecast",
            state_base / "lemd_max_temp_forecast_state.json",
            nwp_base / "lemd_latest_nwp_snapshot.json",
        ),
        "munich_eddm_rules": ModelProjectConfig(
            "munich_eddm_rules",
            model_base / "eddm_max_temp_forecast",
            state_base / "eddm_max_temp_forecast_state.json",
            nwp_base / "eddm_latest_nwp_snapshot.json",
        ),
        "paris_lfpb_rules": ModelProjectConfig(
            "paris_lfpb_rules",
            model_base / "lfpb_max_temp_forecast",
            state_base / "lfpb_max_temp_forecast_state.json",
            nwp_base / "lfpb_latest_nwp_snapshot.json",
        ),
    }


@lru_cache(maxsize=None)
def _project_config_for_slug(profile_slug: str) -> ModelProjectConfig:
    try:
        return _project_map()[profile_slug]
    except KeyError as exc:
        raise RuntimeError(f"No ML nowcast project configured for profile {profile_slug}.") from exc


@lru_cache(maxsize=None)
def _load_model_bundle(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid model artifact structure: {path}")
    return bundle


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().upper()
    if text in MISSING_MARKERS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_cloud_base_ft(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    if hasattr(raw_value, "value"):
        try:
            return float(raw_value.value("FT"))
        except Exception:
            pass
    direct = parse_float(raw_value)
    if direct is not None:
        return direct
    text = str(raw_value).strip().lower()
    feet_match = re.search(r"(-?\d+(?:\.\d+)?)\s*feet?\b", text)
    if feet_match:
        return float(feet_match.group(1))
    return None


def cloud_height_bucket(base_ft: float | None) -> str | None:
    if base_ft is None:
        return None
    if base_ft < 6500.0:
        return "low"
    if base_ft < 20000.0:
        return "mid"
    return "high"


def parse_visibility_sm(raw_value: Any) -> float | None:
    numeric = parse_float(raw_value)
    if numeric is not None:
        return numeric
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    fraction_match = re.match(r"(?:(\d+)\s+)?(\d+)/(\d+)", text)
    if fraction_match:
        whole = float(fraction_match.group(1) or 0.0)
        numerator = float(fraction_match.group(2))
        denominator = float(fraction_match.group(3))
        return whole + numerator / denominator
    return None


def extract_weather_flags(weather_text: str) -> dict[str, int]:
    upper_text = weather_text.upper()
    flags: dict[str, int] = {}
    for category in PRESENT_WEATHER_CATEGORIES:
        tokens = WEATHER_TOKEN_MAP.get(category, ())
        flags[category] = int(any(token in upper_text for token in tokens))
    return flags


def extract_cloud_cover_from_layers(cloud_layers: list[dict[str, Any]] | None) -> float | None:
    if not cloud_layers:
        return None
    fractions: list[float] = []
    for layer in cloud_layers:
        cover = str(layer.get("cover") or "").strip().upper()
        if cover in CLOUD_FRACTION_MAP:
            fractions.append(CLOUD_FRACTION_MAP[cover])
    if not fractions:
        return None
    return float(np.mean(fractions))


def extract_cloud_structure_from_layers(cloud_layers: list[dict[str, Any]] | None) -> dict[str, float | int | None]:
    if not cloud_layers:
        return {
            "lowest_cloud_base_ft": None,
            "lowest_bkn_ovc_base_ft": None,
            "low_cloud_fraction": None,
            "mid_cloud_fraction": None,
            "high_cloud_fraction": None,
            "has_low_ceiling": 0,
            "fog_or_low_stratus_flag": 0,
        }
    lowest_cloud_base_ft: float | None = None
    lowest_bkn_ovc_base_ft: float | None = None
    layer_fractions: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for layer in cloud_layers:
        cover = str(layer.get("cover") or "").strip().upper()
        if cover not in CLOUD_FRACTION_MAP:
            continue
        base_ft = parse_cloud_base_ft(layer.get("base"))
        fraction = CLOUD_FRACTION_MAP[cover]
        if base_ft is not None:
            lowest_cloud_base_ft = base_ft if lowest_cloud_base_ft is None else min(lowest_cloud_base_ft, base_ft)
        if cover in {"BKN", "OVC", "VV"} and base_ft is not None:
            lowest_bkn_ovc_base_ft = (
                base_ft if lowest_bkn_ovc_base_ft is None else min(lowest_bkn_ovc_base_ft, base_ft)
            )
        bucket = cloud_height_bucket(base_ft)
        if bucket is not None:
            layer_fractions[bucket].append(fraction)
    low_cloud_fraction = max(layer_fractions["low"]) if layer_fractions["low"] else None
    mid_cloud_fraction = max(layer_fractions["mid"]) if layer_fractions["mid"] else None
    high_cloud_fraction = max(layer_fractions["high"]) if layer_fractions["high"] else None
    has_low_ceiling = int(lowest_bkn_ovc_base_ft is not None and lowest_bkn_ovc_base_ft < 2000.0)
    return {
        "lowest_cloud_base_ft": lowest_cloud_base_ft,
        "lowest_bkn_ovc_base_ft": lowest_bkn_ovc_base_ft,
        "low_cloud_fraction": low_cloud_fraction,
        "mid_cloud_fraction": mid_cloud_fraction,
        "high_cloud_fraction": high_cloud_fraction,
        "has_low_ceiling": has_low_ceiling,
        "fog_or_low_stratus_flag": has_low_ceiling,
    }


def extract_cloud_cover_from_metar(parsed: Metar.Metar) -> float | None:
    layers = getattr(parsed, "sky", None)
    if not layers:
        return None
    fractions: list[float] = []
    for layer in layers:
        if not layer:
            continue
        code = str(layer[0]).strip().upper()
        if code in CLOUD_FRACTION_MAP:
            fractions.append(CLOUD_FRACTION_MAP[code])
    if not fractions:
        return None
    return float(np.mean(fractions))


def extract_cloud_structure_from_metar(parsed: Metar.Metar, weather_text: str = "") -> dict[str, float | int | None]:
    layers = getattr(parsed, "sky", None)
    if not layers:
        return {
            "lowest_cloud_base_ft": None,
            "lowest_bkn_ovc_base_ft": None,
            "low_cloud_fraction": None,
            "mid_cloud_fraction": None,
            "high_cloud_fraction": None,
            "has_low_ceiling": 0,
            "fog_or_low_stratus_flag": 0,
        }
    lowest_cloud_base_ft: float | None = None
    lowest_bkn_ovc_base_ft: float | None = None
    layer_fractions: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for layer in layers:
        if not layer:
            continue
        cover = str(layer[0]).strip().upper()
        if cover not in CLOUD_FRACTION_MAP:
            continue
        base_ft = None
        if len(layer) > 1 and layer[1] is not None:
            base_ft = parse_cloud_base_ft(layer[1])
        fraction = CLOUD_FRACTION_MAP[cover]
        if base_ft is not None:
            lowest_cloud_base_ft = base_ft if lowest_cloud_base_ft is None else min(lowest_cloud_base_ft, base_ft)
        if cover in {"BKN", "OVC", "VV"} and base_ft is not None:
            lowest_bkn_ovc_base_ft = (
                base_ft if lowest_bkn_ovc_base_ft is None else min(lowest_bkn_ovc_base_ft, base_ft)
            )
        bucket = cloud_height_bucket(base_ft)
        if bucket is not None:
            layer_fractions[bucket].append(fraction)
    low_cloud_fraction = max(layer_fractions["low"]) if layer_fractions["low"] else None
    mid_cloud_fraction = max(layer_fractions["mid"]) if layer_fractions["mid"] else None
    high_cloud_fraction = max(layer_fractions["high"]) if layer_fractions["high"] else None
    has_low_ceiling = int(lowest_bkn_ovc_base_ft is not None and lowest_bkn_ovc_base_ft < 2000.0)
    weather_flags = extract_weather_flags(weather_text)
    return {
        "lowest_cloud_base_ft": lowest_cloud_base_ft,
        "lowest_bkn_ovc_base_ft": lowest_bkn_ovc_base_ft,
        "low_cloud_fraction": low_cloud_fraction,
        "mid_cloud_fraction": mid_cloud_fraction,
        "high_cloud_fraction": high_cloud_fraction,
        "has_low_ceiling": has_low_ceiling,
        "fog_or_low_stratus_flag": int(
            has_low_ceiling or weather_flags.get("fog", 0) or weather_flags.get("mist", 0)
        ),
    }


def normalize_altimeter_to_inhg(raw_value: Any) -> float | None:
    altimeter = parse_float(raw_value)
    if altimeter is None:
        return None
    if altimeter > 100.0:
        return altimeter * 0.0295299830714
    return altimeter


def parse_observation_time(raw_value: Any) -> pd.Timestamp:
    if raw_value is None:
        raise ValueError("METAR payload is missing observation time.")
    if isinstance(raw_value, (int, float)) or str(raw_value).strip().isdigit():
        numeric_value = float(raw_value)
        unit = "ms" if numeric_value > 1e11 else "s"
        parsed = pd.to_datetime(numeric_value, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Could not parse METAR observation time from payload.")
    return parsed


def parse_metar_payload(payload: dict[str, Any], profile: CityProfile) -> ParsedRealtimeMetar:
    raw_metar = str(payload.get("rawOb") or payload.get("raw_text") or "").strip()
    weather_text = str(payload.get("wxString") or payload.get("wxcodes") or raw_metar)
    temp_c = parse_float(payload.get("temp"))
    dewpoint_c = parse_float(payload.get("dewp"))
    wind_speed_kt = parse_float(payload.get("wspd"))
    wind_gust_kt = parse_float(payload.get("wgst"))
    wind_dir_deg = parse_float(payload.get("wdir"))
    visibility_sm = parse_visibility_sm(payload.get("visib"))
    cloud_cover_fraction = extract_cloud_cover_from_layers(payload.get("clouds"))
    cloud_structure = extract_cloud_structure_from_layers(payload.get("clouds"))
    altimeter_inhg = normalize_altimeter_to_inhg(payload.get("altim"))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            parsed = Metar.Metar(raw_metar, strict=False)
        temp_c = parsed.temp.value("C") if parsed.temp else temp_c
        dewpoint_c = parsed.dewpt.value("C") if parsed.dewpt else dewpoint_c
        wind_speed_kt = parsed.wind_speed.value("KT") if parsed.wind_speed else wind_speed_kt
        wind_gust_kt = parsed.wind_gust.value("KT") if parsed.wind_gust else wind_gust_kt
        wind_dir_deg = parsed.wind_dir.value() if parsed.wind_dir else wind_dir_deg
        visibility_sm = parsed.vis.value("SM") if parsed.vis else visibility_sm
        altimeter_inhg = parsed.press.value("IN") if parsed.press else altimeter_inhg
        cloud_cover_fraction = extract_cloud_cover_from_metar(parsed) or cloud_cover_fraction
        cloud_structure_metar = extract_cloud_structure_from_metar(parsed, weather_text=weather_text)
        for key, value in cloud_structure_metar.items():
            if value is not None:
                cloud_structure[key] = value
    except Exception:
        pass

    valid = parse_observation_time(payload.get("obsTime") or payload.get("valid"))
    return ParsedRealtimeMetar(
        station=str(payload.get("icaoId") or payload.get("station") or profile.icao),
        valid=valid,
        raw_metar=raw_metar,
        temp_c=temp_c,
        dewpoint_c=dewpoint_c,
        wind_speed_kt=wind_speed_kt,
        wind_gust_kt=wind_gust_kt,
        wind_dir_deg=wind_dir_deg,
        visibility_sm=visibility_sm,
        cloud_cover_fraction=cloud_cover_fraction,
        lowest_cloud_base_ft=cloud_structure["lowest_cloud_base_ft"],
        lowest_bkn_ovc_base_ft=cloud_structure["lowest_bkn_ovc_base_ft"],
        low_cloud_fraction=cloud_structure["low_cloud_fraction"],
        mid_cloud_fraction=cloud_structure["mid_cloud_fraction"],
        high_cloud_fraction=cloud_structure["high_cloud_fraction"],
        has_low_ceiling=int(cloud_structure["has_low_ceiling"]),
        fog_or_low_stratus_flag=int(
            cloud_structure["fog_or_low_stratus_flag"]
            or extract_weather_flags(weather_text).get("fog", 0)
            or extract_weather_flags(weather_text).get("mist", 0)
        ),
        altimeter_inhg=altimeter_inhg,
        weather_flags=extract_weather_flags(weather_text),
    )


def fetch_latest_metar(profile: CityProfile, session: requests.Session) -> dict[str, Any]:
    response = session.get(
        AVIATION_WEATHER_METAR_URL,
        params={"ids": profile.icao, "format": "json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise RuntimeError("AviationWeather returned an empty METAR payload.")
    if isinstance(payload, list):
        return payload[0]
    raise RuntimeError("Unexpected METAR payload structure.")


def fetch_recent_metars(profile: CityProfile, session: requests.Session, hours: int = 36) -> list[dict[str, Any]]:
    response = session.get(
        AVIATION_WEATHER_METAR_URL,
        params={"ids": profile.icao, "format": "json", "hours": hours},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return []
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected recent METAR payload structure.")


def add_time_features(frame: pd.DataFrame, profile: CityProfile) -> pd.DataFrame:
    df = frame.copy()
    local_tz = ZoneInfo(profile.timezone_name)
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df["local_datetime"] = df["valid"].dt.tz_convert(local_tz)
    df["local_date"] = df["local_datetime"].dt.date
    df["local_hour"] = df["local_datetime"].dt.hour
    df["local_dayofyear"] = df["local_datetime"].dt.dayofyear
    df["day_of_week"] = df["local_datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["season"] = df["local_datetime"].dt.month.map(SEASON_BY_MONTH).astype("Int64")
    df["sin_doy"] = np.sin(2.0 * np.pi * df["local_dayofyear"] / 366.0)
    df["cos_doy"] = np.cos(2.0 * np.pi * df["local_dayofyear"] / 366.0)
    df["sin_hour"] = np.sin(2.0 * np.pi * df["local_hour"] / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * df["local_hour"] / 24.0)
    df["dewpoint_depression"] = df["temp_c"] - df["dewpoint_c"]
    return df


def add_solar_features(frame: pd.DataFrame, profile: CityProfile) -> pd.DataFrame:
    df = frame.copy()
    site = location.Location(
        latitude=profile.latitude,
        longitude=profile.longitude,
        tz=profile.timezone_name,
        name=profile.icao,
    )
    local_times = pd.DatetimeIndex(df["local_datetime"])
    solar = solarposition.get_solarposition(
        time=local_times,
        latitude=profile.latitude,
        longitude=profile.longitude,
    )
    df["solar_elevation"] = solar["elevation"].to_numpy()
    df["is_daytime"] = (df["solar_elevation"] > 0.0).astype(int)
    unique_midnights = pd.DatetimeIndex(pd.Series(local_times.normalize()).drop_duplicates().sort_values().tolist())
    sunrise_sunset = site.get_sun_rise_set_transit(times=unique_midnights)
    sunrise_sunset.index = unique_midnights
    lookup = sunrise_sunset[["sunrise", "sunset"]].copy()
    local_dates = local_times.normalize()
    df["sunrise"] = pd.Series(local_dates, index=df.index).map(lookup["sunrise"])
    df["sunset"] = pd.Series(local_dates, index=df.index).map(lookup["sunset"])
    df["hours_since_sunrise"] = (df["local_datetime"] - df["sunrise"]).dt.total_seconds() / 3600.0
    df["hours_to_sunset"] = (df["sunset"] - df["local_datetime"]).dt.total_seconds() / 3600.0
    return df


def build_feature_row(
    parsed: ParsedRealtimeMetar,
    current_max_so_far: float,
    temp_trend_30m: float | None,
    temp_trend_1h: float | None,
    temp_trend_3h: float | None,
    profile: CityProfile,
    delta_from_current_max_c: float | None = None,
    is_temp_below_current_max_now: int = 0,
    minutes_since_last_max_observation: float | None = None,
    peak_not_updated_recently: int = 0,
    prev_day_max_c: float | None = None,
    prev_day_min_c: float | None = None,
    nwp_features: dict[str, Any] | None = None,
) -> pd.DataFrame:
    base = {
        "station": parsed.station,
        "valid": parsed.valid,
        "metar": parsed.raw_metar,
        "raw": parsed.raw_metar,
        "temp_c": parsed.temp_c,
        "dewpoint_c": parsed.dewpoint_c,
        "wind_speed_kt": parsed.wind_speed_kt,
        "wind_gust_kt": parsed.wind_gust_kt,
        "wind_dir_deg": parsed.wind_dir_deg,
        "visibility_sm": parsed.visibility_sm,
        "cloud_cover_fraction": parsed.cloud_cover_fraction,
        "lowest_cloud_base_ft": parsed.lowest_cloud_base_ft,
        "lowest_bkn_ovc_base_ft": parsed.lowest_bkn_ovc_base_ft,
        "low_cloud_fraction": parsed.low_cloud_fraction,
        "mid_cloud_fraction": parsed.mid_cloud_fraction,
        "high_cloud_fraction": parsed.high_cloud_fraction,
        "has_low_ceiling": parsed.has_low_ceiling,
        "fog_or_low_stratus_flag": parsed.fog_or_low_stratus_flag,
        "altimeter_inhg": parsed.altimeter_inhg,
        "current_max_so_far": current_max_so_far,
        "delta_from_current_max_c": delta_from_current_max_c,
        "is_temp_below_current_max_now": is_temp_below_current_max_now,
        "minutes_since_last_max_observation": minutes_since_last_max_observation,
        "peak_not_updated_recently": peak_not_updated_recently,
        "temp_trend_30m": temp_trend_30m,
        "temp_trend_1h": temp_trend_1h,
        "temp_trend_3h": temp_trend_3h,
        "temp_trend_30m_available": int(temp_trend_30m is not None),
        "temp_trend_1h_available": int(temp_trend_1h is not None),
        "temp_trend_3h_available": int(temp_trend_3h is not None),
        "prev_day_max_c": prev_day_max_c,
        "prev_day_min_c": prev_day_min_c,
    }
    if nwp_features:
        adjusted_nwp = dict(nwp_features)
        remaining_max_mean = adjusted_nwp.get("nwp_remaining_max_mean_c")
        if remaining_max_mean is not None:
            adjusted_nwp["nwp_remaining_max_mean_c"] = max(float(current_max_so_far), float(remaining_max_mean))
        base.update(adjusted_nwp)
    for category in PRESENT_WEATHER_CATEGORIES:
        base[f"wx_{category}"] = parsed.weather_flags.get(category, 0)
    frame = pd.DataFrame([base])
    frame = add_time_features(frame, profile)
    frame = add_solar_features(frame, profile)
    return frame.drop(columns=["sunrise", "sunset"], errors="ignore")


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any], state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def initialize_or_roll_state(state: dict[str, Any], local_date: datetime.date) -> dict[str, Any]:
    state_date = state.get("local_date")
    if state_date != local_date.isoformat():
        return {
            "local_date": local_date.isoformat(),
            "current_max_so_far": None,
            "history": [],
            "prediction_history": [],
        }
    state.setdefault("history", [])
    state.setdefault("prediction_history", [])
    return state


def bootstrap_state_from_recent_history(
    state: dict[str, Any],
    latest_valid: pd.Timestamp,
    profile: CityProfile,
    session: requests.Session,
) -> dict[str, Any]:
    local_tz = ZoneInfo(profile.timezone_name)
    local_date = latest_valid.tz_convert(local_tz).date()
    state = initialize_or_roll_state(state, local_date)
    if state.get("history"):
        return state
    recent_payloads = fetch_recent_metars(profile, session=session)
    parsed_recent: list[ParsedRealtimeMetar] = []
    for payload in recent_payloads:
        try:
            parsed_recent.append(parse_metar_payload(payload, profile))
        except Exception:
            continue
    if not parsed_recent:
        return state
    same_day_history = [
        item
        for item in parsed_recent
        if item.valid <= latest_valid and item.valid.tz_convert(local_tz).date() == local_date
    ]
    same_day_history.sort(key=lambda item: item.valid)
    if not same_day_history:
        return state
    history_before_latest = [
        {"valid": item.valid.isoformat(), "temp_c": item.temp_c}
        for item in same_day_history
        if item.valid < latest_valid
    ]
    temps_before_or_at_latest = [item.temp_c for item in same_day_history if item.temp_c is not None]
    state["history"] = history_before_latest[-96:]
    state["current_max_so_far"] = max(temps_before_or_at_latest) if temps_before_or_at_latest else None
    state["bootstrapped_from_recent_metars"] = True
    return state


def compute_temperature_trend(
    history_frame: pd.DataFrame,
    current_valid: pd.Timestamp,
    current_temp: float | None,
    hours: float,
) -> float | None:
    if current_temp is None or history_frame.empty:
        return None
    history_frame = history_frame.sort_values("valid").copy()
    target_time = current_valid - pd.Timedelta(hours=hours)
    history_frame["time_delta"] = (history_frame["valid"] - target_time).abs()
    candidates = history_frame.loc[
        history_frame["valid"] <= current_valid,
        ["valid", "temp_c", "time_delta"],
    ].dropna(subset=["temp_c"])
    if candidates.empty:
        return None
    best = candidates.sort_values("time_delta").iloc[0]
    if hours <= 0.5:
        tolerance = pd.Timedelta(minutes=45)
    elif hours <= 1.0:
        tolerance = pd.Timedelta(hours=1)
    else:
        tolerance = pd.Timedelta(hours=2)
    if best["time_delta"] > tolerance:
        return None
    return float(current_temp - best["temp_c"])


def derive_peak_state_features(
    history_frame: pd.DataFrame,
    current_valid: pd.Timestamp,
    current_temp: float | None,
    current_max_so_far: float | None,
) -> dict[str, float | int | None]:
    if current_temp is None or current_max_so_far is None:
        return {
            "delta_from_current_max_c": None,
            "is_temp_below_current_max_now": 0,
            "minutes_since_last_max_observation": None,
            "peak_not_updated_recently": 0,
        }
    delta_from_current_max_c = float(current_temp - current_max_so_far)
    is_temp_below_current_max_now = int(delta_from_current_max_c < -1e-6)
    observations = (
        history_frame.loc[:, ["valid", "temp_c"]].copy()
        if not history_frame.empty
        else pd.DataFrame(columns=["valid", "temp_c"])
    )
    current_row = pd.DataFrame([{"valid": current_valid, "temp_c": current_temp}])
    observations = pd.concat([observations, current_row], ignore_index=True)
    observations = observations.dropna(subset=["valid", "temp_c"]).sort_values("valid")
    minutes_since_last_max_observation: float | None = None
    if not observations.empty:
        max_hits = observations.loc[np.isclose(observations["temp_c"], float(current_max_so_far), atol=0.05), "valid"]
        if not max_hits.empty:
            last_max_time = pd.to_datetime(max_hits.iloc[-1], utc=True)
            minutes_since_last_max_observation = float((current_valid - last_max_time).total_seconds() / 60.0)
    peak_not_updated_recently = int(
        is_temp_below_current_max_now == 1
        and minutes_since_last_max_observation is not None
        and minutes_since_last_max_observation >= 90.0
    )
    return {
        "delta_from_current_max_c": round(delta_from_current_max_c, 4),
        "is_temp_below_current_max_now": is_temp_below_current_max_now,
        "minutes_since_last_max_observation": minutes_since_last_max_observation,
        "peak_not_updated_recently": peak_not_updated_recently,
    }


def update_state_with_observation(
    state: dict[str, Any],
    parsed: ParsedRealtimeMetar,
    profile: CityProfile,
) -> tuple[dict[str, Any], float, float | None, float | None, float | None, dict[str, float | int | None]]:
    local_tz = ZoneInfo(profile.timezone_name)
    local_time = parsed.valid.tz_convert(local_tz)
    local_date = local_time.date()
    state = initialize_or_roll_state(state, local_date)
    history = state.get("history", [])
    current_temp = parsed.temp_c
    if history:
        history_frame = pd.DataFrame(history)
        history_frame["valid"] = pd.to_datetime(history_frame["valid"], utc=True)
    else:
        history_frame = pd.DataFrame(columns=["valid", "temp_c"])
    current_max = state.get("current_max_so_far")
    if current_temp is None:
        current_max_so_far = current_max
    elif current_max is None:
        current_max_so_far = current_temp
    else:
        current_max_so_far = max(float(current_max), current_temp)
    if current_max_so_far is None:
        raise RuntimeError("current_max_so_far could not be initialized from the current observation.")
    temp_trend_30m = compute_temperature_trend(history_frame, parsed.valid, current_temp, hours=0.5)
    temp_trend_1h = compute_temperature_trend(history_frame, parsed.valid, current_temp, hours=1.0)
    temp_trend_3h = compute_temperature_trend(history_frame, parsed.valid, current_temp, hours=3.0)
    peak_state_features = derive_peak_state_features(
        history_frame=history_frame,
        current_valid=parsed.valid,
        current_temp=current_temp,
        current_max_so_far=current_max_so_far,
    )
    current_valid_iso = parsed.valid.isoformat()
    if not any(item.get("valid") == current_valid_iso for item in history):
        history.append({"valid": current_valid_iso, "temp_c": current_temp})
    state["history"] = history[-96:]
    state["current_max_so_far"] = current_max_so_far
    state["last_valid_utc"] = parsed.valid.isoformat()
    return (
        state,
        float(current_max_so_far),
        temp_trend_30m,
        temp_trend_1h,
        temp_trend_3h,
        peak_state_features,
    )


def enforce_monotonic_quantiles(quantile_values_c: list[float] | np.ndarray, support_max_c: float) -> np.ndarray:
    quantiles = np.asarray(quantile_values_c, dtype=float)
    quantiles = np.nan_to_num(quantiles, nan=0.0, posinf=support_max_c, neginf=0.0)
    quantiles = np.clip(quantiles, 0.0, support_max_c)
    return np.maximum.accumulate(quantiles)


def build_continuous_residual_distribution(
    quantile_levels: list[float] | np.ndarray,
    quantile_values_c: list[float] | np.ndarray,
    support_max_c: float = TEMPERATURE_BIN_EXTENSION_C,
) -> ContinuousResidualDistribution:
    levels = np.asarray(quantile_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("Quantile levels must be a non-empty 1D array.")
    values = enforce_monotonic_quantiles(quantile_values_c=quantile_values_c, support_max_c=support_max_c)
    if values.shape != levels.shape:
        raise ValueError("Quantile levels and values must have the same shape.")
    return ContinuousResidualDistribution(
        quantile_levels=levels,
        quantile_values_c=values,
        support_max_c=float(support_max_c),
    )


def _distribution_knots(distribution: ContinuousResidualDistribution) -> tuple[np.ndarray, np.ndarray]:
    support = np.concatenate(
        [
            np.array([0.0], dtype=float),
            distribution.quantile_values_c.astype(float),
            np.array([distribution.support_max_c], dtype=float),
        ]
    )
    probabilities = np.concatenate(
        [
            np.array([0.0], dtype=float),
            distribution.quantile_levels.astype(float),
            np.array([1.0], dtype=float),
        ]
    )
    support = np.maximum.accumulate(np.clip(support, 0.0, distribution.support_max_c))
    return support, probabilities


def evaluate_continuous_cdf(x: float | list[float] | np.ndarray, distribution: ContinuousResidualDistribution) -> np.ndarray:
    support, probabilities = _distribution_knots(distribution)
    values = np.asarray(x, dtype=float)
    flat_values = values.reshape(-1)
    cdf = np.zeros_like(flat_values, dtype=float)
    for index, item in enumerate(flat_values):
        if item <= 0.0:
            cdf[index] = 0.0
            continue
        if item >= distribution.support_max_c:
            cdf[index] = 1.0
            continue
        right = int(np.searchsorted(support, item, side="right"))
        left = right - 1
        if left < 0:
            cdf[index] = 0.0
            continue
        if left >= len(support) - 1:
            cdf[index] = 1.0
            continue
        left_support = support[left]
        left_prob = probabilities[left]
        next_index = left + 1
        while next_index < len(support) and support[next_index] == left_support:
            left_prob = max(left_prob, probabilities[next_index])
            next_index += 1
        if item <= left_support:
            cdf[index] = left_prob
            continue
        if next_index >= len(support):
            cdf[index] = 1.0
            continue
        right_support = support[next_index]
        right_prob = probabilities[next_index]
        if right_support <= left_support:
            cdf[index] = right_prob
            continue
        weight = (item - left_support) / (right_support - left_support)
        cdf[index] = left_prob + weight * (right_prob - left_prob)
    return cdf.reshape(values.shape)


def probability_mass_by_residual_bins(
    distribution: ContinuousResidualDistribution,
    bin_step_c: float = TEMPERATURE_BIN_STEP_C,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.arange(0.0, distribution.support_max_c + bin_step_c, bin_step_c, dtype=float)
    half_step = bin_step_c / 2.0
    lowers = centers - half_step
    uppers = centers + half_step
    lower_cdf = np.where(lowers <= 0.0, 0.0, evaluate_continuous_cdf(lowers, distribution))
    upper_cdf = evaluate_continuous_cdf(np.minimum(uppers, distribution.support_max_c), distribution)
    probabilities = np.clip(upper_cdf - lower_cdf, 0.0, 1.0)
    total_probability = probabilities.sum()
    if total_probability > 0.0:
        probabilities = probabilities / total_probability
    return centers, probabilities


def expected_residual_from_distribution(
    distribution: ContinuousResidualDistribution,
    grid_step_c: float = CONTINUOUS_DISTRIBUTION_GRID_STEP_C,
) -> float:
    grid = np.arange(0.0, distribution.support_max_c + grid_step_c, grid_step_c, dtype=float)
    cdf = evaluate_continuous_cdf(grid, distribution)
    survival = 1.0 - cdf
    return float(np.trapezoid(survival, grid))


def build_probability_table(
    current_max_so_far: float,
    distribution: ContinuousResidualDistribution,
    min_probability_percent: float = 0.0,
    bin_step_c: float = TEMPERATURE_BIN_STEP_C,
) -> pd.DataFrame:
    residual_centers, probabilities = probability_mass_by_residual_bins(distribution=distribution, bin_step_c=bin_step_c)
    absolute_temps = current_max_so_far + residual_centers
    table = pd.DataFrame(
        {
            "temperature_c": absolute_temps,
            "residual_c": residual_centers,
            "probability": probabilities,
            "probability_percent": probabilities * 100.0,
        }
    )
    table = table.loc[table["temperature_c"] >= current_max_so_far].copy()
    table = table.loc[table["probability_percent"] >= min_probability_percent].copy()
    table["temperature_c"] = table["temperature_c"].round(1)
    table["probability_percent"] = table["probability_percent"].round(1)
    table = table.sort_values(["temperature_c", "probability"], ascending=[True, False]).reset_index(drop=True)
    return table


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values))


def _spread_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


def _request_open_meteo_json(*, session: requests.Session, url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(OPEN_METEO_MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in OPEN_METEO_RETRY_STATUS_CODES and attempt < OPEN_METEO_MAX_RETRIES - 1:
                delay = OPEN_METEO_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= OPEN_METEO_MAX_RETRIES - 1:
                break
            time.sleep(OPEN_METEO_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Open-Meteo request failed without a captured exception.")


def _prepare_open_meteo_frame(payload: dict[str, Any], current_local: pd.Timestamp, profile: CityProfile) -> tuple[pd.DataFrame, float | None]:
    hourly = payload.get("hourly") or {}
    hourly_times = hourly.get("time") or []
    if not hourly_times:
        return pd.DataFrame(), None
    local_tz = ZoneInfo(profile.timezone_name)
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly_times, utc=False, errors="coerce"),
            "temperature_2m": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
            "dew_point_2m": pd.to_numeric(hourly.get("dew_point_2m"), errors="coerce"),
            "relative_humidity_2m": pd.to_numeric(hourly.get("relative_humidity_2m"), errors="coerce"),
            "cloud_cover": pd.to_numeric(hourly.get("cloud_cover"), errors="coerce"),
            "cloud_cover_low": pd.to_numeric(hourly.get("cloud_cover_low"), errors="coerce"),
            "cloud_cover_mid": pd.to_numeric(hourly.get("cloud_cover_mid"), errors="coerce"),
            "cloud_cover_high": pd.to_numeric(hourly.get("cloud_cover_high"), errors="coerce"),
            "wind_speed_10m": pd.to_numeric(hourly.get("wind_speed_10m"), errors="coerce"),
        }
    ).dropna(subset=["time"])
    frame["time_local"] = pd.DatetimeIndex(frame["time"]).tz_localize(local_tz, nonexistent="shift_forward")
    daily_payload = payload.get("daily") or {}
    daily_times = daily_payload.get("time") or []
    daily_maxes = daily_payload.get("temperature_2m_max") or []
    today_max_c: float | None = None
    local_date_str = current_local.date().isoformat()
    if daily_times and daily_maxes:
        daily_frame = pd.DataFrame(
            {
                "date": [str(item) for item in daily_times],
                "temperature_2m_max": pd.to_numeric(daily_maxes, errors="coerce"),
            }
        )
        match = daily_frame.loc[daily_frame["date"] == local_date_str, "temperature_2m_max"]
        if not match.empty and pd.notna(match.iloc[0]):
            today_max_c = float(match.iloc[0])
    return frame, today_max_c


def fetch_single_nwp_model_summary(
    *,
    session: requests.Session,
    model_config: dict[str, str],
    current_local: pd.Timestamp,
    profile: CityProfile,
) -> NwpModelSummary:
    payload = _request_open_meteo_json(
        session=session,
        url=OPEN_METEO_FORECAST_URL,
        params={
            "latitude": profile.latitude,
            "longitude": profile.longitude,
            "timezone": profile.timezone_name,
            "models": model_config["models_param"],
            "hourly": ",".join(NWP_HOURLY_VARIABLES),
            "daily": ",".join(NWP_DAILY_VARIABLES),
            "forecast_days": 2,
            "wind_speed_unit": "kn",
        },
    )
    hourly_frame, today_max_c = _prepare_open_meteo_frame(payload, current_local=current_local, profile=profile)
    if hourly_frame.empty:
        return NwpModelSummary(
            name=model_config["name"],
            provider=model_config["provider"],
            model_param=model_config["models_param"],
            fetched_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            local_date=current_local.date().isoformat(),
            today_max_c=_round_or_none(today_max_c),
            remaining_max_c=None,
            current_hour_temp_c=None,
            current_hour_dewpoint_c=None,
            current_hour_relative_humidity_pct=None,
            current_hour_cloud_cover_pct=None,
            current_hour_cloud_cover_low_pct=None,
            current_hour_cloud_cover_mid_pct=None,
            current_hour_cloud_cover_high_pct=None,
            current_hour_wind_speed_kt=None,
            daytime_mean_cloud_cover_pct=None,
            remaining_mean_cloud_cover_pct=None,
        )
    local_day_frame = hourly_frame.loc[hourly_frame["time_local"].dt.date == current_local.date()].copy()
    remaining_frame = local_day_frame.loc[local_day_frame["time_local"] >= current_local.floor("h")].copy()
    current_hour_temp_c = None
    current_hour_dewpoint_c = None
    current_hour_relative_humidity_pct = None
    current_hour_cloud_cover_pct = None
    current_hour_cloud_cover_low_pct = None
    current_hour_cloud_cover_mid_pct = None
    current_hour_cloud_cover_high_pct = None
    current_hour_wind_speed_kt = None
    if not remaining_frame.empty:
        current_row = remaining_frame.iloc[0]
        current_hour_temp_c = _round_or_none(float(current_row["temperature_2m"]))
        current_hour_dewpoint_c = None if pd.isna(current_row["dew_point_2m"]) else _round_or_none(float(current_row["dew_point_2m"]))
        current_hour_relative_humidity_pct = (
            None if pd.isna(current_row["relative_humidity_2m"]) else _round_or_none(float(current_row["relative_humidity_2m"]))
        )
        current_hour_cloud_cover_pct = None if pd.isna(current_row["cloud_cover"]) else _round_or_none(float(current_row["cloud_cover"]))
        current_hour_cloud_cover_low_pct = (
            None if pd.isna(current_row["cloud_cover_low"]) else _round_or_none(float(current_row["cloud_cover_low"]))
        )
        current_hour_cloud_cover_mid_pct = (
            None if pd.isna(current_row["cloud_cover_mid"]) else _round_or_none(float(current_row["cloud_cover_mid"]))
        )
        current_hour_cloud_cover_high_pct = (
            None if pd.isna(current_row["cloud_cover_high"]) else _round_or_none(float(current_row["cloud_cover_high"]))
        )
        current_hour_wind_speed_kt = (
            None if pd.isna(current_row["wind_speed_10m"]) else _round_or_none(float(current_row["wind_speed_10m"]))
        )
    remaining_max_c = None
    if not remaining_frame.empty and remaining_frame["temperature_2m"].notna().any():
        remaining_max_c = _round_or_none(float(remaining_frame["temperature_2m"].max()))
    daytime_frame = local_day_frame.loc[(local_day_frame["time_local"].dt.hour >= 6) & (local_day_frame["time_local"].dt.hour <= 20)]
    daytime_mean_cloud_cover_pct = None
    if not daytime_frame.empty and daytime_frame["cloud_cover"].notna().any():
        daytime_mean_cloud_cover_pct = _round_or_none(float(daytime_frame["cloud_cover"].mean()))
    remaining_mean_cloud_cover_pct = None
    if not remaining_frame.empty and remaining_frame["cloud_cover"].notna().any():
        remaining_mean_cloud_cover_pct = _round_or_none(float(remaining_frame["cloud_cover"].mean()))
    return NwpModelSummary(
        name=model_config["name"],
        provider=model_config["provider"],
        model_param=model_config["models_param"],
        fetched_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        local_date=current_local.date().isoformat(),
        today_max_c=_round_or_none(today_max_c),
        remaining_max_c=remaining_max_c,
        current_hour_temp_c=current_hour_temp_c,
        current_hour_dewpoint_c=current_hour_dewpoint_c,
        current_hour_relative_humidity_pct=current_hour_relative_humidity_pct,
        current_hour_cloud_cover_pct=current_hour_cloud_cover_pct,
        current_hour_cloud_cover_low_pct=current_hour_cloud_cover_low_pct,
        current_hour_cloud_cover_mid_pct=current_hour_cloud_cover_mid_pct,
        current_hour_cloud_cover_high_pct=current_hour_cloud_cover_high_pct,
        current_hour_wind_speed_kt=current_hour_wind_speed_kt,
        daytime_mean_cloud_cover_pct=daytime_mean_cloud_cover_pct,
        remaining_mean_cloud_cover_pct=remaining_mean_cloud_cover_pct,
    )


def summarize_nwp_ensemble(model_summaries: list[NwpModelSummary], current_local: pd.Timestamp) -> NwpEnsembleSummary:
    tmax_values = [item.today_max_c for item in model_summaries if item.today_max_c is not None]
    remaining_max_values = [item.remaining_max_c for item in model_summaries if item.remaining_max_c is not None]
    current_t2m_values = [item.current_hour_temp_c for item in model_summaries if item.current_hour_temp_c is not None]
    current_dewpoint_values = [item.current_hour_dewpoint_c for item in model_summaries if item.current_hour_dewpoint_c is not None]
    current_rh_values = [
        item.current_hour_relative_humidity_pct
        for item in model_summaries
        if item.current_hour_relative_humidity_pct is not None
    ]
    current_cloud_values = [item.current_hour_cloud_cover_pct for item in model_summaries if item.current_hour_cloud_cover_pct is not None]
    current_low_cloud_values = [item.current_hour_cloud_cover_low_pct for item in model_summaries if item.current_hour_cloud_cover_low_pct is not None]
    current_mid_cloud_values = [item.current_hour_cloud_cover_mid_pct for item in model_summaries if item.current_hour_cloud_cover_mid_pct is not None]
    current_high_cloud_values = [item.current_hour_cloud_cover_high_pct for item in model_summaries if item.current_hour_cloud_cover_high_pct is not None]
    current_wind_values = [item.current_hour_wind_speed_kt for item in model_summaries if item.current_hour_wind_speed_kt is not None]
    remaining_cloud_values = [item.remaining_mean_cloud_cover_pct for item in model_summaries if item.remaining_mean_cloud_cover_pct is not None]
    tmax_mean_c = _mean_or_none(tmax_values)
    tmax_median_c = float(np.median(tmax_values)) if tmax_values else None
    tmax_min_c = min(tmax_values) if tmax_values else None
    tmax_max_c = max(tmax_values) if tmax_values else None
    tmax_spread_c = _spread_or_none(tmax_values)
    tmax_std_c = _std_or_none(tmax_values)
    remaining_max_mean_c = _mean_or_none(remaining_max_values)
    remaining_max_spread_c = _spread_or_none(remaining_max_values)
    remaining_cloud_mean_pct = _mean_or_none(remaining_cloud_values)
    remaining_cloud_spread_pct = _spread_or_none(remaining_cloud_values)
    consensus_count_within_1c = 0
    disagreement_flag = 0
    confidence_flag = "unknown"
    if tmax_values and tmax_mean_c is not None and tmax_spread_c is not None:
        consensus_count_within_1c = int(sum(abs(value - tmax_mean_c) <= 1.0 for value in tmax_values))
        disagreement_flag = int(tmax_spread_c > NWP_MEDIUM_CONFIDENCE_SPREAD_C)
        if tmax_spread_c <= NWP_HIGH_CONFIDENCE_SPREAD_C:
            confidence_flag = "high"
        elif tmax_spread_c <= NWP_MEDIUM_CONFIDENCE_SPREAD_C:
            confidence_flag = "medium"
        else:
            confidence_flag = "low"
    return NwpEnsembleSummary(
        local_date=current_local.date().isoformat(),
        valid_local=current_local.isoformat(),
        available_models=len(model_summaries),
        tmax_mean_c=_round_or_none(tmax_mean_c),
        tmax_median_c=_round_or_none(tmax_median_c),
        tmax_min_c=_round_or_none(tmax_min_c),
        tmax_max_c=_round_or_none(tmax_max_c),
        tmax_spread_c=_round_or_none(tmax_spread_c),
        tmax_std_c=_round_or_none(tmax_std_c),
        remaining_max_mean_c=_round_or_none(remaining_max_mean_c),
        remaining_max_spread_c=_round_or_none(remaining_max_spread_c),
        current_t2m_mean_c=_round_or_none(_mean_or_none(current_t2m_values)),
        current_t2m_spread_c=_round_or_none(_spread_or_none(current_t2m_values)),
        current_dewpoint_mean_c=_round_or_none(_mean_or_none(current_dewpoint_values)),
        current_dewpoint_spread_c=_round_or_none(_spread_or_none(current_dewpoint_values)),
        current_relative_humidity_mean_pct=_round_or_none(_mean_or_none(current_rh_values)),
        current_relative_humidity_spread_pct=_round_or_none(_spread_or_none(current_rh_values)),
        current_cloud_cover_mean_pct=_round_or_none(_mean_or_none(current_cloud_values)),
        current_cloud_cover_spread_pct=_round_or_none(_spread_or_none(current_cloud_values)),
        current_low_cloud_mean_pct=_round_or_none(_mean_or_none(current_low_cloud_values)),
        current_low_cloud_spread_pct=_round_or_none(_spread_or_none(current_low_cloud_values)),
        current_mid_cloud_mean_pct=_round_or_none(_mean_or_none(current_mid_cloud_values)),
        current_mid_cloud_spread_pct=_round_or_none(_spread_or_none(current_mid_cloud_values)),
        current_high_cloud_mean_pct=_round_or_none(_mean_or_none(current_high_cloud_values)),
        current_high_cloud_spread_pct=_round_or_none(_spread_or_none(current_high_cloud_values)),
        current_wind_speed_mean_kt=_round_or_none(_mean_or_none(current_wind_values)),
        current_wind_speed_spread_kt=_round_or_none(_spread_or_none(current_wind_values)),
        remaining_cloud_mean_pct=_round_or_none(remaining_cloud_mean_pct),
        remaining_cloud_spread_pct=_round_or_none(remaining_cloud_spread_pct),
        consensus_count_within_1c=consensus_count_within_1c,
        disagreement_flag=disagreement_flag,
        confidence_flag=confidence_flag,
        models=[asdict(item) for item in model_summaries],
    )


def save_nwp_snapshot(summary: NwpEnsembleSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def load_nwp_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fetch_nwp_ensemble_summary(
    *,
    current_local: pd.Timestamp,
    profile: CityProfile,
    snapshot_path: Path,
    session: requests.Session,
) -> dict[str, Any] | None:
    model_summaries: list[NwpModelSummary] = []
    failures: list[str] = []
    for model_config in NWP_MODELS:
        try:
            model_summaries.append(
                fetch_single_nwp_model_summary(
                    session=session,
                    model_config=model_config,
                    current_local=current_local,
                    profile=profile,
                )
            )
        except Exception as exc:
            failures.append(f"{model_config['name']}: {exc}")
    if not model_summaries:
        cached = load_nwp_snapshot(snapshot_path)
        if cached is not None and str(cached.get("local_date") or "") == current_local.date().isoformat():
            cached["source"] = "cached_snapshot"
            if failures:
                cached["fetch_failures"] = failures
            return cached
        return None
    summary = summarize_nwp_ensemble(model_summaries=model_summaries, current_local=current_local)
    payload = asdict(summary)
    payload["source"] = "live_open_meteo"
    if failures:
        payload["fetch_failures"] = failures
    save_nwp_snapshot(summary, snapshot_path)
    return payload


def nwp_summary_to_feature_dict(summary: dict[str, Any] | None) -> dict[str, float | int | None]:
    if not summary:
        return {}
    features: dict[str, float | int | None] = {
        "nwp_t2m_c_mean": summary.get("current_t2m_mean_c"),
        "nwp_t2m_c_spread": summary.get("current_t2m_spread_c"),
        "nwp_dewpoint_c_mean": summary.get("current_dewpoint_mean_c"),
        "nwp_dewpoint_c_spread": summary.get("current_dewpoint_spread_c"),
        "nwp_relative_humidity_pct_mean": summary.get("current_relative_humidity_mean_pct"),
        "nwp_relative_humidity_pct_spread": summary.get("current_relative_humidity_spread_pct"),
        "nwp_cloud_cover_pct_mean": summary.get("current_cloud_cover_mean_pct"),
        "nwp_cloud_cover_pct_spread": summary.get("current_cloud_cover_spread_pct"),
        "nwp_low_cloud_pct_mean": summary.get("current_low_cloud_mean_pct"),
        "nwp_low_cloud_pct_spread": summary.get("current_low_cloud_spread_pct"),
        "nwp_mid_cloud_pct_mean": summary.get("current_mid_cloud_mean_pct"),
        "nwp_mid_cloud_pct_spread": summary.get("current_mid_cloud_spread_pct"),
        "nwp_high_cloud_pct_mean": summary.get("current_high_cloud_mean_pct"),
        "nwp_high_cloud_pct_spread": summary.get("current_high_cloud_spread_pct"),
        "nwp_wind_speed_kt_mean": summary.get("current_wind_speed_mean_kt"),
        "nwp_wind_speed_kt_spread": summary.get("current_wind_speed_spread_kt"),
        "nwp_today_max_mean_c": summary.get("tmax_mean_c"),
        "nwp_today_max_spread_c": summary.get("tmax_spread_c"),
        "nwp_remaining_max_mean_c": summary.get("remaining_max_mean_c"),
        "nwp_remaining_max_spread_c": summary.get("remaining_max_spread_c"),
        "nwp_models_available": summary.get("available_models"),
    }
    for model in summary.get("models", []):
        prefix = f"nwp_{model.get('name')}"
        features[f"{prefix}_today_max_c"] = model.get("today_max_c")
        features[f"{prefix}_temperature_2m_c"] = model.get("current_hour_temp_c")
        features[f"{prefix}_dew_point_2m_c"] = model.get("current_hour_dewpoint_c")
        features[f"{prefix}_relative_humidity_2m_pct"] = model.get("current_hour_relative_humidity_pct")
        features[f"{prefix}_cloud_cover_pct"] = model.get("current_hour_cloud_cover_pct")
        features[f"{prefix}_cloud_cover_low_pct"] = model.get("current_hour_cloud_cover_low_pct")
        features[f"{prefix}_cloud_cover_mid_pct"] = model.get("current_hour_cloud_cover_mid_pct")
        features[f"{prefix}_cloud_cover_high_pct"] = model.get("current_hour_cloud_cover_high_pct")
        features[f"{prefix}_wind_speed_10m_kt"] = model.get("current_hour_wind_speed_kt")
    return features


def load_previous_day_context(project: ModelProjectConfig, local_date_iso: str) -> tuple[float | None, float | None]:
    daily_context_path = project.model_dir / "daily_context.json"
    if not daily_context_path.exists():
        return None, None
    try:
        payload = json.loads(daily_context_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    record = payload.get(local_date_iso)
    if not isinstance(record, dict):
        return None, None
    prev_day_max_c = record.get("prev_day_max_c")
    prev_day_min_c = record.get("prev_day_min_c")
    return (
        None if prev_day_max_c is None else float(prev_day_max_c),
        None if prev_day_min_c is None else float(prev_day_min_c),
    )


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _bucket_probabilities_from_table(table: pd.DataFrame) -> dict[int, float]:
    bucket_probs: dict[int, float] = {}
    for row in table.to_dict(orient="records"):
        try:
            temperature_c = float(row["temperature_c"])
            probability = float(row["probability"])
        except (KeyError, TypeError, ValueError):
            continue
        bucket = _round_half_up(temperature_c)
        bucket_probs[bucket] = bucket_probs.get(bucket, 0.0) + probability
    total = sum(bucket_probs.values())
    if total > 0:
        bucket_probs = {bucket: prob / total for bucket, prob in bucket_probs.items()}
    return bucket_probs


def _market_probability(market: BucketMarket, bucket_probs: dict[int, float]) -> float:
    if market.temperature_c is None:
        return 0.0
    if market.tail == "exact":
        return bucket_probs.get(market.temperature_c, 0.0)
    if market.tail == "or_higher":
        return sum(prob for bucket, prob in bucket_probs.items() if bucket >= market.temperature_c)
    if market.tail == "or_lower":
        return sum(prob for bucket, prob in bucket_probs.items() if bucket <= market.temperature_c)
    return 0.0


class MLProbabilityEngine:
    def __init__(self) -> None:
        self._project_map = _project_map()

    async def analyze(self, profile: CityProfile, markets: list[BucketMarket]) -> tuple[ProbabilityAnalysis, dict[str, float]]:
        return await asyncio.to_thread(self._analyze_sync, profile, markets)

    def _analyze_sync(self, profile: CityProfile, markets: list[BucketMarket]) -> tuple[ProbabilityAnalysis, dict[str, float]]:
        project = _project_config_for_slug(profile.slug)
        expected_bundle = _load_model_bundle(str(project.expected_model_path))
        probability_bundle = _load_model_bundle(str(project.probability_model_path))
        state = load_state(project.state_path)
        with requests.Session() as session:
            raw_payload = fetch_latest_metar(profile, session)
            parsed = parse_metar_payload(raw_payload, profile)
            state = bootstrap_state_from_recent_history(state, parsed.valid, profile, session)
            (
                state,
                current_max_so_far,
                temp_trend_30m,
                temp_trend_1h,
                temp_trend_3h,
                peak_state_features,
            ) = update_state_with_observation(state, parsed, profile)
            local_date_iso = parsed.valid.tz_convert(ZoneInfo(profile.timezone_name)).date().isoformat()
            prev_day_max_c, prev_day_min_c = load_previous_day_context(project, local_date_iso)
            nwp_summary = fetch_nwp_ensemble_summary(
                current_local=parsed.valid.tz_convert(ZoneInfo(profile.timezone_name)),
                profile=profile,
                snapshot_path=project.nwp_snapshot_path,
                session=session,
            )
            nwp_features = nwp_summary_to_feature_dict(nwp_summary)
            feature_row = build_feature_row(
                parsed=parsed,
                current_max_so_far=current_max_so_far,
                temp_trend_30m=temp_trend_30m,
                temp_trend_1h=temp_trend_1h,
                temp_trend_3h=temp_trend_3h,
                profile=profile,
                delta_from_current_max_c=peak_state_features.get("delta_from_current_max_c"),
                is_temp_below_current_max_now=int(peak_state_features.get("is_temp_below_current_max_now", 0)),
                minutes_since_last_max_observation=peak_state_features.get("minutes_since_last_max_observation"),
                peak_not_updated_recently=int(peak_state_features.get("peak_not_updated_recently", 0)),
                prev_day_max_c=prev_day_max_c,
                prev_day_min_c=prev_day_min_c,
                nwp_features=nwp_features,
            )

        expected_features = feature_row[expected_bundle["feature_columns"]]
        probability_features = feature_row[probability_bundle["feature_columns"]]
        expected_max_c = max(float(current_max_so_far), float(expected_bundle["pipeline"].predict(expected_features)[0]))

        probability_preprocessor = probability_bundle["preprocessor"]
        transformed_probability_features = probability_preprocessor.transform(probability_features)
        quantile_levels = np.asarray(probability_bundle["quantile_levels"], dtype=float)
        quantile_models = probability_bundle["quantile_models"]
        predicted_quantiles = [
            float(quantile_models[float(level)].predict(transformed_probability_features)[0])
            for level in quantile_levels
        ]
        support_max_c = float(probability_bundle.get("support_max_c", probability_bundle.get("max_extension_c", TEMPERATURE_BIN_EXTENSION_C)))
        bin_step_c = float(probability_bundle.get("bin_step_c", TEMPERATURE_BIN_STEP_C))
        distribution = build_continuous_residual_distribution(
            quantile_levels=quantile_levels,
            quantile_values_c=predicted_quantiles,
            support_max_c=support_max_c,
        )
        probability_table_df = build_probability_table(
            current_max_so_far=current_max_so_far,
            distribution=distribution,
            bin_step_c=bin_step_c,
        )
        distribution_mean_max_c = float(current_max_so_far) + expected_residual_from_distribution(distribution)
        top_row = (
            probability_table_df.sort_values(["probability", "temperature_c"], ascending=[False, True]).iloc[0]
            if not probability_table_df.empty
            else None
        )
        record = {
            "valid_utc": parsed.valid.isoformat(),
            "valid_local": parsed.valid.tz_convert(ZoneInfo(profile.timezone_name)).isoformat(),
            "expected_max_c": round(expected_max_c, 2),
            "distribution_mean_max_c": round(distribution_mean_max_c, 2),
            "current_max_so_far": round(current_max_so_far, 2),
            "top_scenario_temp_c": round(float(top_row["temperature_c"]), 2) if top_row is not None else None,
            "top_scenario_probability_percent": round(float(top_row["probability_percent"]), 2) if top_row is not None else None,
            "probability_table": probability_table_df.to_dict(orient="records"),
            "nwp_ensemble_summary": nwp_summary,
            "raw_metar": parsed.raw_metar,
        }
        history = state.get("prediction_history", [])
        history.append(record)
        state["prediction_history"] = history[-256:]
        state["latest_prediction"] = record
        save_state(state, project.state_path)

        probability_table = [
            ProbabilityRecord(
                temperature_c=float(row["temperature_c"]),
                probability=float(row["probability"]),
                probability_percent=float(row["probability_percent"]),
            )
            for row in probability_table_df.to_dict(orient="records")
        ]
        bucket_probs = _bucket_probabilities_from_table(probability_table_df)
        fair_values = {market.market_id: _market_probability(market, bucket_probs) for market in markets}
        analysis = ProbabilityAnalysis(
            expected_max_c=expected_max_c,
            current_max_so_far=current_max_so_far,
            valid_utc=parsed.valid.isoformat(),
            valid_local=parsed.valid.tz_convert(ZoneInfo(profile.timezone_name)).isoformat(),
            raw_metar=parsed.raw_metar,
            probability_table=probability_table,
            bucket_probabilities=bucket_probs,
            distribution_mean_max_c=distribution_mean_max_c,
            nwp_ensemble=nwp_summary,
        )
        return analysis, fair_values
