from __future__ import annotations

import asyncio
import json
import math
import re
import warnings
from dataclasses import dataclass
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
REQUEST_TIMEOUT_SECONDS = 120
TEMPERATURE_BIN_STEP_C = 0.5
TEMPERATURE_BIN_EXTENSION_C = 15.0
TEMP_BUCKET_RE = re.compile(r"\b(\d+)\s*[^0-9A-Za-z]{0,2}C\b", re.IGNORECASE)

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


@dataclass(frozen=True, slots=True)
class ModelProjectConfig:
    profile_slug: str
    model_dir: Path
    state_path: Path

    @property
    def models_dir(self) -> Path:
        return self.model_dir

    @property
    def expected_model_path(self) -> Path:
        return self.models_dir / "expected_max_model.pkl"

    @property
    def probability_model_path(self) -> Path:
        return self.models_dir / "probability_model.pkl"


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
    altimeter_inhg: float | None
    weather_flags: dict[str, int]


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


def _project_map() -> dict[str, ModelProjectConfig]:
    base_dir = Path(__file__).resolve().parent
    model_base = base_dir / "ml_models"
    state_base = Path("data") / "ml_state"
    return {
        "london_eglc_rules": ModelProjectConfig(
            "london_eglc_rules",
            model_base / "eglc_max_temp_forecast",
            state_base / "eglc_max_temp_forecast_state.json",
        ),
        "madrid_lemd_rules": ModelProjectConfig(
            "madrid_lemd_rules",
            model_base / "lemd_max_temp_forecast",
            state_base / "lemd_max_temp_forecast_state.json",
        ),
        "munich_eddm_rules": ModelProjectConfig(
            "munich_eddm_rules",
            model_base / "eddm_max_temp_forecast",
            state_base / "eddm_max_temp_forecast_state.json",
        ),
        "paris_lfpb_rules": ModelProjectConfig(
            "paris_lfpb_rules",
            model_base / "lfpb_max_temp_forecast",
            state_base / "lfpb_max_temp_forecast_state.json",
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
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
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
    unique_midnights = pd.DatetimeIndex(
        pd.Series(local_times.normalize()).drop_duplicates().sort_values().tolist()
    )
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
    temp_trend_1h: float | None,
    temp_trend_3h: float | None,
    profile: CityProfile,
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
        "altimeter_inhg": parsed.altimeter_inhg,
        "current_max_so_far": current_max_so_far,
        "temp_trend_1h": temp_trend_1h,
        "temp_trend_3h": temp_trend_3h,
    }
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
    hours: int,
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
    tolerance_hours = 1 if hours == 1 else 2
    if best["time_delta"] > pd.Timedelta(hours=tolerance_hours):
        return None
    return float(current_temp - best["temp_c"])


def update_state_with_observation(
    state: dict[str, Any],
    parsed: ParsedRealtimeMetar,
    profile: CityProfile,
) -> tuple[dict[str, Any], float, float | None, float | None]:
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
    temp_trend_1h = compute_temperature_trend(history_frame, parsed.valid, current_temp, hours=1)
    temp_trend_3h = compute_temperature_trend(history_frame, parsed.valid, current_temp, hours=3)
    current_valid_iso = parsed.valid.isoformat()
    if not any(item.get("valid") == current_valid_iso for item in history):
        history.append({"valid": current_valid_iso, "temp_c": current_temp})
    state["history"] = history[-96:]
    state["current_max_so_far"] = current_max_so_far
    state["last_valid_utc"] = parsed.valid.isoformat()
    return state, float(current_max_so_far), temp_trend_1h, temp_trend_3h


def expand_probability_vector(
    probabilities: list[float] | pd.Series,
    class_indices: list[int] | None,
    total_classes: int,
) -> list[float]:
    if class_indices is None:
        vector = list(probabilities)
        if len(vector) == total_classes:
            return vector
        padded = vector + [0.0] * max(0, total_classes - len(vector))
        return padded[:total_classes]
    expanded = [0.0] * total_classes
    for probability, class_index in zip(probabilities, class_indices, strict=False):
        expanded[int(class_index)] = float(probability)
    return expanded


def get_default_residual_grid() -> np.ndarray:
    return np.arange(
        0.0,
        TEMPERATURE_BIN_EXTENSION_C + TEMPERATURE_BIN_STEP_C,
        TEMPERATURE_BIN_STEP_C,
    )


def build_probability_table(
    current_max_so_far: float,
    residual_grid_c: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> pd.DataFrame:
    residual_grid = np.asarray(residual_grid_c, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    absolute_temps = current_max_so_far + residual_grid
    table = pd.DataFrame(
        {
            "temperature_c": absolute_temps,
            "probability": probs,
            "probability_percent": probs * 100.0,
        }
    )
    table = table.loc[table["temperature_c"] >= current_max_so_far].copy()
    table["temperature_c"] = table["temperature_c"].round(1)
    table["probability_percent"] = table["probability_percent"].round(1)
    table = table.sort_values(["probability", "temperature_c"], ascending=[False, True]).reset_index(drop=True)
    return table


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
            state, current_max_so_far, temp_trend_1h, temp_trend_3h = update_state_with_observation(state, parsed, profile)
            feature_row = build_feature_row(
                parsed=parsed,
                current_max_so_far=current_max_so_far,
                temp_trend_1h=temp_trend_1h,
                temp_trend_3h=temp_trend_3h,
                profile=profile,
            )
        expected_features = feature_row[expected_bundle["feature_columns"]]
        probability_features = feature_row[probability_bundle["feature_columns"]]
        expected_max_c = float(expected_bundle["pipeline"].predict(expected_features)[0])
        probability_vector_partial = probability_bundle["pipeline"].predict_proba(probability_features)[0]
        residual_grid = probability_bundle.get("residual_grid_c") or get_default_residual_grid().tolist()
        probability_vector = expand_probability_vector(
            probabilities=probability_vector_partial,
            class_indices=probability_bundle.get("class_indices"),
            total_classes=len(residual_grid),
        )
        probability_table_df = build_probability_table(
            current_max_so_far=current_max_so_far,
            residual_grid_c=residual_grid,
            probabilities=probability_vector,
        )
        top_row = probability_table_df.iloc[0] if not probability_table_df.empty else None
        record = {
            "valid_utc": parsed.valid.isoformat(),
            "valid_local": parsed.valid.tz_convert(ZoneInfo(profile.timezone_name)).isoformat(),
            "expected_max_c": round(expected_max_c, 2),
            "current_max_so_far": round(current_max_so_far, 2),
            "top_scenario_temp_c": round(float(top_row["temperature_c"]), 2) if top_row is not None else None,
            "top_scenario_probability_percent": round(float(top_row["probability_percent"]), 2) if top_row is not None else None,
            "probability_table": probability_table_df.to_dict(orient="records"),
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
        )
        return analysis, fair_values
