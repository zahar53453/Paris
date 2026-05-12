from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from paris_today_bot.models import MetarObservation, WeatherSnapshot
from paris_today_bot.profile_loader import CityProfile


def _cover_from_raw(raw: str) -> str:
    upper = raw.upper()
    if "CAVOK" in upper or " CLR" in upper or " NSC" in upper:
        return "clear"
    if " OVC" in upper:
        return "ovc"
    if " BKN" in upper:
        return "bkn"
    if " SCT" in upper or " FEW" in upper:
        return "scattered"
    return "unknown"


def _wx_from_raw(raw: str) -> str:
    upper = raw.upper()
    if "SHRA" in upper or "SHDZ" in upper:
        return "showers"
    if "RA" in upper and "DZ" in upper:
        return "radz"
    if "RA" in upper:
        return "rain"
    if "DZ" in upper:
        return "drizzle"
    if " FG" in upper:
        return "fog"
    if " BR" in upper:
        return "mist"
    return ""


@dataclass(slots=True)
class WeatherDataClient:
    timeout: float = 20.0

    async def fetch_today(self, profile: CityProfile, snapshot_file: str | None = None) -> WeatherSnapshot:
        if snapshot_file:
            payload = json.loads(Path(snapshot_file).read_text(encoding="utf-8"))
            return self._snapshot_from_archive(profile, payload)
        return await self._snapshot_from_live(profile)

    def _snapshot_from_archive(self, profile: CityProfile, payload: dict[str, Any]) -> WeatherSnapshot:
        metar_items = sorted(payload.get("metar", []), key=lambda item: item["obsTime"])
        observations = [
            MetarObservation(
                observed_at=datetime.fromtimestamp(item["obsTime"], UTC),
                temp_c=int(item["temp"]),
                raw=item.get("rawOb", ""),
                cover=(item.get("cover") or _cover_from_raw(item.get("rawOb", ""))).lower(),
                wx=(item.get("wxString") or _wx_from_raw(item.get("rawOb", ""))).lower(),
            )
            for item in metar_items
            if item.get("temp") is not None
        ]
        det_section = payload.get("open_meteo_deterministic", {}) or {}
        ens_section = payload.get("open_meteo_ensemble", {}) or {}
        hourly_det = det_section.get("hourly", {})
        daily_det = det_section.get("daily", {})
        hourly_ens = ens_section.get("hourly", {})
        daily_ens = ens_section.get("daily", {})
        now_utc = observations[-1].observed_at if observations else datetime.now(UTC)
        return self._build_snapshot(profile, now_utc, observations, hourly_det, daily_det, hourly_ens, daily_ens)

    async def _snapshot_from_live(self, profile: CityProfile) -> WeatherSnapshot:
        icao = profile.icao
        lat, lon = profile.latitude, profile.longitude
        metar_url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours=24"
        ens_url = (
            "https://ensemble-api.open-meteo.com/v1/ensemble"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min"
            "&models=ecmwf_ifs025"
            "&forecast_days=2&timezone=GMT"
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            metar_resp, ecmwf_resp, gfs_resp, ens_resp = await asyncio.gather(
                client.get(metar_url),
                client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "hourly": "temperature_2m,precipitation",
                        "daily": "temperature_2m_max,temperature_2m_min",
                        "models": "ecmwf_ifs025",
                        "forecast_days": 2,
                        "timezone": "GMT",
                    },
                ),
                client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "hourly": "temperature_2m,precipitation",
                        "daily": "temperature_2m_max,temperature_2m_min",
                        "models": "gfs_seamless",
                        "forecast_days": 2,
                        "timezone": "GMT",
                    },
                ),
                client.get(ens_url),
            )
        metar_resp.raise_for_status()
        ecmwf_resp.raise_for_status()
        gfs_resp.raise_for_status()

        metar_items = sorted(metar_resp.json(), key=lambda item: item["obsTime"])
        observations = [
            MetarObservation(
                observed_at=datetime.fromtimestamp(item["obsTime"], UTC),
                temp_c=int(item["temp"]),
                raw=item.get("rawOb", ""),
                cover=(item.get("cover") or _cover_from_raw(item.get("rawOb", ""))).lower(),
                wx=(item.get("wxString") or _wx_from_raw(item.get("rawOb", ""))).lower(),
            )
            for item in metar_items
            if item.get("temp") is not None
        ]
        now_utc = datetime.now(UTC)
        ensemble_hourly: dict[str, Any] = {}
        ensemble_daily: dict[str, Any] = {}
        if ens_resp.status_code == 200:
            ensemble_daily = ens_resp.json().get("daily", {})
            ensemble_hourly = self._synthetic_ensemble_hourly_from_daily(ensemble_daily)
        det_hourly = self._merge_live_deterministic_hourly(
            ecmwf_resp.json(),
            gfs_resp.json(),
        )
        det_daily = self._merge_live_deterministic_daily(
            ecmwf_resp.json(),
            gfs_resp.json(),
        )
        return self._build_snapshot(profile, now_utc, observations, det_hourly, det_daily, ensemble_hourly, ensemble_daily)

    def _merge_live_deterministic_hourly(
        self,
        ecmwf_payload: dict[str, Any],
        gfs_payload: dict[str, Any],
    ) -> dict[str, Any]:
        ecmwf_hourly = ecmwf_payload.get("hourly", {}) or {}
        gfs_hourly = gfs_payload.get("hourly", {}) or {}
        times = ecmwf_hourly.get("time") or gfs_hourly.get("time") or []
        return {
            "time": times,
            "temperature_2m_ecmwf_ifs025": ecmwf_hourly.get("temperature_2m", []) or [],
            "temperature_2m_gfs_seamless": gfs_hourly.get("temperature_2m", []) or [],
            "precipitation_ecmwf_ifs025": ecmwf_hourly.get("precipitation", []) or [],
            "precipitation_gfs_seamless": gfs_hourly.get("precipitation", []) or [],
        }

    def _merge_live_deterministic_daily(
        self,
        ecmwf_payload: dict[str, Any],
        gfs_payload: dict[str, Any],
    ) -> dict[str, Any]:
        ecmwf_daily = ecmwf_payload.get("daily", {}) or {}
        gfs_daily = gfs_payload.get("daily", {}) or {}
        times = ecmwf_daily.get("time") or gfs_daily.get("time") or []
        return {
            "time": times,
            "temperature_2m_max_ecmwf_ifs025": ecmwf_daily.get("temperature_2m_max", []) or [],
            "temperature_2m_max_gfs_seamless": gfs_daily.get("temperature_2m_max", []) or [],
        }

    def _synthetic_ensemble_hourly_from_daily(self, daily: dict[str, Any]) -> dict[str, Any]:
        times = []
        temps = []
        days = daily.get("time", []) or []
        maxs = (
            daily.get("temperature_2m_max_mean")
            or daily.get("temperature_2m_max")
            or []
        )
        for day, value in zip(days, maxs, strict=False):
            if value is None:
                continue
            times.append(f"{day}T14:00")
            temps.append(float(value))
        return {"time": times, "temperature_2m": temps}

    def _build_snapshot(
        self,
        profile: CityProfile,
        now_utc: datetime,
        observations: list[MetarObservation],
        det_hourly: dict[str, Any],
        det_daily: dict[str, Any],
        ens_hourly: dict[str, Any],
        ens_daily: dict[str, Any],
    ) -> WeatherSnapshot:
        local_today = now_utc.astimezone(profile.timezone).date()
        today_obs = [obs for obs in observations if obs.observed_at.astimezone(profile.timezone).date() == local_today]
        obs_current = today_obs[-1].temp_c if today_obs else None
        obs_max_so_far = max((obs.temp_c for obs in today_obs), default=None)

        def hour_dict(times: list[str], values: list[Any]) -> dict[str, float]:
            out: dict[str, float] = {}
            for t, value in zip(times, values, strict=False):
                if value is None:
                    continue
                dt = datetime.fromisoformat(t).replace(tzinfo=UTC) if "T" in t else datetime.fromisoformat(f"{local_today.isoformat()}T{t}").replace(tzinfo=UTC)
                local_ts = dt.astimezone(profile.timezone)
                if local_ts.date() != local_today:
                    continue
                out[local_ts.strftime("%H:%M")] = float(value)
            return out

        times = det_hourly.get("time", []) or []
        gfs_hourly = hour_dict(times, det_hourly.get("temperature_2m_gfs_seamless", []) or [])
        ecmwf_hourly = hour_dict(times, det_hourly.get("temperature_2m_ecmwf_ifs025", []) or [])
        ens_hourly = hour_dict(ens_hourly.get("time", []) or [], ens_hourly.get("temperature_2m", []) or [])
        today_key = local_today.isoformat()

        gfs_daily_max = self._extract_daily_value(
            det_daily,
            today_key,
            ["temperature_2m_max_gfs_seamless", "temperature_2m_max_gfs"],
        )
        ecmwf_daily_max = self._extract_daily_value(
            det_daily,
            today_key,
            ["temperature_2m_max_ecmwf_ifs025", "temperature_2m_max_ecmwf"],
        )
        ensemble_daily_max = self._extract_daily_value(
            ens_daily,
            today_key,
            ["temperature_2m_max_mean", "temperature_2m_max"],
        )
        ensemble_daily_spread = self._extract_daily_value(
            ens_daily,
            today_key,
            ["temperature_2m_max_spread", "temperature_2m_max_stddev"],
        )

        if gfs_daily_max is None and gfs_hourly:
            gfs_daily_max = max(gfs_hourly.values())
        if ecmwf_daily_max is None and ecmwf_hourly:
            ecmwf_daily_max = max(ecmwf_hourly.values())
        if ensemble_daily_max is None and ens_hourly:
            ensemble_daily_max = max(ens_hourly.values())

        consensus_inputs = [value for value in [ensemble_daily_max, ecmwf_daily_max, gfs_daily_max] if value is not None]
        model_consensus_max = mean(consensus_inputs) if consensus_inputs else None
        model_agreement_spread = (max(consensus_inputs) - min(consensus_inputs)) if len(consensus_inputs) >= 2 else 0.0 if consensus_inputs else None
        if ensemble_daily_max is not None:
            spread_component = 0.84 * ensemble_daily_spread if ensemble_daily_spread is not None else 0.5 * (model_agreement_spread or 0.0)
            upper_realistic_max = ensemble_daily_max + max(0.0, spread_component)
        elif model_consensus_max is not None:
            upper_realistic_max = model_consensus_max + 0.5 * (model_agreement_spread or 0.0)
        else:
            upper_realistic_max = None

        morning_obs = [obs for obs in today_obs if 8 <= obs.observed_at.astimezone(profile.timezone).hour <= 11]
        day_obs = [obs for obs in today_obs if 8 <= obs.observed_at.astimezone(profile.timezone).hour <= 14]
        morning_clear = sum(obs.cover == "clear" for obs in morning_obs)
        morning_cloud = sum(obs.cover in {"bkn", "ovc"} for obs in morning_obs)
        morning_rain = sum(obs.wx in {"rain", "drizzle", "showers", "radz"} for obs in morning_obs)
        morning_fog = sum(obs.wx in {"fog", "mist"} for obs in morning_obs)
        rain_count_8_14 = sum(obs.wx in {"rain", "drizzle", "showers", "radz"} for obs in day_obs)
        regime = self._classify_regime(
            profile=profile,
            morning_clear=morning_clear,
            morning_cloud=morning_cloud,
            morning_rain=morning_rain,
            morning_fog=morning_fog,
            rain_count_8_14=rain_count_8_14,
        )

        max_10utc = max((obs.temp_c for obs in today_obs if obs.observed_at.astimezone(profile.timezone).hour == 10), default=None)
        max_by_noon = max((obs.temp_c for obs in today_obs if obs.observed_at.astimezone(profile.timezone).hour <= 12), default=None)

        return WeatherSnapshot(
            now_utc=now_utc,
            observations=today_obs,
            obs_current=obs_current,
            obs_max_so_far=obs_max_so_far,
            max_by_10utc=max_10utc,
            max_by_noon=max_by_noon,
            day_regime=regime,
            morning_clear_count=morning_clear,
            morning_cloud_count=morning_cloud,
            morning_rain_count=morning_rain,
            morning_fog_count=morning_fog,
            rain_count_8_14=rain_count_8_14,
            gfs_hourly=gfs_hourly,
            ecmwf_hourly=ecmwf_hourly,
            ensemble_hourly=ens_hourly,
            gfs_daily_max=gfs_daily_max,
            ecmwf_daily_max=ecmwf_daily_max,
            ensemble_daily_max=ensemble_daily_max,
            ensemble_daily_spread=ensemble_daily_spread,
            model_consensus_max=model_consensus_max,
            upper_realistic_max=upper_realistic_max,
            model_agreement_spread=model_agreement_spread,
            gfs_10utc=gfs_hourly.get("10:00"),
            ecmwf_10utc=ecmwf_hourly.get("10:00"),
            model_consensus_10utc=mean([value for value in [gfs_hourly.get("10:00"), ecmwf_hourly.get("10:00")] if value is not None]) if any(value is not None for value in [gfs_hourly.get("10:00"), ecmwf_hourly.get("10:00")]) else None,
        )

    def _extract_daily_value(
        self,
        payload: dict[str, Any],
        target_day: str,
        preferred_keys: list[str],
    ) -> float | None:
        if not payload:
            return None
        times = payload.get("time", []) or []
        if not times:
            return None
        idx = times.index(target_day) if target_day in times else 0
        keys = [key for key in preferred_keys if key in payload]
        for key in keys:
            values = payload.get(key)
            if not values or idx >= len(values):
                continue
            value = values[idx]
            if value is None:
                continue
            return float(value)
        return None

    def _classify_regime(
        self,
        profile: CityProfile,
        morning_clear: int,
        morning_cloud: int,
        morning_rain: int,
        morning_fog: int,
        rain_count_8_14: int,
    ) -> str:
        regimes = profile.payload.get("day_regimes", {})

        rainy = regimes.get("rainy", {}).get("detection", {})
        if rain_count_8_14 >= int(rainy.get("min_rain_observations", 999)):
            return "rainy"

        foggy = regimes.get("foggy", {}).get("detection", {})
        if morning_fog >= int(foggy.get("min_fog_observations", 999)):
            return "foggy"

        clear = regimes.get("clear", {}).get("detection", {})
        if (
            morning_clear >= int(clear.get("min_clear_observations", 999))
            and morning_cloud <= int(clear.get("max_cloudy_observations", -1))
            and morning_rain <= int(clear.get("max_rain_observations", -1))
            and morning_fog <= int(clear.get("max_fog_observations", 999))
        ):
            return "clear"

        cloudy = regimes.get("cloudy", {}).get("detection", {})
        if morning_cloud >= int(cloudy.get("min_cloudy_observations", 999)):
            return "cloudy"

        return "mixed"
