from __future__ import annotations

from paris_today_bot.models import BucketMarket, MarketSnapshot, StrategyDecision, WeatherSnapshot
from paris_today_bot.profile_loader import CityProfile


class ProfiledTodayStrategy:
    def __init__(self, profile: CityProfile) -> None:
        self.profile = profile
        self.rules = profile.payload

    def evaluate(self, weather: WeatherSnapshot, market: MarketSnapshot) -> StrategyDecision:
        reasons: list[str] = []
        local_hour = weather.now_utc.astimezone(self.profile.timezone).hour
        base_max = self._resolve_base_max(weather)
        if base_max is None:
            raise RuntimeError(f"Base model is unavailable; cannot price {self.profile.city_name} market.")

        rounded_base = round(base_max)
        adjustment = 0.0

        for rule in self.rules.get("scoring_model", {}).get("adjustments", []):
            delta = self._rule_delta(rule, weather, rounded_base, adjustment, local_hour)
            if delta != 0.0:
                adjustment += delta
                reasons.append(f"{rule['reason']}: {delta:+.2f}C")

        projected_max = round(rounded_base + adjustment)
        if weather.obs_max_so_far is not None and projected_max < weather.obs_max_so_far:
            reasons.append(
                f"Observed session high already reached {weather.obs_max_so_far}C, so projected max cannot be lower."
            )
            projected_max = weather.obs_max_so_far
        fair_values = self._build_market_fair_values(projected_max, weather)
        fair_by_question: dict[str, float] = {}
        for item in market.markets:
            fair_by_question[item.market_id] = self._market_probability(item, fair_values)
        return StrategyDecision(
            projected_max=projected_max,
            base_max=rounded_base,
            adjustment=adjustment,
            fair_values=fair_by_question,
            reasons=reasons,
        )

    def _resolve_base_max(self, weather: WeatherSnapshot) -> float | None:
        source = str(self.rules.get("scoring_model", {}).get("base_rule", {}).get("source", "GFS_daily_max")).upper()
        if source == "MODEL_CONSENSUS_MAX":
            return weather.model_consensus_max
        if source == "GFS_DAILY_MAX":
            return weather.gfs_daily_max
        if source == "ECMWF_DAILY_MAX":
            return weather.ecmwf_daily_max
        if source == "ENSEMBLE_DAILY_MAX":
            return weather.ensemble_daily_max
        raise RuntimeError(f"Unsupported base model source: {source}")

    def _rule_delta(
        self,
        rule: dict,
        weather: WeatherSnapshot,
        rounded_base: int,
        current_adjustment: float,
        local_hour: int,
    ) -> float:
        name = str(rule.get("name", ""))
        delta = float(rule.get("delta_c", 0.0))
        provisional_target = round(rounded_base + current_adjustment)

        if name == "ecmwf_warmer_than_gfs":
            if weather.ecmwf_daily_max is not None and weather.gfs_daily_max is not None and weather.ecmwf_daily_max >= weather.gfs_daily_max + 1.0:
                return delta
        elif name == "gfs_warmer_than_ecmwf":
            if weather.gfs_daily_max is not None and weather.ecmwf_daily_max is not None and weather.gfs_daily_max >= weather.ecmwf_daily_max + 1.0:
                return delta
        elif name.endswith("_regime"):
            expected_regime = name.removesuffix("_regime")
            if weather.day_regime == expected_regime:
                return delta
        elif name == "obs10_above_gfs10":
            if weather.max_by_10utc is not None and weather.gfs_10utc is not None and weather.max_by_10utc >= weather.gfs_10utc + 1.0:
                return delta
        elif name == "obs10_below_gfs10":
            if weather.max_by_10utc is not None and weather.gfs_10utc is not None and weather.max_by_10utc <= weather.gfs_10utc - 1.0:
                return delta
        elif name == "obs10_above_ecmwf10":
            if weather.max_by_10utc is not None and weather.ecmwf_10utc is not None and weather.max_by_10utc >= weather.ecmwf_10utc + 1.0:
                return delta
        elif name == "obs10_below_ecmwf10":
            if weather.max_by_10utc is not None and weather.ecmwf_10utc is not None and weather.max_by_10utc <= weather.ecmwf_10utc - 1.0:
                return delta
        elif name == "noon_underperformance":
            if local_hour >= 12 and weather.max_by_noon is not None and weather.max_by_noon <= provisional_target - 2.0:
                return delta
        elif name == "after_14_no_new_high_and_cooling":
            if local_hour >= 14 and weather.obs_current is not None and weather.obs_max_so_far is not None and weather.obs_current < weather.obs_max_so_far:
                return delta
        return 0.0

    def _build_market_fair_values(self, projected_max: int, weather: WeatherSnapshot) -> dict[int, float]:
        hour = weather.now_utc.astimezone(self.profile.timezone).hour
        templates = self.rules.get("fair_value_templates", {})
        timing_rules = self.rules.get("timing_rules", {})
        path_driven_hour = int(timing_rules.get("path_driven_start_hour_local", 12))
        late_decay_hour = int(timing_rules.get("late_day_decay_hour_utc", 14))
        harder_decay_hour = int(timing_rules.get("harder_decay_hour_utc", 15))

        if hour < path_driven_hour:
            regime_templates = templates.get("pre_noon", {})
            template = regime_templates.get(weather.day_regime) or regime_templates.get("mixed") or {"lower": 0.3, "primary": 0.55, "upper": 0.15}
            probs = {
                projected_max - 1: float(template["lower"]),
                projected_max: float(template["primary"]),
                projected_max + 1: float(template["upper"]),
            }
        else:
            current_high = weather.obs_max_so_far if weather.obs_max_so_far is not None else projected_max
            if hour >= harder_decay_hour and weather.obs_current is not None and weather.obs_current <= current_high - 1:
                template = templates.get("late_day_cooling") or {"lower": 0.05, "primary": 0.88, "upper": 0.07}
            elif weather.obs_current is not None and weather.obs_max_so_far is not None and weather.obs_current < weather.obs_max_so_far:
                template = templates.get("post_noon_cooling") or {"lower": 0.07, "primary": 0.83, "upper": 0.10}
            else:
                template = templates.get("post_noon_default") or {"lower": 0.12, "primary": 0.73, "upper": 0.15}
            probs = {
                current_high - 1: float(template["lower"]),
                current_high: float(template["primary"]),
                current_high + 1: float(template["upper"]),
            }

        if weather.upper_realistic_max is not None:
            upper_cap = max(projected_max + 1, round(weather.upper_realistic_max))
            probs = {bucket: value for bucket, value in probs.items() if bucket <= upper_cap}

        if weather.obs_max_so_far is not None:
            probs = {
                bucket: value
                for bucket, value in probs.items()
                if bucket >= weather.obs_max_so_far
            }
            if probs and weather.obs_max_so_far not in probs and weather.obs_max_so_far >= min(probs):
                probs[weather.obs_max_so_far] = 1.0

        total = sum(probs.values())
        if total <= 0:
            fallback = weather.obs_max_so_far if weather.obs_max_so_far is not None else projected_max
            return {fallback: 1.0}
        return {bucket: value / total for bucket, value in probs.items()}

    def _market_probability(self, market: BucketMarket, bucket_probs: dict[int, float]) -> float:
        if market.temperature_c is None:
            return 0.0
        if market.tail == "exact":
            return bucket_probs.get(market.temperature_c, 0.0)
        if market.tail == "or_higher":
            return sum(prob for bucket, prob in bucket_probs.items() if bucket >= market.temperature_c)
        if market.tail == "or_lower":
            return sum(prob for bucket, prob in bucket_probs.items() if bucket <= market.temperature_c)
        return 0.0
