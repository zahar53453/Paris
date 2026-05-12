from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from paris_today_bot.models import BucketMarket, MarketSnapshot
from paris_today_bot.profile_loader import CityProfile


TEMP_RE = re.compile(r"\b(\d+)\s*[^0-9A-Za-z]{0,2}C\b", re.IGNORECASE)


class CityMarketClient:
    def __init__(self, cfg: object, profile: CityProfile) -> None:
        self.cfg = cfg
        self.profile = profile

    async def fetch_today_market(self) -> MarketSnapshot:
        target_date = datetime.now(UTC).date().isoformat()
        url = f"{self.cfg.gamma_api_url}/events?limit=200&offset=0&tag_slug=weather&active=true&closed=false"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            events = response.json()
        matches: list[dict] = []
        for event in events:
            title = str(event.get("title", ""))
            if "highest temperature" not in title.lower():
                continue
            if self.profile.city_name.lower() not in title.lower():
                continue
            event_date = self._extract_title_date(title)
            if event_date and event_date != target_date:
                continue
            matches.append(event)
        if not matches:
            raise RuntimeError(f"No active {self.profile.city_name} weather event found for today.")
        event = matches[0]
        markets = []
        raw_markets = event.get("markets", [])
        for raw_market in raw_markets:
            parsed = self._parse_market(raw_market)
            if parsed is not None:
                markets.append(parsed)
        if not markets:
            raise RuntimeError(f"{self.profile.city_name} event found, but no parseable temperature bucket markets were available.")
        await self._hydrate_live_orderbooks(markets)
        return MarketSnapshot(
            event_title=str(event.get("title", "")),
            event_slug=str(event.get("slug", "")),
            target_date=target_date,
            markets=markets,
        )

    def _extract_title_date(self, title: str) -> str | None:
        m = re.search(r"on ([A-Za-z]+ \d{1,2}, \d{4})", title)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat()
        except ValueError:
            return None

    def _parse_market(self, market: dict) -> BucketMarket | None:
        question = str(market.get("question", ""))
        token_ids = market.get("clobTokenIds", [])
        outcomes = market.get("outcomes", [])
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not token_ids:
            return None
        yes_token_id = str(token_ids[0])
        temp_match = TEMP_RE.search(question)
        temp_c = int(temp_match.group(1)) if temp_match else None
        tail = "exact"
        lower_q = question.lower()
        if "or higher" in lower_q or "or above" in lower_q:
            tail = "or_higher"
        elif "or lower" in lower_q or "or below" in lower_q:
            tail = "or_lower"
        best_ask = self._coerce_float(market.get("bestAsk"))
        best_bid = self._coerce_float(market.get("bestBid"))
        midpoint = None
        if best_ask is not None and best_bid is not None:
            midpoint = (best_ask + best_bid) / 2.0
        return BucketMarket(
            market_id=str(market.get("conditionId") or market.get("id") or question),
            question=question,
            slug=str(market.get("slug", "")),
            token_id=yes_token_id,
            no_token_id=str(token_ids[1]) if len(token_ids) > 1 else None,
            best_ask=best_ask,
            best_bid=best_bid,
            midpoint=midpoint,
            temperature_c=temp_c,
            tail=tail,
        )

    async def _hydrate_live_orderbooks(self, markets: list[BucketMarket]) -> None:
        async with httpx.AsyncClient(timeout=8.0) as client:
            token_ids = {
                token_id
                for market in markets
                for token_id in [market.token_id, market.no_token_id]
                if token_id
            }
            tasks = {token_id: self._fetch_book(client, token_id) for token_id in token_ids}
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        books_by_token: dict[str, dict[str, float | None]] = {}
        for token_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                continue
            books_by_token[token_id] = result

        for market in markets:
            yes_book = books_by_token.get(market.token_id)
            no_book = books_by_token.get(market.no_token_id or "")
            if yes_book is not None:
                market.best_ask = yes_book["best_ask"]
                market.best_bid = yes_book["best_bid"]
                market.midpoint = yes_book["midpoint"]
            if no_book is not None:
                market.no_best_ask = no_book["best_ask"]
                market.no_best_bid = no_book["best_bid"]
                market.no_midpoint = no_book["midpoint"]

    async def _fetch_book(self, client: httpx.AsyncClient, token_id: str) -> dict[str, float | None]:
        response = await client.get(f"{self.cfg.clob_api_url}/book", params={"token_id": token_id})
        response.raise_for_status()
        payload = response.json()
        asks = payload.get("asks") or []
        bids = payload.get("bids") or []
        best_ask = self._best_price(asks, lowest=True)
        best_bid = self._best_price(bids, lowest=False)
        midpoint = None
        if best_ask is not None and best_bid is not None:
            midpoint = (best_ask + best_bid) / 2.0
        return {
            "best_ask": best_ask,
            "best_bid": best_bid,
            "midpoint": midpoint,
        }

    def _best_price(self, levels: list[dict[str, Any]], *, lowest: bool) -> float | None:
        prices: list[float] = []
        for level in levels:
            try:
                prices.append(float(level["price"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not prices:
            return None
        return min(prices) if lowest else max(prices)

    def _coerce_float(self, value: object) -> float | None:
        try:
            if value in (None, "", "null"):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
