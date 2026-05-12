from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from paris_today_bot.config import BotConfig
from paris_today_bot.models import BucketMarket, TradeAction
from paris_today_bot.profile_loader import CityProfile


@dataclass(slots=True)
class PaperTrade:
    id: str
    status: str
    profile_slug: str
    city_name: str
    icao: str
    market_id: str
    question: str
    token_id: str
    side: str
    entry_price: float
    size_usd: float
    shares: float
    entry_edge: float
    entry_fair: float
    opened_at: str
    closed_at: str | None = None
    exit_price: float | None = None
    exit_edge: float | None = None
    realized_pnl: float | None = None
    close_reason: str | None = None
    last_price: float | None = None
    last_fair: float | None = None
    last_edge: float | None = None
    last_unrealized_pnl: float | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PaperStore:
    def __init__(self, path: Path, start_balance_usd: float) -> None:
        self.path = path
        self.start_balance_usd = start_balance_usd

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "start_balance_usd": self.start_balance_usd,
                "trades": [],
                "events": [],
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload.setdefault("start_balance_usd", self.start_balance_usd)
        payload.setdefault("trades", [])
        payload.setdefault("events", [])
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_trades(self) -> list[PaperTrade]:
        return [PaperTrade(**item) for item in self.load().get("trades", [])]

    def save_trades(self, trades: list[PaperTrade], events: list[dict[str, Any]]) -> None:
        payload = self.load()
        payload["trades"] = [asdict(trade) for trade in trades]
        payload["events"] = payload.get("events", []) + events
        self.save(payload)


class PaperBroker:
    def __init__(self, cfg: BotConfig, store: PaperStore) -> None:
        self.cfg = cfg
        self.store = store

    def process_profile(
        self,
        profile: CityProfile,
        markets: Iterable[BucketMarket],
        fair_values: dict[str, float],
        actions: Iterable[TradeAction],
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        trades = self.store.load_trades()
        events: list[dict[str, Any]] = []
        market_by_id = {market.market_id: market for market in markets}
        open_trades = [
            trade
            for trade in trades
            if trade.status == "OPEN" and trade.profile_slug == profile.slug
        ]
        open_token_ids = {trade.token_id for trade in open_trades}

        closed = self._close_invalidated(open_trades, market_by_id, fair_values, now, events)
        opened = self._open_new(profile, trades, actions, market_by_id, fair_values, open_token_ids, now, events)

        if events:
            self.store.save_trades(trades, events)

        return {
            "opened": [asdict(trade) for trade in opened],
            "closed": [asdict(trade) for trade in closed],
            "open_count": len([trade for trade in trades if trade.status == "OPEN"]),
            "realized_pnl": round(sum(float(trade.realized_pnl or 0.0) for trade in trades), 4),
        }

    def mark_to_market(self, profile: CityProfile, markets: Iterable[BucketMarket]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        trades = self.store.load_trades()
        market_by_id = {market.market_id: market for market in markets}
        updated = 0
        for trade in trades:
            if trade.status != "OPEN" or trade.profile_slug != profile.slug:
                continue
            market = market_by_id.get(trade.market_id)
            if market is None:
                continue
            exit_price = self._exit_price_for_side(market, trade.side)
            if exit_price is None:
                continue
            trade.last_price = exit_price
            trade.last_unrealized_pnl = (exit_price - trade.entry_price) * trade.shares
            trade.updated_at = now
            updated += 1
        if updated:
            self.store.save_trades(trades, [])
        return {"updated": updated}

    def _close_invalidated(
        self,
        open_trades: list[PaperTrade],
        market_by_id: dict[str, BucketMarket],
        fair_values: dict[str, float],
        now: str,
        events: list[dict[str, Any]],
    ) -> list[PaperTrade]:
        closed: list[PaperTrade] = []
        for trade in open_trades:
            market = market_by_id.get(trade.market_id)
            if market is None:
                continue

            fair_yes = fair_values.get(trade.market_id, 0.0)
            fair = fair_yes if trade.side == "YES" else 1.0 - fair_yes
            current_buy_price = self._entry_price_for_side(market, trade.side)
            current_exit_price = self._exit_price_for_side(market, trade.side)
            if current_buy_price is None or current_exit_price is None:
                continue

            current_edge = fair - current_buy_price
            unrealized_pnl = (current_exit_price - trade.entry_price) * trade.shares
            trade.last_price = current_exit_price
            trade.last_fair = fair
            trade.last_edge = current_edge
            trade.last_unrealized_pnl = unrealized_pnl
            trade.updated_at = now

            should_close = current_edge <= self.cfg.paper_close_edge
            if not should_close:
                continue

            trade.status = "CLOSED"
            trade.closed_at = now
            trade.exit_price = current_exit_price
            trade.exit_edge = current_edge
            trade.realized_pnl = unrealized_pnl
            trade.close_reason = "Target edge disappeared or became negative."
            closed.append(trade)
            events.append(self._event("CLOSE", trade, now))
        return closed

    def _open_new(
        self,
        profile: CityProfile,
        trades: list[PaperTrade],
        actions: Iterable[TradeAction],
        market_by_id: dict[str, BucketMarket],
        fair_values: dict[str, float],
        open_token_ids: set[str],
        now: str,
        events: list[dict[str, Any]],
    ) -> list[PaperTrade]:
        opened: list[PaperTrade] = []
        for action in actions:
            if action.action not in {"BUY_YES", "BUY_NO"}:
                continue
            if action.token_id in open_token_ids:
                continue
            market = market_by_id.get(action.market_id)
            if market is None:
                continue

            side = "YES" if action.action == "BUY_YES" else "NO"
            price = self._entry_price_for_side(market, side)
            if price is None or price <= 0:
                continue

            fair_yes = fair_values.get(action.market_id, 0.0)
            fair = fair_yes if side == "YES" else 1.0 - fair_yes
            edge = fair - price
            if edge < self.cfg.min_edge_to_open:
                continue

            size_usd = self._position_size(edge, price, market)
            trade = PaperTrade(
                id=uuid4().hex,
                status="OPEN",
                profile_slug=profile.slug,
                city_name=profile.city_name,
                icao=profile.icao,
                market_id=action.market_id,
                question=action.question,
                token_id=action.token_id,
                side=side,
                entry_price=price,
                size_usd=size_usd,
                shares=size_usd / price,
                entry_edge=edge,
                entry_fair=fair,
                opened_at=now,
                last_price=price,
                last_fair=fair,
                last_edge=edge,
                last_unrealized_pnl=0.0,
                updated_at=now,
                metadata={
                    "model_reason": action.reason,
                    "market_spread": self._spread_for_side(market, side),
                },
            )
            trades.append(trade)
            opened.append(trade)
            open_token_ids.add(action.token_id)
            events.append(self._event("OPEN", trade, now))
        return opened

    def _position_size(self, edge: float, price: float, market: BucketMarket) -> float:
        if edge >= 0.35:
            base = self.cfg.paper_max_trade_usd
        elif edge >= 0.20:
            base = min(self.cfg.paper_max_trade_usd, 10.0)
        elif edge >= 0.12:
            base = min(self.cfg.paper_max_trade_usd, 5.0)
        else:
            base = self.cfg.paper_min_trade_usd

        spread = min(
            item
            for item in [self._spread_for_side(market, "YES"), self._spread_for_side(market, "NO")]
            if item is not None
        ) if self._spread_for_side(market, "YES") is not None or self._spread_for_side(market, "NO") is not None else None
        if spread is not None and spread >= 0.10:
            base *= 0.5
        if price <= 0.02:
            base *= 0.5
        return round(max(self.cfg.paper_min_trade_usd, min(self.cfg.paper_max_trade_usd, base)), 2)

    def _entry_price_for_side(self, market: BucketMarket, side: str) -> float | None:
        if side == "YES":
            return market.best_ask if market.best_ask is not None else market.midpoint
        return market.no_best_ask if market.no_best_ask is not None else market.no_midpoint

    def _exit_price_for_side(self, market: BucketMarket, side: str) -> float | None:
        if side == "YES":
            return market.best_bid
        return market.no_best_bid

    def _spread_for_side(self, market: BucketMarket, side: str) -> float | None:
        if side == "YES" and market.best_ask is not None and market.best_bid is not None:
            return max(0.0, market.best_ask - market.best_bid)
        if side == "NO" and market.no_best_ask is not None and market.no_best_bid is not None:
            return max(0.0, market.no_best_ask - market.no_best_bid)
        return None

    def _event(self, event_type: str, trade: PaperTrade, timestamp: str) -> dict[str, Any]:
        return {
            "type": event_type,
            "timestamp": timestamp,
            "trade_id": trade.id,
            "token_id": trade.token_id,
            "city_name": trade.city_name,
            "side": trade.side,
            "price": trade.exit_price if event_type == "CLOSE" else trade.entry_price,
            "size_usd": trade.size_usd,
            "pnl": trade.realized_pnl,
            "reason": trade.close_reason,
        }


class PaperReporter:
    def __init__(self, cfg: BotConfig, store: PaperStore) -> None:
        self.cfg = cfg
        self.store = store

    def summary(self) -> dict[str, Any]:
        trades = self.store.load_trades()
        open_trades = [trade for trade in trades if trade.status == "OPEN"]
        closed_trades = [trade for trade in trades if trade.status == "CLOSED"]
        realized = sum(float(trade.realized_pnl or 0.0) for trade in closed_trades)
        unrealized = sum(float(trade.last_unrealized_pnl or 0.0) for trade in open_trades)
        exposure = sum(float(trade.size_usd or 0.0) for trade in open_trades)
        return {
            "start_balance_usd": self.store.load().get("start_balance_usd", self.cfg.paper_start_balance_usd),
            "realized_pnl": round(realized, 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_pnl": round(realized + unrealized, 4),
            "open_exposure_usd": round(exposure, 2),
            "open_count": len(open_trades),
            "closed_count": len(closed_trades),
        }

    def open_trades(self) -> list[dict[str, Any]]:
        return [asdict(trade) for trade in self.store.load_trades() if trade.status == "OPEN"]

    def closed_trades(self, limit: int = 30) -> list[dict[str, Any]]:
        trades = [trade for trade in self.store.load_trades() if trade.status == "CLOSED"]
        trades.sort(key=lambda trade: trade.closed_at or "")
        return [asdict(trade) for trade in trades[-limit:]]

    def balance_text(self) -> str:
        item = self.summary()
        equity = float(item["start_balance_usd"]) + float(item["total_pnl"])
        return (
            "Paper balance\n\n"
            f"Start balance: ${item['start_balance_usd']:.2f}\n"
            f"Current equity: ${equity:.2f}\n"
            f"Realized PnL: {item['realized_pnl']:+.2f}$\n"
            f"Unrealized PnL: {item['unrealized_pnl']:+.2f}$\n"
            f"Open exposure: ${item['open_exposure_usd']:.2f}\n"
            f"Open trades: {item['open_count']}\n"
            f"Closed trades: {item['closed_count']}"
        )

    def open_trades_text(self) -> str:
        trades = [PaperTrade(**item) for item in self.open_trades()]
        total_unrealized = sum(float(trade.last_unrealized_pnl or 0.0) for trade in trades)
        if not trades:
            return "Open paper trades: none."
        lines = [f"Open paper trades\nUnrealized PnL: {total_unrealized:+.2f}$\n"]
        for trade in trades:
            lines.append(
                f"{trade.city_name} | {trade.side} | {self._short_question(trade.question)}\n"
                f"Entry: {trade.shares:.2f} sh @ {trade.entry_price:.3f} | Size: ${trade.size_usd:.2f}\n"
                f"Now: {(trade.last_price or 0.0):.3f} | Edge: {(trade.last_edge or 0.0):+.3f} | PnL: {(trade.last_unrealized_pnl or 0.0):+.2f}$\n"
            )
        return "\n".join(lines)

    def closed_trades_text(self, limit: int = 30) -> str:
        trades = [PaperTrade(**item) for item in self.closed_trades(limit=limit)]
        total_realized = sum(float(trade.realized_pnl or 0.0) for trade in self.store.load_trades() if trade.status == "CLOSED")
        if not trades:
            return f"Closed paper trades: none.\nRealized PnL: {total_realized:+.2f}$"
        lines = [f"Closed paper trades\nTotal realized PnL: {total_realized:+.2f}$\n"]
        for trade in trades:
            lines.append(
                f"{trade.city_name} | {trade.side} | {self._short_question(trade.question)}\n"
                f"Entry: {trade.entry_price:.3f} -> Exit: {(trade.exit_price or 0.0):.3f} | "
                f"PnL: {(trade.realized_pnl or 0.0):+.2f}$\n"
                f"Reason: {trade.close_reason or 'n/a'}\n"
            )
        return "\n".join(lines)

    def _short_question(self, question: str) -> str:
        return question if len(question) <= 90 else question[:87] + "..."
