from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from paris_today_bot.config import BotConfig
from paris_today_bot.models import BucketMarket, TradeAction
from paris_today_bot.profile_loader import CityProfile
from paris_today_bot.runtime_log import log_runtime


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

    def reset(self) -> None:
        self.save(
            {
                "start_balance_usd": self.start_balance_usd,
                "trades": [],
                "events": [],
            }
        )


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
        local_today = datetime.now(UTC).astimezone(profile.timezone).date()
        trades = self.store.load_trades()
        events: list[dict[str, Any]] = []
        market_by_id = {market.market_id: market for market in markets}
        open_trades = [
            trade
            for trade in trades
            if trade.status == "OPEN" and trade.profile_slug == profile.slug
        ]
        open_token_ids = {trade.token_id for trade in open_trades}

        expired = self._close_expired(open_trades, local_today, now, events)
        take_profit = self._close_take_profit(open_trades, market_by_id, now, events)
        untradable = self._close_untradable(open_trades, market_by_id, now, events)
        closed = self._close_invalidated(open_trades, market_by_id, fair_values, now, events)
        opened = self._open_new(profile, trades, actions, market_by_id, fair_values, open_token_ids, now, events)

        if events:
            self.store.save_trades(trades, events)

        return {
            "opened": [asdict(trade) for trade in opened],
            "closed": [asdict(trade) for trade in [*expired, *take_profit, *untradable, *closed]],
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

    def mark_to_market_by_books(
        self,
        profile: CityProfile,
        books_by_token: dict[str, dict[str, float | None]],
        closed_market_states: dict[str, dict[str, Any]] | None = None,
        now: str | None = None,
    ) -> dict[str, Any]:
        timestamp = now or datetime.now(UTC).isoformat()
        trades = self.store.load_trades()
        updated = 0
        closed = 0
        events: list[dict[str, Any]] = []
        for trade in trades:
            if trade.status != "OPEN" or trade.profile_slug != profile.slug:
                continue
            book = books_by_token.get(trade.token_id)
            state = (closed_market_states or {}).get(trade.market_id, {})
            resolved_price = self._resolved_price_for_token(state, trade.token_id)
            if resolved_price is not None:
                trade.last_price = resolved_price
                trade.last_unrealized_pnl = (resolved_price - trade.entry_price) * trade.shares
                trade.updated_at = timestamp
                trade.status = "CLOSED"
                trade.closed_at = timestamp
                trade.exit_price = resolved_price
                trade.exit_edge = trade.last_edge
                trade.realized_pnl = trade.last_unrealized_pnl
                trade.close_reason = "Market resolved; paper trade settled from final outcome prices."
                events.append(self._event("CLOSE", trade, timestamp))
                updated += 1
                closed += 1
                log_runtime(
                    f"[paper] resolved trade closed city={trade.city_name} side={trade.side} "
                    f"question={trade.question} final={resolved_price:.3f}"
                )
                continue
            if book is None:
                log_runtime(
                    f"[paper] no refresh data city={trade.city_name} side={trade.side} "
                    f"question={trade.question} token={trade.token_id[:10]}"
                )
                continue
            exit_price = book.get("best_bid")
            entry_price_now = book.get("best_ask")
            if exit_price is None:
                continue
            trade.last_price = float(exit_price)
            trade.last_unrealized_pnl = (float(exit_price) - trade.entry_price) * trade.shares
            trade.updated_at = timestamp
            updated += 1
            if float(exit_price) >= 0.99:
                trade.status = "CLOSED"
                trade.closed_at = timestamp
                trade.exit_price = float(exit_price)
                trade.exit_edge = trade.last_edge
                trade.realized_pnl = trade.last_unrealized_pnl
                trade.close_reason = "Contract reached forced take-profit threshold >= 0.99."
                events.append(self._event("CLOSE", trade, timestamp))
                closed += 1
                log_runtime(
                    f"[paper] take-profit trade closed city={trade.city_name} side={trade.side} "
                    f"question={trade.question} exit={float(exit_price):.3f}"
                )
            elif entry_price_now is not None and float(entry_price_now) <= self.cfg.paper_min_contract_price:
                trade.status = "CLOSED"
                trade.closed_at = timestamp
                trade.exit_price = float(exit_price)
                trade.exit_edge = trade.last_edge
                trade.realized_pnl = trade.last_unrealized_pnl
                trade.close_reason = "Contract fell below tradable price floor during mark-to-market."
                events.append(self._event("CLOSE", trade, timestamp))
                closed += 1
                log_runtime(
                    f"[paper] floor-close trade closed city={trade.city_name} side={trade.side} "
                    f"question={trade.question} ask={float(entry_price_now):.3f} exit={float(exit_price):.3f}"
                )
        if updated or events:
            self.store.save_trades(trades, events)
        return {"updated": updated, "closed": closed}

    def _resolved_price_for_token(self, state: dict[str, Any], token_id: str) -> float | None:
        if not state:
            return None
        is_closed = bool(state.get("closed"))
        is_resolved = str(state.get("uma_resolution_status") or "").lower() == "resolved"
        auto_resolved = bool(state.get("automatically_resolved"))
        accepting_orders = bool(state.get("accepting_orders", True))
        if not is_closed and accepting_orders and not is_resolved and not auto_resolved:
            return None
        prices_by_token = state.get("prices_by_token", {})
        if not isinstance(prices_by_token, dict) or not prices_by_token:
            return None
        direct_price = prices_by_token.get(token_id)
        if direct_price is None:
            return None
        return float(direct_price)

    def _close_expired(
        self,
        open_trades: list[PaperTrade],
        local_today: date,
        now: str,
        events: list[dict[str, Any]],
    ) -> list[PaperTrade]:
        closed: list[PaperTrade] = []
        for trade in open_trades:
            trade_day = self._question_date(trade.question)
            if trade_day is None or trade_day >= local_today:
                continue
            exit_price = trade.last_price if trade.last_price is not None else 0.0
            trade.status = "CLOSED"
            trade.closed_at = now
            trade.exit_price = exit_price
            trade.exit_edge = trade.last_edge
            trade.realized_pnl = (exit_price - trade.entry_price) * trade.shares
            trade.close_reason = "Market day passed; position expired from active universe."
            trade.updated_at = now
            closed.append(trade)
            events.append(self._event("CLOSE", trade, now))
            log_runtime(
                f"[paper] expired trade closed city={trade.city_name} side={trade.side} "
                f"question={trade.question} exit={exit_price:.3f}"
            )
        return closed

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
            log_runtime(
                f"[paper] invalidated trade closed city={trade.city_name} side={trade.side} "
                f"question={trade.question} entry={trade.entry_price:.3f} exit={current_exit_price:.3f} "
                f"pnl={unrealized_pnl:+.2f}"
            )
        return closed

    def _close_untradable(
        self,
        open_trades: list[PaperTrade],
        market_by_id: dict[str, BucketMarket],
        now: str,
        events: list[dict[str, Any]],
    ) -> list[PaperTrade]:
        closed: list[PaperTrade] = []
        for trade in open_trades:
            if trade.status != "OPEN":
                continue
            market = market_by_id.get(trade.market_id)
            if market is None:
                continue
            current_entry_price = self._entry_price_for_side(market, trade.side)
            current_exit_price = self._exit_price_for_side(market, trade.side)
            if current_entry_price is None or current_exit_price is None:
                continue
            if current_entry_price > self.cfg.paper_min_contract_price:
                continue
            trade.last_price = current_exit_price
            trade.last_unrealized_pnl = (current_exit_price - trade.entry_price) * trade.shares
            trade.updated_at = now
            trade.status = "CLOSED"
            trade.closed_at = now
            trade.exit_price = current_exit_price
            trade.exit_edge = trade.last_edge
            trade.realized_pnl = trade.last_unrealized_pnl
            trade.close_reason = "Contract fell below tradable price floor."
            closed.append(trade)
            events.append(self._event("CLOSE", trade, now))
            log_runtime(
                f"[paper] untradable trade closed city={trade.city_name} side={trade.side} "
                f"question={trade.question} current_entry={current_entry_price:.3f} exit={current_exit_price:.3f}"
            )
        return closed

    def _close_take_profit(
        self,
        open_trades: list[PaperTrade],
        market_by_id: dict[str, BucketMarket],
        now: str,
        events: list[dict[str, Any]],
    ) -> list[PaperTrade]:
        closed: list[PaperTrade] = []
        for trade in open_trades:
            if trade.status != "OPEN":
                continue
            market = market_by_id.get(trade.market_id)
            if market is None:
                continue
            current_exit_price = self._exit_price_for_side(market, trade.side)
            if current_exit_price is None or float(current_exit_price) < 0.99:
                continue
            trade.last_price = float(current_exit_price)
            trade.last_unrealized_pnl = (float(current_exit_price) - trade.entry_price) * trade.shares
            trade.updated_at = now
            trade.status = "CLOSED"
            trade.closed_at = now
            trade.exit_price = float(current_exit_price)
            trade.exit_edge = trade.last_edge
            trade.realized_pnl = trade.last_unrealized_pnl
            trade.close_reason = "Contract reached forced take-profit threshold >= 0.99."
            closed.append(trade)
            events.append(self._event("CLOSE", trade, now))
            log_runtime(
                f"[paper] take-profit trade closed city={trade.city_name} side={trade.side} "
                f"question={trade.question} exit={float(current_exit_price):.3f}"
            )
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
            market_day = self._question_date(action.question)
            local_today = datetime.now(UTC).astimezone(profile.timezone).date()
            if market_day is not None and market_day != local_today:
                log_runtime(
                    f"[paper] skipped stale market city={profile.city_name} side={side} question={action.question}"
                )
                continue
            price = self._entry_price_for_side(market, side)
            if price is None or price <= 0:
                continue
            if price <= self.cfg.paper_min_contract_price:
                log_runtime(
                    f"[paper] skipped cheap contract city={profile.city_name} side={side} "
                    f"question={action.question} price={price:.3f}"
                )
                continue

            fair_yes = fair_values.get(action.market_id, 0.0)
            fair = fair_yes if side == "YES" else 1.0 - fair_yes
            edge = fair - price
            if edge < self.cfg.min_edge_to_open:
                continue

            size_usd = self._position_size(trades, edge, fair, price, market, side)
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
            log_runtime(
                f"[paper] opened city={trade.city_name} side={trade.side} question={trade.question} "
                f"price={trade.entry_price:.3f} fair={trade.entry_fair:.3f} edge={trade.entry_edge:+.3f} "
                f"size={trade.size_usd:.2f}"
            )
        return opened

    def _position_size(
        self,
        trades: list[PaperTrade],
        edge: float,
        fair: float,
        price: float,
        market: BucketMarket,
        side: str,
    ) -> float:
        bankroll = self._bankroll(trades)
        full_kelly = max(0.0, min(1.0, edge / max(1e-6, 1.0 - price)))
        base = bankroll * full_kelly * self.cfg.paper_kelly_fraction

        if price <= 0.02:
            base *= 0.15
        elif price <= 0.05:
            base *= 0.25
        elif price <= 0.10:
            base *= 0.40
        elif price <= 0.20:
            base *= 0.60

        if fair < 0.20:
            base *= 0.25
        elif fair < 0.35:
            base *= 0.50
        elif fair < 0.50:
            base *= 0.75

        spread = self._spread_for_side(market, side)
        if spread is not None and spread >= 0.10:
            base *= 0.50
        elif spread is not None and spread >= 0.05:
            base *= 0.75

        size = round(max(self.cfg.paper_min_trade_usd, min(self.cfg.paper_max_trade_usd, base)), 2)
        return size

    def _bankroll(self, trades: list[PaperTrade]) -> float:
        realized = sum(float(trade.realized_pnl or 0.0) for trade in trades if trade.status == "CLOSED")
        return max(100.0, self.store.load().get("start_balance_usd", self.cfg.paper_start_balance_usd) + realized)

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

    def _question_date(self, question: str) -> date | None:
        try:
            marker = " on "
            if marker not in question:
                return None
            raw = question.split(marker, 1)[1].rstrip("?").strip()
            return datetime.strptime(raw, "%B %d, %Y").date()
        except ValueError:
            return None


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

    def city_open_stats(self) -> dict[str, dict[str, float | int]]:
        stats: dict[str, dict[str, float | int]] = {}
        for trade in self.store.load_trades():
            if trade.status != "OPEN":
                continue
            item = stats.setdefault(
                trade.profile_slug,
                {
                    "open_count": 0,
                    "unrealized_pnl": 0.0,
                    "exposure_usd": 0.0,
                },
            )
            item["open_count"] = int(item["open_count"]) + 1
            item["unrealized_pnl"] = float(item["unrealized_pnl"]) + float(trade.last_unrealized_pnl or 0.0)
            item["exposure_usd"] = float(item["exposure_usd"]) + float(trade.size_usd or 0.0)
        return stats

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
