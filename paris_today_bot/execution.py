from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs
except Exception:  # pragma: no cover - optional runtime dependency
    ClobClient = None
    OrderArgs = None

from paris_today_bot.config import BotConfig
from paris_today_bot.models import BucketMarket, Position, TradeAction
from paris_today_bot.state import StateStore


class CityExecutor:
    def __init__(self, cfg: BotConfig, state: StateStore) -> None:
        self.cfg = cfg
        self.state = state
        self.advisory_mode = cfg.dry_run
        self.positions = [] if self.advisory_mode else state.load_positions()
        self.client = None
        if not cfg.dry_run and ClobClient and cfg.polymarket_private_key and cfg.funder_address:
            host = cfg.clob_api_url
            self.client = ClobClient(
                host=host,
                key=cfg.polymarket_private_key,
                chain_id=cfg.chain_id,
                funder=cfg.funder_address,
                signature_type=1,
            )
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)

    def decide_actions(
        self,
        markets: Iterable[BucketMarket],
        fair_values: dict[str, float],
    ) -> list[TradeAction]:
        actions: list[TradeAction] = []
        open_by_market = {position.market_id: position for position in self.positions}
        for market in markets:
            fair = fair_values.get(market.market_id, 0.0)
            yes_price = market.best_ask if market.best_ask is not None else market.midpoint
            no_price = market.no_best_ask if market.no_best_ask is not None else market.no_midpoint
            position = open_by_market.get(market.market_id)

            if not self.advisory_mode and position:
                hold_edge = fair - position.entry_price if position.side == "YES" else (1.0 - fair) - position.entry_price
                invalidated = position.side == "YES" and fair < 0.50 or position.side == "NO" and fair > 0.50
                if hold_edge < self.cfg.min_edge_to_hold or invalidated:
                    actions.append(
                        TradeAction(
                            action="SELL",
                            market_id=market.market_id,
                            question=market.question,
                            token_id=market.token_id,
                            side=position.side,
                            price=market.best_bid if position.side == "YES" else market.no_best_bid,
                            size_usd=position.size_usd,
                            edge=hold_edge,
                            reason="Edge compressed or scenario invalidated.",
                        )
                    )
                else:
                    actions.append(
                        TradeAction(
                            action="HOLD",
                            market_id=market.market_id,
                            question=market.question,
                            token_id=market.token_id,
                            side=position.side,
                            price=position.entry_price,
                            size_usd=position.size_usd,
                            edge=hold_edge,
                            reason="Position still has sufficient edge.",
                        )
                    )
                continue

            if not self.advisory_mode and len(self.positions) >= self.cfg.max_positions:
                actions.append(
                    TradeAction(
                        action="SKIP",
                        market_id=market.market_id,
                        question=market.question,
                        token_id=market.token_id,
                        side="NONE",
                        price=None,
                        size_usd=0.0,
                        edge=0.0,
                        reason="Max position count reached.",
                    )
                )
                continue

            if yes_price is not None and yes_price <= self.cfg.max_yes_price:
                yes_edge = fair - yes_price
                if yes_edge >= self.cfg.min_edge_to_open:
                    actions.append(
                        TradeAction(
                            action="BUY_YES",
                            market_id=market.market_id,
                            question=market.question,
                            token_id=market.token_id,
                            side="YES",
                            price=yes_price,
                            size_usd=self.cfg.trade_size_usd,
                            edge=yes_edge,
                            reason="YES edge exceeds entry threshold.",
                        )
                    )
                    continue

            if no_price is not None and no_price >= self.cfg.min_no_price:
                no_edge = (1.0 - fair) - no_price
                if no_edge >= self.cfg.min_edge_to_open:
                    actions.append(
                        TradeAction(
                            action="BUY_NO",
                            market_id=market.market_id,
                            question=market.question,
                            token_id=market.no_token_id or market.token_id,
                            side="NO",
                            price=no_price,
                            size_usd=self.cfg.trade_size_usd,
                            edge=no_edge,
                            reason="NO edge exceeds entry threshold.",
                        )
                    )
                    continue

            actions.append(
                TradeAction(
                    action="SKIP",
                    market_id=market.market_id,
                    question=market.question,
                    token_id=market.token_id,
                    side="NONE",
                    price=yes_price,
                    size_usd=0.0,
                    edge=(fair - yes_price) if yes_price is not None else 0.0,
                    reason="No side met the minimum edge threshold.",
                )
            )
        return actions

    def execute(self, actions: Iterable[TradeAction]) -> list[str]:
        logs: list[str] = []
        open_by_market = {position.market_id: position for position in self.positions}
        for action in actions:
            if action.action == "BUY_YES":
                if not self.advisory_mode:
                    self.positions.append(
                        Position(
                            market_id=action.market_id,
                            question=action.question,
                            token_id=action.token_id,
                            side="YES",
                            entry_price=action.price or 0.0,
                            size_usd=action.size_usd,
                            opened_at=datetime.now(UTC).isoformat(),
                        )
                    )
                self._post_live_order(action.token_id, "BUY", action.size_usd, action.price)
                logs.append(f"{action.action} {action.question} @ {action.price:.3f} edge={action.edge:+.3f}")
            elif action.action == "BUY_NO":
                if not self.advisory_mode:
                    self.positions.append(
                        Position(
                            market_id=action.market_id,
                            question=action.question,
                            token_id=action.token_id,
                            side="NO",
                            entry_price=action.price or 0.0,
                            size_usd=action.size_usd,
                            opened_at=datetime.now(UTC).isoformat(),
                        )
                    )
                self._post_live_order(action.token_id, "BUY", action.size_usd, action.price)
                logs.append(f"{action.action} {action.question} @ {action.price:.3f} edge={action.edge:+.3f}")
            elif action.action == "SELL":
                position = open_by_market.get(action.market_id)
                if position is not None:
                    self._post_live_order(position.token_id, "SELL", position.size_usd, action.price or position.entry_price, position=position)
                self.positions = [position for position in self.positions if position.market_id != action.market_id]
                logs.append(f"SELL {action.question} reason={action.reason}")
            else:
                logs.append(f"{action.action} {action.question} reason={action.reason}")
        if not self.advisory_mode:
            self.state.save_positions(self.positions)
        return logs

    def _post_live_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float | None,
        position: Position | None = None,
    ) -> None:
        if not self.client or self.cfg.dry_run or OrderArgs is None or not price:
            return
        if side.upper() == "BUY":
            shares = round(size_usd / max(price, 0.01), 2)
        else:
            ref_price = position.entry_price if position is not None else price
            shares = round(size_usd / max(ref_price, 0.01), 2)
        self.client.create_and_post_order(
            OrderArgs(
                price=round(price, 3),
                size=shares,
                side=side.upper(),
                token_id=token_id,
            )
        )
