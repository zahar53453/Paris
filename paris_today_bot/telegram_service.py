from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import re

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from paris_today_bot.config import BotConfig
from paris_today_bot.runtime_log import log_runtime


MENU_OPEN = "Open Trades"
MENU_CLOSED = "Closed Trades"
MENU_BALANCE = "Balance"
MENU_STATUS = "Status"
MENU_PROBABILITIES = "Probabilities"
MENU_LOGS = "Logs"
MENU_RESTART = "Restart Bot"
MENU_CLEAR = "Clear History"


@dataclass(slots=True)
class RuntimeStatus:
    started_at: str | None = None
    last_cycle_started_at: str | None = None
    last_cycle_finished_at: str | None = None
    last_result: dict | None = None
    last_error: str | None = None


class PaperTelegramService:
    def __init__(
        self,
        cfg: BotConfig,
        runtime: RuntimeStatus,
        restart_callback: Callable[[], Awaitable[str]] | None = None,
        clear_history_callback: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.restart_callback = restart_callback
        self.clear_history_callback = clear_history_callback
        self.application: Application | None = None

    async def start(self) -> None:
        log_runtime(
            f"[telegram] start requested menu_enabled={self.cfg.telegram_menu_enabled} "
            f"token_present={bool(self.cfg.telegram_bot_token)} "
            f"chat_present={bool(self.cfg.telegram_chat_id)}"
        )
        if not self.cfg.telegram_menu_enabled or not self.cfg.telegram_bot_token:
            log_runtime("[telegram] startup skipped because telegram is not fully configured")
            return
        try:
            self.application = ApplicationBuilder().token(self.cfg.telegram_bot_token).build()
            self.application.add_handler(CommandHandler("start", self.menu_command))
            self.application.add_handler(CommandHandler("menu", self.menu_command))
            self.application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_menu_text))
            await self.application.initialize()
            log_runtime("[telegram] initialized")
            await self.application.start()
            log_runtime("[telegram] application started")
            await self.application.updater.start_polling()
            log_runtime("[telegram] polling started")
        except Exception as exc:
            log_runtime(f"[telegram] startup failed: {type(exc).__name__}: {exc}")
            raise

    async def stop(self) -> None:
        if self.application is None:
            return
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    async def push_message(self, text: str) -> None:
        if not self.application or not self.cfg.telegram_chat_id:
            log_runtime("[telegram] push skipped because application/chat is unavailable")
            return
        for chunk in self._split_text(text):
            try:
                await self.application.bot.send_message(chat_id=self.cfg.telegram_chat_id, text=chunk)
                log_runtime(f"[telegram] pushed message chunk len={len(chunk)}")
            except Exception as exc:
                log_runtime(f"[telegram] push failed: {type(exc).__name__}: {exc}")
                raise

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        keyboard = [
            [MENU_OPEN, MENU_CLOSED],
            [MENU_BALANCE, MENU_STATUS],
            [MENU_PROBABILITIES, MENU_LOGS],
            [MENU_RESTART, MENU_CLEAR],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        if update.message is not None:
            await update.message.reply_text("Paris today paper bot menu:", reply_markup=reply_markup)

    async def handle_menu_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update) or update.message is None:
            return
        text_cmd = update.message.text
        if text_cmd == MENU_OPEN:
            from paris_today_bot.main import build_paper_report
            report = await build_paper_report(refresh_prices=True)
            await self._safe_reply(update, report["telegram_text"]["open_trades"])
            return
        if text_cmd == MENU_CLOSED:
            from paris_today_bot.main import build_paper_report
            report = await build_paper_report(refresh_prices=False)
            await self._safe_reply(update, report["telegram_text"]["closed_trades"])
            return
        if text_cmd == MENU_BALANCE:
            from paris_today_bot.main import build_paper_report
            report = await build_paper_report(refresh_prices=True)
            await self._safe_reply(update, report["telegram_text"]["balance"])
            return
        if text_cmd == MENU_STATUS:
            await self._safe_reply(update, self.status_text())
            return
        if text_cmd == MENU_PROBABILITIES:
            await self._safe_reply(update, self.probabilities_text())
            return
        if text_cmd == MENU_LOGS:
            await self._safe_reply(update, self.runtime_log_text())
            return
        if text_cmd == MENU_RESTART:
            if self.restart_callback is None:
                await self._safe_reply(update, "Restart action is not configured.")
                return
            await self._safe_reply(update, await self.restart_callback())
            return
        if text_cmd == MENU_CLEAR:
            if self.clear_history_callback is None:
                await self._safe_reply(update, "Clear-history action is not configured.")
                return
            await self._safe_reply(update, await self.clear_history_callback())
            return
        await update.message.reply_text("Use the menu buttons.")

    def status_text(self) -> str:
        summary = self.runtime.last_result.get("paper_summary") if self.runtime.last_result else None
        city_stats = (self.runtime.last_result or {}).get("paper_city_stats", {})
        probability_changes = (self.runtime.last_result or {}).get("probability_changes", {})
        lines = ["Paper bot status"]
        if self.runtime.started_at:
            lines.append(f"Started: {self.runtime.started_at}")
        if self.runtime.last_cycle_started_at:
            lines.append(f"Last cycle start: {self.runtime.last_cycle_started_at}")
        if self.runtime.last_cycle_finished_at:
            lines.append(f"Last cycle finish: {self.runtime.last_cycle_finished_at}")
        if summary is not None:
            lines.append(f"Open trades: {summary.get('open_count', 0)}")
            lines.append(f"Closed trades: {summary.get('closed_count', 0)}")
            lines.append(f"Realized PnL: {float(summary.get('realized_pnl', 0.0)):+.2f}$")
            lines.append(f"Unrealized PnL: {float(summary.get('unrealized_pnl', 0.0)):+.2f}$")
        if self.runtime.last_error:
            lines.append(f"Last error: {self.runtime.last_error}")
        else:
            lines.append("Last error: none")
        for item in (self.runtime.last_result or {}).get("results", []):
            profile = item.get("profile", {})
            weather = item.get("weather", {})
            decision = item.get("decision", {})
            stat = city_stats.get(profile.get("slug"), {})
            lines.append(
                f"{profile.get('city_name')}: max {decision.get('projected_max')}C | "
                f"obs {weather.get('obs_current')}C | city_open {int(stat.get('open_count', 0))} | "
                f"city_uPnL {float(stat.get('unrealized_pnl', 0.0)):+.2f}$"
            )
        lines.extend(
            _probability_changes_lines(
                probability_changes,
                (self.runtime.last_result or {}).get("results", []),
                include_header=True,
            )
        )
        return "\n".join(lines)

    def runtime_log_text(self, limit_lines: int = 60) -> str:
        path = self.cfg.runtime_log_file
        if not path.exists():
            return f"Runtime log not found.\nExpected path: {path}"
        lines = path.read_text(encoding="utf-8").splitlines()
        tail = lines[-limit_lines:]
        return "Runtime log tail\n\n" + ("\n".join(tail) if tail else "Log is empty.")

    def probabilities_text(self) -> str:
        results = (self.runtime.last_result or {}).get("results", [])
        if not results:
            return "No analysis results yet."
        lines = ["Latest probabilities"]
        for item in results:
            profile = item.get("profile", {})
            actions = item.get("actions", [])
            fair_values = (item.get("decision") or {}).get("fair_values", {})
            positive_lines: list[str] = []
            for action in actions:
                market_id = action.get("market_id")
                if not market_id:
                    continue
                probability = fair_values.get(market_id)
                if probability is None or float(probability) <= 0:
                    continue
                positive_lines.append(
                    f"{self._question_label(action.get('question', 'Unknown market'))}: {float(probability) * 100:.1f}%"
                )
            if not positive_lines:
                continue
            lines.append("")
            lines.append(profile.get("city_name", "Unknown city"))
            lines.extend(positive_lines)
        return "\n".join(lines) if len(lines) > 1 else "No positive probabilities in the latest analysis."

    async def _safe_reply(self, update: Update, text: str) -> None:
        if update.message is None:
            return
        for chunk in self._split_text(text):
            await update.message.reply_text(chunk)

    def _allowed(self, update: Update) -> bool:
        if not self.cfg.telegram_chat_id:
            return False
        actual_chat_id = str(update.effective_chat.id) if update.effective_chat else ""
        return actual_chat_id == str(self.cfg.telegram_chat_id)

    def _split_text(self, text: str, limit: int = 4000) -> list[str]:
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        current = ""
        for line in text.splitlines():
            candidate = f"{current}\n{line}".strip("\n") if current else line
            if len(candidate) > limit and current:
                parts.append(current)
                current = line
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    def _question_label(self, question: str) -> str:
        match = re.search(r"be\s+(.+?)\s+on\s+[A-Z][a-z]{2}\s+\d{1,2}\??$", question)
        if match:
            return match.group(1).replace(" or below", " or below").replace(" or higher", " or higher")
        return question


def render_cycle_notifications(result: dict) -> list[str]:
    messages: list[str] = []
    for item in result.get("results", []):
        paper = item.get("paper") or {}
        for trade in paper.get("opened", []):
            messages.append(
                "\n".join(
                    [
                        "OPEN",
                        f"{trade['city_name']} | {trade['side']}",
                        trade["question"],
                        f"Entry: {float(trade['entry_price']):.3f} | Size: ${float(trade['size_usd']):.2f}",
                        f"Edge: {float(trade['entry_edge']):+.3f} | Fair: {float(trade['entry_fair']):.3f}",
                    ]
                )
            )
        for trade in paper.get("closed", []):
            messages.append(
                "\n".join(
                    [
                        "CLOSE",
                        f"{trade['city_name']} | {trade['side']}",
                        trade["question"],
                        f"Entry: {float(trade['entry_price']):.3f} -> Exit: {float(trade['exit_price'] or 0.0):.3f}",
                        f"PnL: {float(trade['realized_pnl'] or 0.0):+.2f}$",
                        f"Reason: {trade.get('close_reason') or 'n/a'}",
                    ]
                )
            )
    summary = result.get("paper_summary", {})
    city_stats = result.get("paper_city_stats", {})
    city_lines = []
    for item in result.get("results", []):
        profile = item.get("profile", {})
        decision = item.get("decision", {})
        weather = item.get("weather", {})
        stat = city_stats.get(profile.get("slug"), {})
        city_lines.append(
            f"{profile.get('city_name')}: max {decision.get('projected_max')}C | "
            f"obs {weather.get('obs_current')}C | city_open {int(stat.get('open_count', 0))} | "
            f"city_uPnL {float(stat.get('unrealized_pnl', 0.0)):+.2f}$"
        )
    if result.get("errors"):
        for error in result["errors"]:
            city_lines.append(f"{error['profile']['city_name']}: ERROR {error['error']}")
    probability_change_lines = _probability_changes_lines(result.get("probability_changes", {}), result.get("results", []), include_header=True)
    messages.append(
        "\n".join(
            [
                "STATUS",
                f"Cycle: {datetime.now(UTC).isoformat()}",
                f"Open trades: {summary.get('open_count', 0)} | Closed trades: {summary.get('closed_count', 0)}",
                f"Realized: {float(summary.get('realized_pnl', 0.0)):+.2f}$ | Unrealized: {float(summary.get('unrealized_pnl', 0.0)):+.2f}$",
                *city_lines,
                *probability_change_lines,
            ]
        )
    )
    return messages


def _probability_changes_lines(
    probability_changes: dict[str, list[dict[str, float | str]]],
    results: list[dict],
    *,
    include_header: bool,
) -> list[str]:
    if not probability_changes:
        return []
    city_names = {
        (item.get("profile") or {}).get("slug"): (item.get("profile") or {}).get("city_name", "Unknown city")
        for item in results
    }
    lines: list[str] = []
    if include_header:
        lines.extend(["", "Probability changes"])
    for slug, changes in probability_changes.items():
        if not changes:
            continue
        lines.append(city_names.get(slug, slug))
        for change in changes:
            lines.append(
                f"{change['label']}: {float(change['previous']) * 100:.1f}% -> {float(change['current']) * 100:.1f}%"
            )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines
