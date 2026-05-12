from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from paris_today_bot.config import BotConfig


MENU_OPEN = "Open Trades"
MENU_CLOSED = "Closed Trades"
MENU_BALANCE = "Balance"
MENU_STATUS = "Status"


@dataclass(slots=True)
class RuntimeStatus:
    started_at: str | None = None
    last_cycle_started_at: str | None = None
    last_cycle_finished_at: str | None = None
    last_result: dict | None = None
    last_error: str | None = None


class PaperTelegramService:
    def __init__(self, cfg: BotConfig, runtime: RuntimeStatus) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.application: Application | None = None

    async def start(self) -> None:
        if not self.cfg.telegram_menu_enabled or not self.cfg.telegram_bot_token:
            return
        self.application = ApplicationBuilder().token(self.cfg.telegram_bot_token).build()
        self.application.add_handler(CommandHandler("start", self.menu_command))
        self.application.add_handler(CommandHandler("menu", self.menu_command))
        self.application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_menu_text))
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def stop(self) -> None:
        if self.application is None:
            return
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    async def push_message(self, text: str) -> None:
        if not self.application or not self.cfg.telegram_chat_id:
            return
        for chunk in self._split_text(text):
            await self.application.bot.send_message(chat_id=self.cfg.telegram_chat_id, text=chunk)

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        keyboard = [
            [MENU_OPEN, MENU_CLOSED],
            [MENU_BALANCE, MENU_STATUS],
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
        await update.message.reply_text("Use the menu buttons.")

    def status_text(self) -> str:
        summary = self.runtime.last_result.get("paper_summary") if self.runtime.last_result else None
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
        return "\n".join(lines)

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
    city_lines = []
    for item in result.get("results", []):
        profile = item.get("profile", {})
        decision = item.get("decision", {})
        weather = item.get("weather", {})
        city_lines.append(
            f"{profile.get('city_name')}: max {decision.get('projected_max')}C | "
            f"obs {weather.get('obs_current')}C | open {int((item.get('paper') or {}).get('open_count', 0))}"
        )
    if result.get("errors"):
        for error in result["errors"]:
            city_lines.append(f"{error['profile']['city_name']}: ERROR {error['error']}")
    messages.append(
        "\n".join(
            [
                "STATUS",
                f"Cycle: {datetime.now(UTC).isoformat()}",
                f"Open trades: {summary.get('open_count', 0)} | Closed trades: {summary.get('closed_count', 0)}",
                f"Realized: {float(summary.get('realized_pnl', 0.0)):+.2f}$ | Unrealized: {float(summary.get('unrealized_pnl', 0.0)):+.2f}$",
                *city_lines,
            ]
        )
    )
    return messages
