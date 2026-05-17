from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class BotConfig:
    dry_run: bool = _env_bool("PARIS_BOT_DRY_RUN", True)
    poll_seconds: int = int(os.getenv("PARIS_BOT_POLL_SECONDS", "60"))
    telegram_bot_token: str = os.getenv("PARIS_BOT_TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("PARIS_BOT_TELEGRAM_CHAT_ID", "")
    telegram_menu_enabled: bool = _env_bool("PARIS_BOT_TELEGRAM_MENU_ENABLED", True)
    trade_size_usd: float = float(os.getenv("PARIS_BOT_TRADE_SIZE_USD", "15"))
    paper_min_trade_usd: float = float(os.getenv("PARIS_BOT_PAPER_MIN_TRADE_USD", "1"))
    paper_max_trade_usd: float = float(os.getenv("PARIS_BOT_PAPER_MAX_TRADE_USD", "15"))
    paper_start_balance_usd: float = float(os.getenv("PARIS_BOT_PAPER_START_BALANCE_USD", "1000"))
    paper_close_edge: float = float(os.getenv("PARIS_BOT_PAPER_CLOSE_EDGE", "0.00"))
    paper_min_contract_price: float = float(os.getenv("PARIS_BOT_PAPER_MIN_CONTRACT_PRICE", "0.01"))
    paper_kelly_fraction: float = float(os.getenv("PARIS_BOT_PAPER_KELLY_FRACTION", "0.05"))
    min_edge_to_open: float = float(os.getenv("PARIS_BOT_MIN_EDGE_TO_OPEN", "0.10"))
    min_edge_to_hold: float = float(os.getenv("PARIS_BOT_MIN_EDGE_TO_HOLD", "0.03"))
    min_no_price: float = float(os.getenv("PARIS_BOT_MIN_NO_PRICE", "0.05"))
    max_yes_price: float = float(os.getenv("PARIS_BOT_MAX_YES_PRICE", "0.95"))
    chain_id: int = int(os.getenv("CHAIN_ID", "137"))
    polymarket_private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    funder_address: str = os.getenv("FUNDER_ADDRESS", "")
    max_positions: int = int(os.getenv("PARIS_BOT_MAX_POSITIONS", "3"))
    gamma_api_url: str = os.getenv("PARIS_BOT_GAMMA_URL", "https://gamma-api.polymarket.com")
    clob_api_url: str = os.getenv(
        "PARIS_BOT_CLOB_URL",
        "https://clob.polymarket.com" if int(os.getenv("CHAIN_ID", "137")) == 137 else "https://clob.amoy.polymarket.com",
    )

    def state_file_for_profile(self, profile_slug: str) -> Path:
        env_path = os.getenv("PARIS_BOT_STATE_FILE")
        if env_path:
            return Path(env_path)
        return Path("data") / f"{profile_slug}_state.json"

    @property
    def paper_state_file(self) -> Path:
        env_path = os.getenv("PARIS_BOT_PAPER_STATE_FILE")
        if env_path:
            return Path(env_path)
        return Path("data") / "paris_today_bot_paper_state.json"

    @property
    def runtime_log_file(self) -> Path:
        env_path = os.getenv("PARIS_BOT_RUNTIME_LOG_FILE")
        if env_path:
            return Path(env_path)
        return Path("data") / "paris_today_bot_runtime.log"


config = BotConfig()
