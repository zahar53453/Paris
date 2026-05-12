# Paris Paper Bot

Standalone Railway-ready paper-trading bot for Polymarket weather markets.

## What it does

- runs every 5 minutes
- analyzes supported city weather markets
- opens and closes virtual paper trades from its own signals
- sends Telegram push notifications for new opens and closes
- sends a status message every cycle
- serves Telegram menu buttons:
  - `Open Trades`
  - `Closed Trades`
  - `Balance`
  - `Status`

## Run mode

```bash
python -m paris_today_bot.main --serve-paper --interval 300
```

## Railway

This repo already includes:

- `Dockerfile`
- `railway.json`

Use a persistent volume mounted at:

```text
/data
```

The container symlinks `/app/data` to `/data`, so paper state survives restarts.

## Required variables

```env
PARIS_BOT_DRY_RUN=true
PARIS_BOT_POLL_SECONDS=300
PARIS_BOT_PAPER_MIN_TRADE_USD=1
PARIS_BOT_PAPER_MAX_TRADE_USD=15
PARIS_BOT_PAPER_START_BALANCE_USD=1000
PARIS_BOT_PAPER_CLOSE_EDGE=0.00
PARIS_BOT_MIN_EDGE_TO_OPEN=0.10
PARIS_BOT_MIN_EDGE_TO_HOLD=0.03
PARIS_BOT_MIN_NO_PRICE=0.05
PARIS_BOT_MAX_YES_PRICE=0.95
PARIS_BOT_TELEGRAM_BOT_TOKEN=...
PARIS_BOT_TELEGRAM_CHAT_ID=...
PARIS_BOT_TELEGRAM_MENU_ENABLED=true
CHAIN_ID=137
```
