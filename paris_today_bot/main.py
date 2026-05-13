from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime

from paris_today_bot.config import config
from paris_today_bot.execution import CityExecutor
from paris_today_bot.paper import PaperBroker, PaperReporter, PaperStore
from paris_today_bot.polymarket_client import CityMarketClient
from paris_today_bot.profile_loader import list_profiles, load_profile
from paris_today_bot.runtime_log import log_runtime
from paris_today_bot.state import StateStore
from paris_today_bot.strategy import ProfiledTodayStrategy
from paris_today_bot.telegram_service import PaperTelegramService, RuntimeStatus, render_cycle_notifications
from paris_today_bot.weather_client import WeatherDataClient


async def run_for_profile(
    profile_name: str,
    snapshot_file: str | None = None,
    market_file: str | None = None,
    paper: bool = False,
    paper_broker: PaperBroker | None = None,
) -> dict:
    profile = load_profile(profile_name)
    weather_client = WeatherDataClient()
    weather = await weather_client.fetch_today(profile, snapshot_file=snapshot_file)

    if market_file:
        market_payload = json.loads(open(market_file, "r", encoding="utf-8").read())
        raise NotImplementedError("market_file replay is not implemented yet.")

    market_client = CityMarketClient(config, profile)
    market = await market_client.fetch_today_market()

    strategy = ProfiledTodayStrategy(profile)
    decision = strategy.evaluate(weather, market)

    state = StateStore(config.state_file_for_profile(profile.slug))
    executor = CityExecutor(config, state)
    actions = executor.decide_actions(market.markets, decision.fair_values)
    logs = executor.execute(actions)
    paper_result = None
    if paper:
        broker = paper_broker or PaperBroker(config, PaperStore(config.paper_state_file, config.paper_start_balance_usd))
        paper_result = broker.process_profile(profile, market.markets, decision.fair_values, actions)

    result = {
        "profile": {
            "slug": profile.slug,
            "city_name": profile.city_name,
            "icao": profile.icao,
        },
        "weather": {
            "local_time": weather.now_utc.astimezone(profile.timezone).isoformat(),
            "obs_current": weather.obs_current,
            "obs_max_so_far": weather.obs_max_so_far,
            "max_by_10utc": weather.max_by_10utc,
            "max_by_noon": weather.max_by_noon,
            "day_regime": weather.day_regime,
            "gfs_daily_max": weather.gfs_daily_max,
            "ecmwf_daily_max": weather.ecmwf_daily_max,
            "ensemble_daily_max": weather.ensemble_daily_max,
            "ensemble_daily_spread": weather.ensemble_daily_spread,
            "model_consensus_max": weather.model_consensus_max,
            "upper_realistic_max": weather.upper_realistic_max,
            "model_agreement_spread": weather.model_agreement_spread,
        },
        "decision": asdict(decision),
        "actions": [asdict(action) for action in actions],
        "logs": logs,
    }
    if paper_result is not None:
        result["paper"] = paper_result
    buy_actions = sum(1 for action in actions if action.action in {"BUY_YES", "BUY_NO"})
    log_runtime(
        f"[profile] {profile.city_name} projected_max={decision.projected_max} "
        f"actions={len(actions)} buy_actions={buy_actions} "
        f"paper_opened={len((paper_result or {}).get('opened', []))} "
        f"paper_closed={len((paper_result or {}).get('closed', []))}"
    )
    return result


async def run_all_profiles(paper: bool = False) -> dict:
    results: list[dict] = []
    errors: list[dict] = []
    paper_store = PaperStore(config.paper_state_file, config.paper_start_balance_usd) if paper else None
    paper_broker = PaperBroker(config, paper_store) if paper and paper_store is not None else None
    for profile in list_profiles():
        try:
            result = await run_for_profile(profile_name=str(profile.path), paper=paper, paper_broker=paper_broker)
            results.append(result)
        except Exception as exc:
            errors.append(
                {
                    "profile": {
                        "slug": profile.slug,
                        "city_name": profile.city_name,
                        "icao": profile.icao,
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    response = {
        "results": results,
        "errors": errors,
    }
    if paper:
        reporter = PaperReporter(config, paper_store or PaperStore(config.paper_state_file, config.paper_start_balance_usd))
        response["paper_summary"] = reporter.summary()
        response["paper_city_stats"] = reporter.city_open_stats()
    return response


async def run_paper_loop(profile_name: str | None, interval_seconds: int) -> None:
    while True:
        log_runtime(f"[paper-loop] cycle start profile={profile_name or 'ALL'}")
        if profile_name:
            result = await run_for_profile(profile_name=profile_name, paper=True)
        else:
            result = await run_all_profiles(paper=True)
        log_runtime(
            f"[paper-loop] cycle finish results={len(result.get('results', []))} "
            f"errors={len(result.get('errors', []))}"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        await asyncio.sleep(interval_seconds)


async def run_paper_telegram_service(profile_name: str | None, interval_seconds: int) -> None:
    log_runtime(
        f"[paper-service] boot profile={profile_name or 'ALL'} interval={interval_seconds}s "
        f"telegram_enabled={config.telegram_menu_enabled}"
    )
    runtime = RuntimeStatus(started_at=datetime.now(UTC).isoformat())
    store = PaperStore(config.paper_state_file, config.paper_start_balance_usd)
    cycle_lock = asyncio.Lock()

    async def execute_cycle(trigger: str) -> dict:
        runtime.last_cycle_started_at = datetime.now(UTC).isoformat()
        log_runtime(f"[paper-service] cycle started trigger={trigger} at {runtime.last_cycle_started_at}")
        refresh_state = await refresh_paper_prices()
        if profile_name:
            result = {
                "results": [await run_for_profile(profile_name=profile_name, paper=True)],
                "errors": [],
            }
            result["paper_summary"] = PaperReporter(config, store).summary()
            result["paper_city_stats"] = PaperReporter(config, store).city_open_stats()
        else:
            result = await run_all_profiles(paper=True)
        result["paper_refresh"] = refresh_state
        runtime.last_result = result
        runtime.last_error = None
        runtime.last_cycle_finished_at = datetime.now(UTC).isoformat()
        log_runtime(
            f"[paper-service] cycle finished trigger={trigger} at {runtime.last_cycle_finished_at} "
            f"results={len(result.get('results', []))} errors={len(result.get('errors', []))}"
        )
        return result

    async def run_managed_cycle(trigger: str, *, notify: bool) -> dict | None:
        if cycle_lock.locked():
            log_runtime(f"[paper-service] cycle skipped trigger={trigger} reason=busy")
            return None
        async with cycle_lock:
            result = await execute_cycle(trigger)
            if notify:
                for message in render_cycle_notifications(result):
                    await telegram.push_message(message)
            return result

    async def restart_callback() -> str:
        if cycle_lock.locked():
            return "Restart skipped: a scan is already running."
        try:
            result = await run_managed_cycle("telegram_restart", notify=True)
        except Exception as exc:
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            runtime.last_cycle_finished_at = datetime.now(UTC).isoformat()
            log_runtime(f"[paper-service] manual restart failed: {runtime.last_error}")
            return f"Restart failed.\n{runtime.last_error}"
        if result is None:
            return "Restart skipped: a scan is already running."
        summary = result.get("paper_summary", {})
        return (
            "Restart scan completed.\n"
            f"Open trades: {summary.get('open_count', 0)}\n"
            f"Closed trades: {summary.get('closed_count', 0)}\n"
            f"Realized PnL: {float(summary.get('realized_pnl', 0.0)):+.2f}$\n"
            f"Unrealized PnL: {float(summary.get('unrealized_pnl', 0.0)):+.2f}$"
        )

    async def clear_history_callback() -> str:
        if cycle_lock.locked():
            return "Clear-history skipped: wait for the current scan to finish."
        store.reset()
        runtime.last_result = {
            "results": [],
            "errors": [],
            "paper_summary": PaperReporter(config, store).summary(),
        }
        runtime.last_error = None
        log_runtime("[paper-service] paper history reset from telegram")
        return "Paper history fully cleared. Open and closed trades were removed."

    telegram = PaperTelegramService(
        config,
        runtime,
        restart_callback=restart_callback,
        clear_history_callback=clear_history_callback,
    )
    await telegram.start()
    await telegram.push_message("Paris today paper bot started.")
    try:
        while True:
            try:
                await run_managed_cycle("scheduler", notify=True)
            except Exception as exc:
                runtime.last_error = f"{type(exc).__name__}: {exc}"
                runtime.last_cycle_finished_at = datetime.now(UTC).isoformat()
                log_runtime(f"[paper-service] cycle failed: {runtime.last_error}")
                await telegram.push_message(f"Cycle error\n{runtime.last_error}")
            await asyncio.sleep(interval_seconds)
    finally:
        log_runtime("[paper-service] shutting down telegram service")
        await telegram.stop()


async def refresh_paper_prices() -> dict:
    store = PaperStore(config.paper_state_file, config.paper_start_balance_usd)
    broker = PaperBroker(config, store)
    open_trades = [trade for trade in store.load_trades() if trade.status == "OPEN"]
    profiles_by_slug = {profile.slug: profile for profile in list_profiles()}
    refreshed: list[dict] = []
    errors: list[dict] = []
    for profile_slug in sorted({trade.profile_slug for trade in open_trades}):
        profile = profiles_by_slug.get(profile_slug)
        if profile is None:
            continue
        try:
            profile_trades = [trade for trade in open_trades if trade.profile_slug == profile_slug]
            client = CityMarketClient(config, profile)
            token_ids = [trade.token_id for trade in profile_trades]
            books_by_token = await client.fetch_books_by_token(token_ids)
            event_slugs = sorted(
                {
                    event_slug
                    for trade in profile_trades
                    if (event_slug := client.event_slug_for_question(trade.question))
                }
            )
            closed_market_states = await client.fetch_event_market_states(event_slugs)
            item = broker.mark_to_market_by_books(
                profile,
                books_by_token,
                closed_market_states=closed_market_states,
            )
            log_runtime(
                f"[paper-refresh] profile={profile.slug} open={len(profile_trades)} "
                f"books={len(books_by_token)} closed_states={len(closed_market_states)} "
                f"updated={item.get('updated', 0)} closed={item.get('closed', 0)}"
            )
            refreshed.append({"profile": profile_slug, **item})
        except Exception as exc:
            log_runtime(f"[paper-refresh] profile={profile_slug} failed: {type(exc).__name__}: {exc}")
            errors.append({"profile": profile_slug, "error": f"{type(exc).__name__}: {exc}"})
    return {"refreshed": refreshed, "errors": errors}


async def build_paper_report(refresh_prices: bool = True) -> dict:
    store = PaperStore(config.paper_state_file, config.paper_start_balance_usd)
    refresh_result = await refresh_paper_prices() if refresh_prices else {"refreshed": [], "errors": []}
    reporter = PaperReporter(config, store)
    return {
        "summary": reporter.summary(),
        "city_stats": reporter.city_open_stats(),
        "open_trades": reporter.open_trades(),
        "closed_trades": reporter.closed_trades(),
        "refresh": refresh_result,
        "telegram_text": {
            "open_trades": reporter.open_trades_text(),
            "closed_trades": reporter.closed_trades_text(),
            "balance": reporter.balance_text(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-driven same-day Polymarket weather bot.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--paper", action="store_true", help="Record virtual paper trades from this run.")
    parser.add_argument("--paper-loop", action="store_true", help="Run continuously and record virtual paper trades.")
    parser.add_argument("--paper-report", action="store_true", help="Print paper trading open trades, closed trades and balance.")
    parser.add_argument("--serve-paper", action="store_true", help="Run the 5-minute paper bot loop together with Telegram push and menu polling.")
    parser.add_argument("--interval", type=int, default=config.poll_seconds, help="Loop interval in seconds.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile JSON filename from paris_today_bot/profiles or an absolute path.",
    )
    parser.add_argument("--snapshot-file", default=None, help="Optional local weather archive file for replay/debugging.")
    args = parser.parse_args()
    log_runtime(
        f"[main] args once={args.once} paper={args.paper} paper_loop={args.paper_loop} "
        f"paper_report={args.paper_report} serve_paper={args.serve_paper} profile={args.profile}"
    )

    if args.paper_report:
        result = asyncio.run(build_paper_report())
    elif args.serve_paper:
        if args.snapshot_file:
            raise RuntimeError("`--snapshot-file` cannot be used with `--serve-paper`.")
        asyncio.run(run_paper_telegram_service(args.profile, args.interval))
        return
    elif args.paper_loop:
        if args.snapshot_file:
            raise RuntimeError("`--snapshot-file` cannot be used with `--paper-loop`.")
        asyncio.run(run_paper_loop(args.profile, args.interval))
        return
    elif args.profile:
        result = asyncio.run(run_for_profile(profile_name=args.profile, snapshot_file=args.snapshot_file, paper=args.paper))
    else:
        if args.snapshot_file:
            raise RuntimeError("`--snapshot-file` can only be used together with a single `--profile` run.")
        result = asyncio.run(run_all_profiles(paper=args.paper))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
