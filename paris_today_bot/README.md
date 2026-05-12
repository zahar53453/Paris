# Paris Today Bot

Autonomous Paris-only same-day weather bot for `LFPB`.

Rules:

- Uses `GFS` as the base forecast for Paris.
- Applies deterministic adjustments from the live `METAR` path.
- Trades only the active Paris weather event for today's date.
- Keeps its own state in `data/paris_today_bot_state.json`.
- Does not import or use the repo's existing analysis or trading logic.

Run one cycle:

```bash
python -m paris_today_bot.main --once
```

Replay a local weather snapshot:

```bash
python -m paris_today_bot.main --once --snapshot-file path\\to\\LFPB_2026-05-04_full_data.txt
```
