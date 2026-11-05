# Swing Screener (Streamlit) — Phase 1

Mobile-first swing-trade screener. Pick a strategy, tap **Run**, get the top 10 long setups with entry trigger,
structure + ATR stops, R target, hold period / time stop, sector context and earnings check. Nothing runs until you tap Run.

## What's in phase 1
- Universe: S&P 500 + Nasdaq-100 by default; Quality dip v2 uses the Russell 1000 only (iShares IWB holdings,
  falling back to S&P 500 + S&P 400 if that file can't be read). Set `universe = [...]` in any strategy file.
- Screeners: Breakout · Pullback · Oversold bounce · Momentum · Quality dip **v2** (each in its own file)
- Quality dip v2: full Russell 1000 classification (5 lists), 40/40/20 scoring, triggers A–D, 2R/3R, resistance,
  position sizing, CSV + ranked report download, backtest module — see `docs/QUALITY_DIP_V2.md`
- Per-strategy trade plan: hold period, time stop, R target and ATR-stop multiple
- Screening Definitions tab, generated from the strategy files (rules, score, trigger, market fit, glossary)
- Shared pipeline: earnings ≤7d excluded (later reports inside the hold move the time stop), top 10 by score, SPY/50-SMA regime warning, breadth, VIX, sector state
- Live mode during market hours: today's partial bar + volume projected to a full day
- Data freshness stamp on every result (and a stale-data warning)
- Google Sheets log of every run + outcome scoring over each strategy's hold (logging currently off) (Tracker tab)

## Layout
```
app.py                  UI (Screen / Screening Definitions / Tracker / Status tabs)
core/
  settings.py           defaults (top N, earnings window, ATR stop multiple, …)
  data.py               data providers: Yahoo (no key) → Alpaca (when keys are set)
  universe.py           ticker lists (sp500, ndx100, sp400, russell1000) — add new universes here
  indicators.py         shared indicators (add a column once, every strategy can use it)
  context.py            regime, breadth, VIX, sector strength, relative-strength percentiles
  strategy.py           Strategy template + auto-discovery
  pipeline.py           rank → earnings filter → top N → card fields
  earnings.py           Finnhub calendar (key) or Yahoo fallback
  fundamentals.py       Yahoo fundamentals (+ Finnhub ROI), fetched only for prefilter survivors
  storage.py            Google Sheets log
  outcomes.py           outcome scoring (per-strategy hold & target)
  qd2.py                Quality dip v2 engine (shared by app, CLI and backtest)
  sizing.py             position sizing
  backtest.py           Quality dip v2 backtest engine
run_quality_dip.py      full-universe screen → CSV + report (CLI)
backtest_quality_dip.py backtest CLI
docs/QUALITY_DIP_V2.md  audit, schemas, compromises, implemented/tested/blocked status
strategies/             ONE FILE PER SCREENER (auto-loaded)
  _template.py          copy this to add a screener
tests/                  synthetic-data tests (pytest)
```

## Add / remove a screener
- **Add:** copy `strategies/_template.py` → `strategies/my_setup.py`; edit `key`, `name`, `params`, `hold`,
  `target_r`, `atr_stop_mult`, the `rules`/`score_parts` text and `evaluate()`.
- **Remove:** delete the file, or set `enabled = False`.
- **Tune:** edit the `params` dict at the top of the strategy file (the Definitions tab updates itself; bump `changed`).

| Strategy | Hold (trading days) | Target | ATR stop |
|---|---|---|---|
| Oversold bounce | 2–5 | 1.5R | 1.5× |
| Pullback | 3–10 | 2R | 1.5× |
| Breakout | 5–15 | 2.5R | 1.5× |
| Momentum | 5–20 | 3R | 2× |
| Quality dip v2 | 10–30 | 3R (2R secondary) | structural only |

No other file changes. Keep `key` stable once you start logging (the Tracker groups by it).

## Deploy (Streamlit Community Cloud)
1. **GitHub:** create a *private* repo (e.g. `swing-screener`) and upload this folder's contents.
   The `.streamlit` folder is hidden on Mac/Windows; if it doesn't upload, create
   `.streamlit/config.toml` in GitHub with **Add file → Create new file** and paste its contents.
2. **Streamlit:** go to share.streamlit.io → sign in with GitHub → **Create app** → pick the repo,
   branch `main`, main file `app.py` → Advanced settings: Python 3.12 → **Deploy**.
   It runs immediately on Yahoo data with no keys.
3. **Secrets (when you have them):** App → Settings → Secrets → paste the contents of
   `.streamlit/secrets.toml.example` with your values filled in. The Status tab shows what's connected.

### Getting the keys
- **Alpaca (data):** sign up at alpaca.markets → Paper Trading dashboard → API Keys → generate.
  Paper keys work for the free market-data plan. → `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- **Finnhub:** sign up at finnhub.io → dashboard shows your free key. → `FINNHUB_API_KEY`
- **Google Sheets log:**
  1. console.cloud.google.com → new project → enable **Google Sheets API** and **Google Drive API**.
  2. IAM & Admin → Service Accounts → create → Keys → Add key → JSON (downloads a file).
  3. Create a blank Google Sheet, **Share** it with the service account's `client_email` as Editor.
  4. Copy the sheet ID (between `/d/` and `/edit` in its URL) → `SHEET_ID`.
  5. Copy each field of the JSON into the `[gcp_service_account]` block of the secrets.

## Run locally (optional)
```
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests
```

## Phase 2 (planned)
Select results → AI analysis (Claude/Gemini toggle, default = cheaper) with entry/stop/target/invalidation,
macro impact, Finnhub news + analyst changes, FRED macro snapshot and event calendar.
