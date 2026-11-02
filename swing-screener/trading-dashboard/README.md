# Swing Screener (Streamlit) — Phase 1

Mobile-first swing-trade screener. Pick a strategy, tap **Run**, get the top 10 long setups with entry trigger,
structure + ATR stops, 2R level, sector context and earnings check. Nothing runs until you tap Run.

## What's in phase 1
- Universe: S&P 500 + Nasdaq-100 (live lists from Wikipedia, bundled fallback if unavailable)
- Screeners: Breakout · Pullback · Oversold bounce · Momentum (each in its own file)
- Shared pipeline: earnings ≤7d excluded, top 10 by score, SPY/50-SMA regime warning, breadth, VIX, sector state
- Live mode during market hours: today's partial bar + volume projected to a full day
- Data freshness stamp on every result (and a stale-data warning)
- Google Sheets log of every run + automatic 5-day outcome scoring (Tracker tab)

## Layout
```
app.py                  UI (Screen / Tracker / Status tabs)
core/
  settings.py           defaults (top N, earnings window, ATR stop multiple, …)
  data.py               data providers: Yahoo (no key) → Alpaca (when keys are set)
  universe.py           ticker lists — add new universes here
  indicators.py         shared indicators (add a column once, every strategy can use it)
  context.py            regime, breadth, VIX, sector strength, relative-strength percentiles
  strategy.py           Strategy template + auto-discovery
  pipeline.py           rank → earnings filter → top N → card fields
  earnings.py           Finnhub calendar (key) or Yahoo fallback
  storage.py            Google Sheets log
  outcomes.py           5-day outcome scoring
strategies/             ONE FILE PER SCREENER (auto-loaded)
  _template.py          copy this to add a screener
tests/                  synthetic-data tests (pytest)
```

## Add / remove a screener
- **Add:** copy `strategies/_template.py` → `strategies/my_setup.py`, edit `key`, `name`, `params`, `evaluate()`.
- **Remove:** delete the file, or set `enabled = False`.
- **Tune:** edit the `params` dict at the top of the strategy file.
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
