# Quality Dip v2 — implementation notes

Spec: "Build Pullback Swing Trading Screener — Version 2.0" (Nov 5, 2026).
Code: `core/qd2.py` (engine) · `strategies/quality_dip.py` (app plug-in) · `core/fundamentals.py` ·
`core/sizing.py` · `core/backtest.py` · `run_quality_dip.py` · `backtest_quality_dip.py`.

## 1. Audit of the existing code (before changes)

| Area | Finding | Action |
|---|---|---|
| Data sources | Prices: Alpaca (if keys) else Yahoo/yfinance. Fundamentals: Yahoo `info` + quarterly income statement, Finnhub ROI. Earnings: Finnhub calendar else Yahoo per ticker. Universe: S&P 500 + Nasdaq-100 (Wikipedia), Russell 1000 from iShares IWB holdings. | Kept. Quality dip now uses **Russell 1000 only** (falls back to S&P 500 + 400 if IWB file fails, shown in Status). |
| Look-ahead | Live indicators are causal (rolling/EWM, `shift`). Outcome scoring already used next-bar fills. | Backtest uses the same causal indicators; a test confirms truncating future data doesn't change a signal. |
| Look-ahead / bias | Fundamentals and constituent lists are *today's* values — using them historically would be look-ahead + survivorship bias. | Backtest only uses fundamentals/membership you supply point-in-time; otherwise runs technical-only and labels the bias. |
| Accounting mix | v1 forward EPS growth = Yahoo `forwardEps` (analyst-adjusted) ÷ `trailingEps` (GAAP). | Replaced with consensus FY+1 ÷ FY0 (same basis). The mixed figure is kept as `fwd_eps_growth_mixed` but never used. |
| Proxies | v1 silently used ROE when ROIC was missing. | ROIC now computed (NOPAT ÷ invested capital). Finnhub ROI / ROE fallbacks are labeled proxies and lower confidence. |
| Inconsistent fields | v1 gap check used 50-day volume; spec uses prior 20-day. Gross margin / sales basis varied silently. | Added `vol20`; every metric now carries a `*_basis` label shown on cards and in the CSV. |
| Missing fields | No 200-SMA-20-ago, 20/60-day returns, sector-relative RS, regime points. | Added (existing indicators/fields untouched). |
| Dividend adjustment | Yahoo `auto_adjust` / Alpaca `adjustment=all` back-adjust prices for dividends — tiny level shifts in old bars, not look-ahead in signals. | Noted; unchanged. |

## 2. Unsupported / blocked metrics (free data)

| Metric | Status |
|---|---|
| Revenue-estimate revisions (30d) | **Blocked** — not in free Yahoo/Finnhub. Always `None`, shown as "not evaluated". |
| Guidance revisions | **Blocked** — no free source. The "confirmed guidance cut" exclusion exists but can never fire. |
| EPS-estimate revisions (30d) | Yahoo `eps_trend` (current vs 30 days ago, FY+1 then FY0). Works when Yahoo has coverage. |
| Point-in-time fundamentals | **Blocked for the live screen** (values are as of today). Backtest accepts your own PIT file. |
| Point-in-time Russell 1000 membership | **Blocked** on free data. Backtest accepts your own membership file; otherwise flags survivorship bias. |
| Historical earnings dates | Not fetched automatically. Backtest accepts `earnings.csv`. |
| TTM gross margin vs prior TTM | Needs 8 quarters; Yahoo often has ~5. Falls back to FY vs prior FY (labeled "not TTM"). |

## 3. Input schemas

**Live screen** — no files; data comes from the providers above.

**Fundamentals metrics** (`core/fundamentals.FIELDS`, fractions not %): `roic, roic_src, roic_is_proxy, sales_growth,
sales_growth_src, sales_growth_latest_q, gm, gm_prior, gm_delta_pts, gm_src, op_margin_delta_pts, fcf_ttm, fcf_basis,
fcf_yield, fcf_margin, de, nd_ebitda, eps_cy, eps_ny, eps_basis, fwd_eps_growth, fwd_pe, peg, peg_meaningful, peg_note,
trailing_eps, eps_rev_30d, rev_rev_30d (None), guidance_cut (None)`.

**Backtest CSVs** (long format):
- `prices.csv`: `date, symbol, open, high, low, close, volume` — include SPY and the 11 SPDR sector ETFs.
- `membership.csv` (optional): `date, symbol, sector` — the constituent list effective from each date.
- `fundamentals.csv` (optional): `asof_date, symbol, roic, roic_is_proxy, sales_growth, sales_growth_latest_q,
  gm_delta_pts, op_margin_delta_pts, fcf_yield, fcf_margin, de, nd_ebitda, fwd_eps_growth, peg, peg_meaningful,
  eps_rev_30d, guidance_cut` — `asof_date` = date the data became public.
- `earnings.csv` (optional): `symbol, date`.

**Output CSV** (`qd2.CSV_COLUMNS`): one row per constituent with every spec field, triggers A–D separately, the three
scores, coverage, confidence, data completeness, stop/targets, resistance, classification, reasons, review notes and
the stage reached (`failed technical screen (fundamentals not fetched)` / `gap exclusion` / `fully evaluated`).

## 4. Implementation compromises

1. **Fundamentals are fetched only for stocks that pass the technical screen** (≈ 5–15% of the universe). Fetching
   all ~1,000 every week would take 15–30 minutes on free APIs. Every constituent still appears in the CSV with its
   technical values and exact exclusion reasons; fundamental columns are blank with the stage explained.
2. **Entry price on live cards = signal-day close** as an estimate of the next open (unknown until tomorrow). The
   backtest uses the actual next-session open.
3. **Regime "above 50 SMA but 50 SMA not rising"** isn't in the spec; it scores 3 of 4.
4. **Warnings penalty**: −2 fundamental points each for material EPS cuts, decelerating sales (latest quarter ≥5 pts
   below TTM), operating margin down ≥2 pts. Gross-margin reduced credit and PEG > 2 are reflected in their own
   components instead of a second penalty.
5. **Gap-downs** ≥5% on ≥2.0× prior 20-day volume are excluded (not queued for review); 1.5–2.0× → manual review flag.
6. **Financials** skip FCF and leverage (n/a) — no sector-specific alternative yet.
7. **Resistance** = nearest of 50 SMA, 60-session high, 52-week high above entry. A 3R target beyond it is flagged with
   the available upside; it is not capped.
8. **Structural stop** uses the 10 sessions ending on the signal day.
9. **Backtest without PIT fundamentals** runs technical + market only (total rescaled to 100) and says so.

## 5. Status

| Functionality | Status |
|---|---|
| Hard technical screen, 200-SMA slope | Implemented + unit-tested |
| Fundamental qualification, unavailable vs failed, financials n/a, proxy labeling | Implemented + unit-tested |
| Fundamental metric computation from statements (ROIC, TTM sales, GM, FCF, EPS basis, PEG rules, revisions) | Implemented + unit-tested on synthetic statements; **not yet run against live Yahoo data** |
| Gross-margin bands, warnings, penalties | Implemented + unit-tested |
| Earnings exclusion / flags / time stop before earnings | Implemented + unit-tested |
| Gap-down exclusion / review | Implemented + unit-tested |
| Triggers A–D, priority ranking | Implemented + unit-tested |
| 40/40/20 scores, progressive scoring, normalization, coverage, confidence | Implemented + unit-tested |
| Stop, ATR-distance flags, 2R/3R, % risk/return, resistance | Implemented + unit-tested |
| Position sizing | Implemented + unit-tested; in app (Screen tab → Position sizing) |
| Full-universe CSV + ranked/exclusion report, five lists | Implemented + tested on a synthetic universe; app download buttons |
| Run on the live Russell 1000 | **Not run** — this workspace can't reach Yahoo/iShares. Run it in the app or `run_quality_dip.py`. |
| Backtest engine (next-open fills, stop-first, gaps, costs, earnings exits, variants, walk-forward OOS, regime/sector breakdowns) | Implemented + unit-tested on synthetic data |
| Backtest results | **Blocked** — needs historical data. Point-in-time fundamentals and membership aren't available free; a technical-only run (`--download`) is possible but biased and labeled so. |
| Revenue-estimate revisions, guidance revisions | **Blocked by data** |
