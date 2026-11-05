"""Quality dip v2 historical backtest.

Uses the SAME engine as the live screener (core/qd2.py) on each historical session, with only data
available at that session's close:
  - indicators are causal (rolling / EWM over past bars only);
  - fundamentals: latest point-in-time row with asof_date ≤ signal date (if supplied);
  - membership: point-in-time list (if supplied) — otherwise a static list and a survivorship-bias flag;
  - earnings: next report date after the signal (if supplied) → exclusion / exit before the report.
Execution: entry at the NEXT session's open (+ slippage); structural stop; target = entry + target_R × risk
computed from the actual fill; if stop and target are touched in the same bar the stop is assumed first;
gaps through the stop exit at the open; time stop after hold_max sessions; exit at the close of the session
before an earnings report.

Input schemas (CSV, long format):
  prices.csv        date, symbol, open, high, low, close, volume        (include SPY and sector ETFs)
  membership.csv    date, symbol, sector                                 (optional; constituents per date)
  fundamentals.csv  asof_date, symbol, roic, roic_is_proxy, sales_growth, sales_growth_latest_q, gm_delta_pts,
                    op_margin_delta_pts, fcf_yield, fcf_margin, de, nd_ebitda, fwd_eps_growth, peg,
                    peg_meaningful, eps_rev_30d, guidance_cut                   (optional; fractions, not %)
  earnings.csv      symbol, date                                         (optional; historical report dates)
"""
from __future__ import annotations

import bisect
import datetime as dt
import math

import numpy as np
import pandas as pd

from . import indicators, qd2
from .settings import SECTOR_ETF

FUND_COLS = ["roic", "roic_is_proxy", "sales_growth", "sales_growth_latest_q", "gm_delta_pts", "op_margin_delta_pts",
             "fcf_yield", "fcf_margin", "de", "nd_ebitda", "fwd_eps_growth", "peg", "peg_meaningful",
             "eps_rev_30d", "guidance_cut"]


# ───────────────────────── data prep
def load_prices_csv(path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    out = {}
    for s, g in df.groupby("symbol"):
        g = g.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
        out[str(s)] = g
    return out


def prepare(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for s, d in prices.items():
        d = d.copy()
        d["vol_proj"] = d.volume
        out[s] = indicators.add_all(d)
    return out


class PITFundamentals:
    def __init__(self, df: pd.DataFrame | None):
        self.by = {}
        if df is not None and len(df):
            df = df.copy()
            df["asof_date"] = pd.to_datetime(df["asof_date"])
            for s, g in df.sort_values("asof_date").groupby("symbol"):
                self.by[str(s)] = (list(g.asof_date), g.to_dict("records"))

    @property
    def available(self) -> bool:
        return bool(self.by)

    def get(self, sym: str, day: pd.Timestamp) -> dict | None:
        if sym not in self.by:
            return None
        dates, rows = self.by[sym]
        i = bisect.bisect_right(dates, day) - 1
        if i < 0:
            return None
        r = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in rows[i].items() if k in FUND_COLS}
        r.setdefault("peg_meaningful", r.get("peg") is not None)
        r["roic_src"] = "point-in-time input"
        return r


class Earnings:
    def __init__(self, df: pd.DataFrame | None):
        self.by = {}
        if df is not None and len(df):
            for s, g in df.groupby("symbol"):
                self.by[str(s)] = sorted(pd.to_datetime(g["date"]).dt.date)

    @property
    def available(self) -> bool:
        return bool(self.by)

    def next_after(self, sym: str, day: dt.date) -> dt.date | None:
        lst = self.by.get(sym, [])
        i = bisect.bisect_left(lst, day)
        return lst[i] if i < len(lst) else None


# ───────────────────────── execution
def simulate_trade(bars: pd.DataFrame, fill: float, stop: float, target: float, hold_max: int,
                   exit_before: dt.date | None = None, slip_bps: float = 5.0, commission_ps: float = 0.0) -> dict:
    """bars start with the ENTRY session (filled at its open). Stop checked before target on every bar."""
    risk = fill - stop
    slip = slip_bps / 1e4
    for i, (day, b) in enumerate(bars.iterrows()):
        if i >= hold_max:
            break
        later = i > 0
        if b.low <= stop:
            px = float(b.open) if (later and b.open < stop) else stop
            return _close(fill, px, risk, slip, commission_ps, day, "stop", i + 1)
        if b.high >= target:
            px = float(b.open) if (later and b.open > target) else target
            return _close(fill, px, risk, slip, commission_ps, day, "target", i + 1)
        if exit_before and day.date() >= exit_before:      # flat at the close of the session before earnings
            return _close(fill, float(b.close), risk, slip, commission_ps, day, "earnings exit", i + 1)
    n = min(hold_max, len(bars))
    if n == 0:
        return None
    if len(bars) < hold_max:
        return dict(open=True)                 # still open at the end of the data — not counted
    return _close(fill, float(bars.iloc[n - 1].close), risk, slip, commission_ps, bars.index[n - 1], "time", n)


def _close(fill, px, risk, slip, comm, day, why, n):
    gross = (px * (1 - slip) - fill)
    r = (gross - 2 * comm) / risk
    return dict(exit_date=day.date(), exit_price=px * (1 - slip), reason=why, R=r, sessions=n)


# ───────────────────────── context per session
def _panel(frames, col, dates, syms):
    return pd.DataFrame({s: frames[s][col].reindex(dates) for s in syms if s in frames})


def build_daily_context(frames, dates, syms):
    spy = frames["SPY"].reindex(dates)
    r20, r60 = _panel(frames, "ret20", dates, syms), _panel(frames, "ret60", dates, syms)
    ex_spy = 0.5 * (r20.sub(spy.ret20, axis=0)) + 0.5 * (r60.sub(spy.ret60, axis=0))
    etf = {n: frames[e].reindex(dates) for n, e in SECTOR_ETF.items() if e in frames}
    return dict(spy=spy, s50_prev=frames["SPY"].sma50.shift(10).reindex(dates), ex_spy=ex_spy, r20=r20, r60=r60, etf=etf)


def ctx_for(day, base, members, sector_of):
    spy = base["spy"].loc[day]
    c, s50, s200 = spy.close, spy.sma50, spy.sma200
    rising = s50 > base["s50_prev"].loc[day]
    pts, lab = ((4, "SPY above rising 50 SMA") if c > s50 and rising else (3, "SPY above flat/falling 50 SMA") if c > s50
                else (2, "SPY above 200 SMA, below 50 SMA") if c > s200 else (0, "SPY below 200 SMA"))
    exs = base["ex_spy"].loc[day].reindex(list(members)).dropna()
    sec = {}
    for s in exs.index:
        e = base["etf"].get(sector_of.get(s))
        if e is not None:
            er = e.loc[day]
            v = 0.5 * (base["r20"].at[day, s] - er.ret20) + 0.5 * (base["r60"].at[day, s] - er.ret60)
            if np.isfinite(v):
                sec[s] = v
    sec = pd.Series(sec, dtype=float)
    return dict(regime_points=pts, regime_label=lab, rs_spy_pct=(exs.rank(pct=True) * 100).to_dict(),
                rs_sector_pct=(sec.rank(pct=True) * 100).to_dict())


# ───────────────────────── backtest
def run(frames: dict, universe: pd.DataFrame | None = None, membership: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None, earnings: pd.DataFrame | None = None, p=qd2.PARAMS,
        target_r: float = 3.0, classes=("Highest priority", "Strong"), start=None, end=None,
        slip_bps: float = 5.0, commission_ps: float = 0.005, fundamentals_mode: str = "auto") -> dict:
    """frames: prepared (indicator) frames incl. SPY + sector ETFs.
    fundamentals_mode: 'auto' (use PIT data if given, else skip), 'skip' (technical + market only)."""
    pit, ern = PITFundamentals(fundamentals), Earnings(earnings)
    skip_f = fundamentals_mode == "skip" or (fundamentals_mode == "auto" and not pit.available)
    dates = frames["SPY"].index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if membership is not None and len(membership):
        m = membership.copy()
        m["date"] = pd.to_datetime(m["date"])
        mdates = sorted(m.date.unique())
        mem = {d: set(g.symbol) for d, g in m.groupby("date")}
        sec_map = dict(zip(m.symbol, m.sector)) if "sector" in m else {}

        def members_on(day):
            i = bisect.bisect_right(mdates, day) - 1
            return mem[mdates[i]] if i >= 0 else set()
        survivorship = False
    else:
        static = set(universe.symbol) if universe is not None else {s for s in frames if s != "SPY" and s not in SECTOR_ETF.values()}
        sec_map = dict(zip(universe.symbol, universe.sector)) if universe is not None else {}

        def members_on(day):
            return static
        survivorship = True
    syms = sorted({s for day in dates[::21] for s in members_on(day)} | set(members_on(dates[-1]))) if len(dates) else []
    base = build_daily_context(frames, dates, [s for s in syms if s in frames])

    # cheap causal prefilter columns (hard technical + 'something reversed today')
    pre = {}
    for s in base["ex_spy"].columns:
        d = frames[s]
        off = 1 - d.close / d.hi252
        hard = ((off >= p["off_high_min"]) & (off <= p["off_high_max"]) & (d.close < d.sma50) & (d.close > d.sma200)
                & (d.sma200 >= d.sma200_20ago))
        rev = (d.close > d.high.shift(1)) | ((d.close > d.close.shift(1)) & (d.close > d.ema10) & (d.close.shift(1) <= d.ema10.shift(1)))
        pre[s] = (hard & rev).reindex(dates).fillna(False)
    pre = pd.DataFrame(pre)

    trades, open_until = [], {}
    pos_index = {s: {d: i for i, d in enumerate(frames[s].index)} for s in pre.columns}
    for day in dates[:-1]:
        row = pre.loc[day]
        cands = [s for s in row.index[row.values] if s in members_on(day) and open_until.get(s, pd.Timestamp.min) < day]
        if not cands:
            continue
        ctx = ctx_for(day, base, members_on(day), sec_map)
        for s in cands:
            d = frames[s]
            i = pos_index[s].get(day)
            if i is None or i + 1 >= len(d) or i < 260:
                continue
            hist = d.iloc[: i + 1]
            f = None if skip_f else pit.get(s, day)
            earn = (ern.next_after(s, day.date()), True) if ern.available else (None, False)
            pp = dict(p, max_unavailable=99) if skip_f else p
            r = qd2.evaluate(s, hist, sec_map.get(s), f, ctx, earn, day.date(), pp)
            if skip_f:   # technical + market only, rescaled to 100
                r["total"] = (r["tech_score"]["score"] + r["mkt_score"]["score"]) * 100 / 60
                r["classification"] = qd2.classify(r["total"], bool([e for e in r["exclusions"] if "unavailable" not in e]),
                                                   r["trig_rank"] is not None, r["risk"]["risk_ok"], p)
            if r["classification"] not in classes:
                continue
            nxt = d.iloc[i + 1]
            fill = float(nxt.open) * (1 + slip_bps / 1e4)
            stop = r["risk"]["stop"]
            if fill <= stop:
                continue                                  # gapped below the stop before entry
            risk = fill - stop
            res = simulate_trade(d.iloc[i + 1:], fill, stop, fill + target_r * risk, p["hold_max"],
                                 qd2.bday_before(earn[0]) if earn[0] else None, slip_bps, commission_ps)
            if not res or res.get("open"):
                continue
            trades.append(dict(symbol=s, sector=sec_map.get(s), signal_date=day.date(), entry_date=d.index[i + 1].date(),
                               entry=fill, stop=stop, target=fill + target_r * risk, score=r["total"],
                               classification=r["classification"], trigger=r["trig_label"],
                               regime=ctx["regime_label"], **res))
            open_until[s] = pd.Timestamp(res["exit_date"])
    t = pd.DataFrame(trades)
    return dict(trades=t, metrics=metrics(t, dates), survivorship_bias=survivorship,
                fundamentals_used=not skip_f, earnings_used=ern.available, sessions=len(dates))


def metrics(t: pd.DataFrame, dates=None) -> dict:
    if t is None or t.empty:
        return dict(trades=0)
    w, l = t[t.R > 0], t[t.R <= 0]
    cum = t.sort_values("exit_date").R.cumsum()
    dd = float((cum.cummax().clip(lower=0) - cum).max())
    exposure = None
    if dates is not None and len(dates):
        days = pd.Series(0, index=pd.DatetimeIndex(dates))
        for _, r in t.iterrows():
            days[(days.index >= pd.Timestamp(r.entry_date)) & (days.index <= pd.Timestamp(r.exit_date))] += 1
        exposure = dict(avg_open_positions=float(days.mean()), pct_sessions_invested=float((days > 0).mean() * 100))
    return dict(trades=len(t), win_rate=100 * len(w) / len(t), avg_win_R=float(w.R.mean()) if len(w) else 0.0,
                avg_loss_R=float(l.R.mean()) if len(l) else 0.0, expectancy_R=float(t.R.mean()),
                profit_factor=float(w.R.sum() / -l.R.sum()) if len(l) and l.R.sum() < 0 else float("inf"),
                max_drawdown_R=dd, avg_hold_sessions=float(t.sessions.mean()), exposure=exposure,
                by_reason=t.reason.value_counts().to_dict())


def breakdown(t: pd.DataFrame, col: str) -> pd.DataFrame:
    if t is None or t.empty:
        return pd.DataFrame()
    g = t.groupby(col).R
    return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda x: 100 * (x > 0).mean()).round(1),
                         "expectancy_R": g.mean().round(3), "total_R": g.sum().round(2)}).reset_index()


VARIANTS = {
    "revised_3R": dict(params=qd2.PARAMS, target_r=3.0),
    "revised_2R": dict(params=qd2.PARAMS, target_r=2.0),
    "strict_3R": dict(params=qd2.STRICT, target_r=3.0),
    "no_sector_rs_3R": dict(params=dict(qd2.PARAMS, use_sector_rs=False), target_r=3.0),
    "volume_required_3R": dict(params=dict(qd2.PARAMS, require_volume=True), target_r=3.0),
    "wider_stop_3R": dict(params=dict(qd2.PARAMS, stop_buffer_atr=0.5), target_r=3.0),
    "tight_atr_cap_3R": dict(params=dict(qd2.PARAMS, atr_max=2.0), target_r=3.0),
}


def run_variants(frames, variants=None, **kw) -> dict:
    out = {}
    for name, v in (variants or VARIANTS).items():
        out[name] = run(frames, p=v["params"], target_r=v["target_r"], **kw)
    return out


def walk_forward(results: dict, folds: int = 4, min_trades: int = 20) -> pd.DataFrame:
    """Expanding-window selection: for fold k, pick the variant with the best expectancy on folds < k
    (≥ min_trades), then report that variant's results on fold k only. Selection never sees fold k."""
    all_t = [r["trades"] for r in results.values() if len(r["trades"])]
    if not all_t:
        return pd.DataFrame()
    exits = pd.concat(all_t).signal_date
    edges = pd.to_datetime(pd.Series(exits)).quantile(np.linspace(0, 1, folds + 1)).dt.date.tolist()
    rows = []
    for k in range(1, folds):
        lo, hi = edges[k], edges[k + 1]
        best, best_e = None, -1e9
        for name, r in results.items():
            t = r["trades"]
            if t.empty:
                continue
            ins = t[t.signal_date < lo]
            if len(ins) >= min_trades and ins.R.mean() > best_e:
                best, best_e = name, ins.R.mean()
        if best is None:
            rows.append(dict(fold=k, oos_start=lo, oos_end=hi, chosen=None, note="not enough in-sample trades"))
            continue
        t = results[best]["trades"]
        oos = t[(t.signal_date >= lo) & (t.signal_date <= hi)]
        m = metrics(oos)
        rows.append(dict(fold=k, oos_start=lo, oos_end=hi, chosen=best, in_sample_expectancy_R=round(best_e, 3),
                         oos_trades=m.get("trades", 0), oos_expectancy_R=round(m.get("expectancy_R", 0), 3) if m.get("trades") else None,
                         oos_win_rate=round(m.get("win_rate", 0), 1) if m.get("trades") else None))
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "—"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(out)


def summary_markdown(results: dict, wf: pd.DataFrame) -> str:
    first = next(iter(results.values()))
    lines = ["# Quality dip v2 — backtest", "",
             f"- Sessions: {first['sessions']}",
             f"- Survivorship bias: {'YES — static (current) constituent list' if first['survivorship_bias'] else 'no — point-in-time membership'}",
             f"- Fundamentals: {'point-in-time input' if first['fundamentals_used'] else 'NOT USED (technical + market only)'}",
             f"- Earnings dates: {'historical input' if first['earnings_used'] else 'NOT AVAILABLE — no earnings exclusion/exit'}",
             "- Entry next-session open + slippage; stop before target on the same bar; gaps through the stop fill at the open.",
             "", "| Variant | Trades | Win % | Avg win R | Avg loss R | Expectancy R | Profit factor | Max DD R | Avg hold |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        m = r["metrics"]
        if not m.get("trades"):
            lines.append(f"| {n} | 0 | | | | | | | |")
            continue
        lines.append(f"| {n} | {m['trades']} | {m['win_rate']:.1f} | {m['avg_win_R']:.2f} | {m['avg_loss_R']:.2f} | "
                     f"{m['expectancy_R']:.3f} | {m['profit_factor']:.2f} | {m['max_drawdown_R']:.1f} | {m['avg_hold_sessions']:.1f} |")
    lines += ["", "Full-period variant rows are descriptive (no parameters were fitted). Walk-forward (out-of-sample):", ""]
    lines.append(md_table(wf) if len(wf) else "Not enough trades for walk-forward.")
    base = results.get("revised_3R")
    if base and len(base["trades"]):
        lines += ["", "## revised_3R by market regime", "", md_table(breakdown(base["trades"], "regime")),
                  "", "## revised_3R by sector", "", md_table(breakdown(base["trades"], "sector"))]
    return "\n".join(lines)
