"""Market context shared by all strategies: regime, breadth, VIX, sector strength, relative strength."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import SECTOR_ETF


def _last(df, col):
    try:
        return float(df[col].iloc[-1])
    except Exception:
        return np.nan


def build_context(frames: dict[str, pd.DataFrame], universe: pd.DataFrame, vix: pd.Series | None,
                  settings: dict) -> dict:
    ctx: dict = {"warnings": []}
    spy = frames.get(settings["regime_symbol"])
    spy21 = _last(spy, "ret21") if spy is not None else 0.0
    spy63 = _last(spy, "ret63") if spy is not None else 0.0

    # ── regime
    if spy is not None:
        c, s50, s200 = _last(spy, "close"), _last(spy, "sma50"), _last(spy, "sma200")
        s50_prev = float(spy.sma50.iloc[-11]) if len(spy) > 11 else np.nan
        rising = bool(s50 > s50_prev)
        ctx["spy"] = dict(close=c, sma50=s50, sma200=s200, chg1=_last(spy, "chg1"), sma50_rising=rising)
        ctx["regime_ok"] = bool(c > s50)
        # v2 regime points: above rising 50 SMA 4 · above 50 SMA (not rising) 3 · above 200 only 2 · below 200 0
        if c > s50 and rising:
            ctx["regime_points"], ctx["regime_label"] = 4, "SPY above rising 50 SMA"
        elif c > s50:
            ctx["regime_points"], ctx["regime_label"] = 3, "SPY above flat/falling 50 SMA"
        elif c > s200:
            ctx["regime_points"], ctx["regime_label"] = 2, "SPY above 200 SMA, below 50 SMA"
        else:
            ctx["regime_points"], ctx["regime_label"] = 0, "SPY below 200 SMA"
        if not ctx["regime_ok"]:
            ctx["warnings"].append(f"SPY is below its 50-day SMA ({c:,.2f} < {s50:,.2f}) — breakouts and momentum fail more often.")
    else:
        ctx["regime_ok"] = None
        ctx["warnings"].append("SPY data unavailable — regime check skipped.")

    # ── breadth: % of universe above 50 SMA
    syms = [s for s in universe.symbol if s in frames]
    above = [frames[s].close.iloc[-1] > frames[s].sma50.iloc[-1] for s in syms if not np.isnan(frames[s].sma50.iloc[-1])]
    ctx["breadth"] = 100 * float(np.mean(above)) if above else np.nan
    if ctx["breadth"] < 40:
        ctx["warnings"].append(f"Weak breadth: only {ctx['breadth']:.0f}% of the universe is above its 50-day SMA.")

    # ── VIX
    if vix is not None and len(vix) >= 6:
        v, v5 = float(vix.iloc[-1]), float(vix.iloc[-6])
        ctx["vix"] = dict(level=v, chg5=v - v5)
        if v >= settings["vix_warn_level"]:
            ctx["warnings"].append(f"VIX elevated at {v:.1f}.")
        elif v - v5 >= 4:
            ctx["warnings"].append(f"VIX jumped {v - v5:+.1f} pts in 5 days (now {v:.1f}).")

    # ── sectors
    sec = {}
    for name, etf in SECTOR_ETF.items():
        d = frames.get(etf)
        if d is None:
            continue
        c = _last(d, "close")
        a20, a50 = c > _last(d, "sma20"), c > _last(d, "sma50")
        rs = _last(d, "ret21") - spy21
        state = "Leading" if (a20 and a50 and rs > 0) else "Lagging" if (not a50 and rs < 0) else "Neutral"
        sec[name] = dict(etf=etf, above20=bool(a20), above50=bool(a50), rs21=rs, ret21=_last(d, "ret21"), state=state)
    ctx["sectors"] = sec

    # ── relative strength percentile vs SPY across the universe (blend of 1M and 3M)
    rs = {}
    for s in syms:
        d = frames[s]
        r21, r63 = d.ret21.iloc[-1], d.ret63.iloc[-1]
        if np.isfinite(r21) and np.isfinite(r63):
            rs[s] = 0.5 * (r21 - spy21) + 0.5 * (r63 - spy63)
    rs = pd.Series(rs, dtype=float)
    ctx["rs_pct"] = (rs.rank(pct=True) * 100).to_dict()
    ctx["spy_ret21"] = spy21

    # ── v2: 20/60-day relative performance vs SPY and vs the stock's sector ETF, as percentiles
    spy20 = _last(spy, "ret20") if spy is not None else 0.0
    spy60 = _last(spy, "ret60") if spy is not None else 0.0
    sector_of = dict(zip(universe.symbol, universe.sector))
    etf_ret = {}
    for name, etf in SECTOR_ETF.items():
        d = frames.get(etf)
        if d is not None:
            etf_ret[name] = (_last(d, "ret20"), _last(d, "ret60"))
    ex_spy, ex_sec = {}, {}
    for s in syms:
        d = frames[s]
        r20, r60 = _last(d, "ret20"), _last(d, "ret60")
        if not (np.isfinite(r20) and np.isfinite(r60)):
            continue
        ex_spy[s] = 0.5 * (r20 - spy20) + 0.5 * (r60 - spy60)
        er = etf_ret.get(sector_of.get(s))
        if er and np.isfinite(er[0]) and np.isfinite(er[1]):
            ex_sec[s] = 0.5 * (r20 - er[0]) + 0.5 * (r60 - er[1])
    ctx["rs_spy_excess"], ctx["rs_sector_excess"] = ex_spy, ex_sec
    ctx["rs_spy_pct"] = (pd.Series(ex_spy, dtype=float).rank(pct=True) * 100).to_dict()
    ctx["rs_sector_pct"] = (pd.Series(ex_sec, dtype=float).rank(pct=True) * 100).to_dict()
    return ctx
