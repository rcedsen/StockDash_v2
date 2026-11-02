"""Score logged signals after the fact.

Rules (same for every strategy):
- Entry fills when price trades through the entry trigger (at the open if it gaps above).
- Stop = structure stop; target = entry + target_R × risk (default 2R).
- If stop and target are both touched on one bar, the stop is assumed first (conservative).
- Still open after `hold_days` trading days → exit at that day's close.
- Never triggered within `hold_days` → 'no trigger'.
"""
from __future__ import annotations

import pandas as pd


def score_signal(entry: float, stop: float, target: float, bars: pd.DataFrame, hold_days: int) -> dict | None:
    """bars = completed daily bars AFTER the signal date. Returns None while still unresolved."""
    risk0 = entry - stop
    if not risk0 > 0:
        return dict(status="closed", outcome="invalid", outcome_r=None)
    bars = bars.iloc[:hold_days]
    fill, hi, lo, fill_i = None, None, None, None
    for i, (day, b) in enumerate(bars.iterrows()):
        if fill is None:
            if b.high < entry:
                continue
            fill, fill_i = max(entry, b.open), i
            hi, lo = b.high, b.low
        else:
            hi, lo = max(hi, b.high), min(lo, b.low)
        risk = fill - stop
        mfe, mae = (hi - fill) / risk, (fill - lo) / risk
        later = i > fill_i          # on later bars a gap through stop/target fills at the open
        if b.low <= stop:
            px = b.open if (later and b.open < stop) else stop
            return dict(status="closed", outcome="stop", outcome_r=(px - fill) / risk,
                        exit_date=day.date(), mfe_r=mfe, mae_r=mae)
        if b.high >= target:
            px = b.open if (later and b.open > target) else max(target, fill)
            return dict(status="closed", outcome="target", outcome_r=(px - fill) / risk,
                        exit_date=day.date(), mfe_r=mfe, mae_r=mae)
    if len(bars) < hold_days:
        return None
    if fill is None:
        return dict(status="closed", outcome="no trigger", outcome_r=None, exit_date=bars.index[-1].date())
    last = bars.iloc[-1]
    risk = fill - stop
    return dict(status="closed", outcome="time", outcome_r=(last.close - fill) / risk, exit_date=bars.index[-1].date(),
                mfe_r=(hi - fill) / risk, mae_r=(fill - lo) / risk)


def update_outcomes(sig: pd.DataFrame, history: dict[str, pd.DataFrame], hold_days: int) -> tuple[pd.DataFrame, int]:
    sig = sig.copy()
    n = 0
    for i, row in sig.iterrows():
        if str(row.get("status", "")).strip() not in ("", "open"):
            continue
        h = history.get(row.symbol)
        try:
            d0 = pd.Timestamp(row.bar_date)
            res = score_signal(float(row.entry), float(row.stop_structure), float(row.target_2r),
                               h[h.index > d0] if h is not None else pd.DataFrame(), hold_days)
        except Exception:
            continue
        if res:
            for k, v in res.items():
                sig.at[i, k] = v
            n += 1
        else:
            sig.at[i, "status"] = "open"
    return sig, n


def summarize(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame()
    s = sig.copy()
    s["outcome_r"] = pd.to_numeric(s.outcome_r, errors="coerce")
    g = s.groupby("strategy")
    out = pd.DataFrame({
        "signals": g.size(),
        "closed": g.apply(lambda x: (x.status == "closed").sum()),
        "triggered": g.apply(lambda x: x.outcome.isin(["stop", "target", "time"]).sum()),
        "hit_target": g.apply(lambda x: (x.outcome == "target").sum()),
        "avg_R": g.outcome_r.mean().round(2),
        "total_R": g.outcome_r.sum().round(1),
    })
    return out.reset_index()
