"""Position sizing: fixed-fractional risk with an optional position-size cap.

dollar_risk = equity × risk_pct;  shares = floor(dollar_risk ÷ (entry − stop)).
The default 0.5% risk is an example, not a recommendation — set your own in the app."""
from __future__ import annotations

import math

DEFAULT_RISK_PCT = 0.5


def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  max_position_pct: float | None = None) -> dict:
    """risk_pct and max_position_pct are percentages (0.5 = 0.5%). Returns shares and the numbers behind them."""
    out = dict(shares=0, dollar_risk=0.0, position_value=0.0, actual_risk=0.0, capped=False, note="")
    if not (equity and equity > 0 and risk_pct and risk_pct > 0 and entry and stop is not None):
        out["note"] = "enter account equity and risk %"
        return out
    rps = entry - stop
    if rps <= 0:
        out["note"] = "stop is not below entry"
        return out
    dollar_risk = equity * risk_pct / 100
    shares = math.floor(dollar_risk / rps)
    capped = False
    if max_position_pct and max_position_pct > 0:
        cap = math.floor(equity * max_position_pct / 100 / entry)
        if cap < shares:
            shares, capped = cap, True
    out.update(shares=int(max(shares, 0)), dollar_risk=dollar_risk, position_value=shares * entry,
               actual_risk=shares * rps, capped=capped,
               note=("capped by max position size" if capped else "") or ("risk too small for 1 share" if shares < 1 else ""))
    return out
