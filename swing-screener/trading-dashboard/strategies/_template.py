"""TEMPLATE — copy this file to strategies/<your_name>.py (no leading underscore) to add a screener.
Available columns on `d` (see core/indicators.py): open high low close volume vol_proj vol50
sma20 sma50 sma200 ema10 ema20 atr rsi2 rsi14 hi20_prior lo20_prior hi10 hi252 lo50 ret21 ret63 chg1
Context `ctx`: rs_pct[symbol], sectors, regime_ok, breadth, vix."""
from core.strategy import Setup, Strategy
from strategies._util import ok, scale


class MyStrategy(Strategy):
    key = "my_strategy"          # stable id used in the log
    name = "My strategy"
    description = "One line describing the setup."
    order = 90
    enabled = True
    params = dict(rsi_max=30)

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        if not ok(r.rsi14, r.atr):
            return None
        if not (r.rsi14 < p["rsi_max"]):
            return None
        return Setup(score=100 * scale(r.rsi14, p["rsi_max"], 10),
                     entry=r.high, entry_note="Above today's high",
                     stop_structure=d.low.iloc[-3:].min(), stop_note="Below the 3-day low",
                     notes=[f"RSI(14) {r.rsi14:.0f}"])
