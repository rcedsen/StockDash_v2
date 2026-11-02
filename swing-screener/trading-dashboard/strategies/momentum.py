from core.indicators import trend_r2
from core.strategy import Setup, Strategy
from strategies._util import ok, scale


class Momentum(Strategy):
    key = "momentum"
    name = "Momentum"
    description = "Top-20% relative strength vs SPY, within 5% of the 52-week high, above a rising 20 EMA."
    order = 40
    params = dict(
        near_high=0.05,   # within this fraction of the 52-week high
        rs_min=80,        # relative-strength percentile (blend of 1M and 3M vs SPY)
        slope_days=5,
    )

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        rs = ctx["rs_pct"].get(symbol)
        if len(d) < 210 or rs is None or not ok(r.hi252, r.ema20, r.ema10, r.atr):
            return None
        off_high = 1 - r.close / r.hi252
        ema20_prev = d.ema20.iloc[-1 - p["slope_days"]]
        if not (off_high <= p["near_high"] and rs >= p["rs_min"] and r.close > r.ema20 and r.ema20 > ema20_prev):
            return None
        r2 = trend_r2(d.close, 63)
        score = (40 * scale(rs, p["rs_min"], 100) + 30 * (r2 if r2 == r2 else 0)
                 + 15 * scale(off_high, p["near_high"], 0) + 15 * (r.close > r.ema10))
        return Setup(
            score=score,
            entry=r.high, entry_note="Buy-stop above today's high, or a dip to the 10 EMA",
            stop_structure=d.low.iloc[-5:].min(), stop_note="Below the 5-day low",
            notes=[f"RS {rs:.0f}th pct", f"{off_high * 100:.1f}% off 52-week high",
                   f"Trend smoothness R² {r2:.2f}", f"1M {r.ret21 * 100:+.1f}% · 3M {r.ret63 * 100:+.1f}%"],
        )
