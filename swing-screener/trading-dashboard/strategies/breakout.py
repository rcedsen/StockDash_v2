from core.strategy import Setup, Strategy
from strategies._util import ok, scale


class Breakout(Strategy):
    key = "breakout"
    name = "Breakout"
    description = "Close above the 20-day high on heavy volume, out of a tight base, in an uptrend."
    order = 10
    params = dict(
        vol_mult=1.5,        # projected volume ≥ this × 50-day average
        base_max_atr=4.0,    # prior 20-day range ≤ this many ATRs (lower = tighter base)
        slope_days=5,        # 50 SMA must be higher than it was this many days ago
    )

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        if len(d) < 60 or not ok(r.hi20_prior, r.lo20_prior, r.vol50, r.sma50, r.atr, r.vol_proj):
            return None
        sma50_prev = d.sma50.iloc[-1 - p["slope_days"]]
        base = (r.hi20_prior - r.lo20_prior) / r.atr
        vol_ratio = r.vol_proj / r.vol50
        if not (r.close > r.hi20_prior and vol_ratio >= p["vol_mult"]
                and r.close > r.sma50 and r.sma50 > sma50_prev and base <= p["base_max_atr"]):
            return None
        rs = ctx["rs_pct"].get(symbol, 50)
        rng = r.high - r.low
        cl = (r.close - r.low) / rng if rng > 0 else 0.5
        score = (35 * scale(vol_ratio, p["vol_mult"], 3.0) + 25 * scale(base, p["base_max_atr"], 2.0)
                 + 25 * scale(rs, 50, 100) + 15 * cl)
        return Setup(
            score=score,
            entry=r.high, entry_note="Buy-stop above today's high (or a retest of the breakout level)",
            stop_structure=r.low, stop_note="Below the breakout-day low",
            notes=[f"Cleared 20-day high {r.hi20_prior:,.2f}",
                   f"Volume {vol_ratio:.1f}× avg", f"Base {base:.1f} ATR wide",
                   f"RS {rs:.0f}th pct", f"Closed in top {100 - cl * 100:.0f}% of range"],
            target_hint=None,
        )
