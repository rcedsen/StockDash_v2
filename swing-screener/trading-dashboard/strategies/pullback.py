from core.strategy import Setup, Strategy
from strategies._util import ok, scale


class Pullback(Strategy):
    key = "pullback"
    name = "Pullback"
    description = "Uptrend (50 > 200 SMA) pulling back to the 20 EMA / 50 SMA on lighter volume."
    order = 20
    params = dict(
        touch_atr=1.0,        # today's low within this many ATRs of the 20 EMA or 50 SMA
        min_pullback_atr=1.0, # close at least this far below the 10-day high
        slope_days=5,
    )

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        if len(d) < 210 or not ok(r.sma50, r.sma200, r.ema20, r.atr, r.vol50, r.hi10):
            return None
        sma50_prev = d.sma50.iloc[-1 - p["slope_days"]]
        touch = min(abs(r.low - r.ema20), abs(r.low - r.sma50)) / r.atr
        depth = (r.hi10 - r.close) / r.atr
        vol3 = (d.volume.iloc[-3:-1].sum() + r.vol_proj) / 3
        if not (r.close > r.sma50 > r.sma200 and r.sma50 > sma50_prev
                and touch <= p["touch_atr"] and depth >= p["min_pullback_atr"] and vol3 < r.vol50):
            return None
        rs = ctx["rs_pct"].get(symbol, 50)
        trend = scale((r.sma50 - r.sma200) / r.sma200, 0, 0.15)
        depth_sc = 1 - min(abs(depth - 2.25) / 1.75, 1)        # sweet spot ≈ 1.5–3 ATR
        rng = r.high - r.low
        cl = (r.close - r.low) / rng if rng > 0 else 0.5
        reversal = 1.0 if (r.close > r.open and cl >= 0.5) else 0.5 if cl >= 0.5 else 0.0
        score = 30 * trend + 25 * scale(rs, 50, 100) + 20 * depth_sc + 25 * reversal
        swing_low = d.low.iloc[-5:].min()
        lvl = "20 EMA" if abs(r.low - r.ema20) <= abs(r.low - r.sma50) else "50 SMA"
        return Setup(
            score=score,
            entry=r.high, entry_note="Buy-stop above today's high (confirms the bounce)",
            stop_structure=swing_low, stop_note="Below the 5-day swing low",
            notes=[f"Tagged the {lvl} ({touch:.1f} ATR)", f"Pulled back {depth:.1f} ATR from 10-day high",
                   f"3-day volume {vol3 / r.vol50:.2f}× avg", f"RS {rs:.0f}th pct",
                   "Reversal candle today" if reversal == 1 else "No reversal candle yet"],
            target_hint=r.hi10, target_note="Prior 10-day high",
        )
