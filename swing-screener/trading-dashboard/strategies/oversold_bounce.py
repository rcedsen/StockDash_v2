from core.strategy import Setup, Strategy
from strategies._util import ok, scale


class OversoldBounce(Strategy):
    key = "oversold"
    name = "Oversold bounce"
    description = "Above the 200 SMA but short-term stretched down (RSI(2) < 10 or 2+ ATR below the 10 EMA)."
    order = 30
    regime_sensitive = False
    params = dict(
        rsi2_max=10,      # RSI(2) below this qualifies
        stretch_atr=2.0,  # or this many ATRs below the 10 EMA
        support_atr=1.0,  # "near support" = within this many ATRs of 50 SMA / 200 SMA / 50-day low
    )
    hold = (2, 5)
    target_r = 1.5
    atr_stop_mult = 1.5
    stop_rule = "Below the 3-day low"
    rules = [
        "Close > 200-day SMA (long-term uptrend intact)",
        "RSI(2) < {rsi2_max}, OR close ≥ {stretch_atr} × ATR(14) below the 10 EMA",
    ]
    score_parts = [
        "20% RSI(2) depth below {rsi2_max}",
        "20% stretch below the 10 EMA (1 → 3 ATR)",
        "30% proximity of today's low to support (50 SMA, 200 SMA or 50-day low; 0 → {support_atr}×2 ATR)",
        "15% 200 SMA rising over the last 20 sessions",
        "15% distance above the 200 SMA (0 → 20%)",
    ]
    trigger = "Buy-stop above today's high — don't buy while it's still falling"
    best_when = "Bull markets with brief shakeouts; VIX spiking then turning down"
    fails_when = "Bear markets / stocks with a falling 200 SMA — 'oversold' keeps getting more oversold"
    changed = "2026-11-03"

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        if len(d) < 220 or not ok(r.sma200, r.ema10, r.atr, r.rsi2, r.sma50, r.lo50):
            return None
        stretch = (r.ema10 - r.close) / r.atr
        if not (r.close > r.sma200 and (r.rsi2 < p["rsi2_max"] or stretch >= p["stretch_atr"])):
            return None
        sup = {"50 SMA": r.sma50, "200 SMA": r.sma200, "50-day low": r.lo50}
        name, lvl = min(sup.items(), key=lambda kv: abs(r.low - kv[1]))
        sup_d = abs(r.low - lvl) / r.atr
        sma200_up = r.sma200 > d.sma200.iloc[-21]
        score = (20 * scale(p["rsi2_max"] - r.rsi2, 0, p["rsi2_max"]) + 20 * scale(stretch, 1, 3)
                 + 30 * scale(sup_d, 2 * p["support_atr"], 0)
                 + 15 * sma200_up + 15 * scale((r.close - r.sma200) / r.sma200, 0, 0.2))
        return Setup(
            score=score,
            entry=r.high, entry_note="Buy-stop above today's high (wait for the turn)",
            stop_structure=d.low.iloc[-3:].min(), stop_note="Below the 3-day low",
            notes=[f"RSI(2) {r.rsi2:.0f}", f"{stretch:.1f} ATR below 10 EMA",
                   f"Nearest support: {name} ({sup_d:.1f} ATR)",
                   "200 SMA rising" if sma200_up else "200 SMA flat/falling"],
            target_hint=r.ema10, target_note="Reversion to 10 EMA",
        )
