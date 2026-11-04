from core.strategy import Setup, Strategy
from strategies._util import ok, scale


def _pct(x, nd=0):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


class QualityDip(Strategy):
    key = "quality_dip"
    name = "Quality dip"
    description = "Russell 1000 · profitable fast growers 15–30% off their high, below the 50 SMA but above the 200 SMA."
    order = 50
    regime_sensitive = False
    universe = ["sp500", "ndx100", "russell1000"]   # mid-cap growers too; S&P labels win on overlap
    params = dict(
        # price (prefilter — checked before any fundamentals are fetched)
        off_high_min=0.15, off_high_max=0.30,
        # fundamentals
        roic_min=0.15,
        gm_drop_max_pts=1.0,          # gross margin may slip at most this many points Y/Y
        fcf_yield_min=0.02, fcf_margin_min=0.15,
        de_max=0.8, nd_ebitda_max=2.0,
        sales_growth_min=0.15, fwd_eps_growth_min=0.15,
        peg_max=1.5, peg_max_fast=2.0, fast_growth=0.25,
        max_missing=1,                # metrics allowed to be unavailable (shown as n/a)
        # "why did it dip" check
        gap_down=0.05, gap_vol_mult=2.0, gap_lookback=15, exclude_gap=True,
        # timing trigger (Ready badge)
        rsi2_trigger=10, sma200_touch_atr=1.0,
    )
    hold = (10, 30)
    target_r = 3.0
    atr_stop_mult = 2.0
    stop_rule = "Below the 10-day low"
    rules = [
        "PRICE · Close {off_high_min:.0%}–{off_high_max:.0%} below the 52-week high",
        "PRICE · Close < 50-day SMA and > 200-day SMA (short-term dip inside a long-term uptrend)",
        "QUALITY · ROIC ≥ {roic_min:.0%} (Finnhub ROI TTM; ROE used as proxy without a Finnhub key)",
        "QUALITY · Gross margin stable or expanding: TTM ≥ prior-year TTM − {gm_drop_max_pts} pt",
        "QUALITY · FCF yield ≥ {fcf_yield_min:.0%} OR FCF margin ≥ {fcf_margin_min:.0%}",
        "QUALITY · Debt/Equity ≤ {de_max} OR net debt/EBITDA ≤ {nd_ebitda_max}",
        "GROWTH · Sales growth ≥ {sales_growth_min:.0%} Y/Y (TTM when 8 quarters are available)",
        "GROWTH · Forward EPS growth ≥ {fwd_eps_growth_min:.0%} (next-FY EPS estimate vs trailing EPS)",
        "VALUE · PEG ≤ {peg_max} (≤ {peg_max_fast} when forward growth ≥ {fast_growth:.0%}); PEG = forward P/E ÷ forward EPS growth",
        "At most {max_missing} metric may be unavailable; every available metric must pass",
        "EXCLUDE · Any gap-down ≥ {gap_down:.0%} on ≥ {gap_vol_mult}× volume in the last {gap_lookback} sessions (likely earnings/news re-rating, not a dip)",
    ]
    trigger = ("'Ready' badge when ANY fires: (a) RSI(2) < {rsi2_trigger} yesterday and today closes above "
               "yesterday's high; (b) close reclaims the 10 EMA; (c) a low within {sma200_touch_atr} ATR of the "
               "200 SMA in the last 5 sessions and today closes above yesterday's high. Otherwise 'Setting up' "
               "— watch, don't buy yet.")
    score_parts = [
        "50% QUALITY/GROWTH: ROIC 15→35% (10), sales growth 15→40% (10), fwd EPS growth 15→40% (8), "
        "PEG 1.5→0.8 (8), FCF yield 2→6% (7), gross-margin trend (4), low leverage (3)",
        "50% ENTRY: trigger fired (20), relative strength percentile (10), dip depth sweet spot ≈20% off high (10), "
        "proximity to the 200 SMA (10)",
    ]
    best_when = "Market pullbacks / sector rotations that drag good companies down with everything else"
    fails_when = "Company-specific bad news (guidance cuts) — the gap check filters the obvious ones"
    data = "Daily price & volume + quarterly fundamentals (Yahoo; Finnhub ROI if key set), cached 7 days"
    changed = "2026-11-03"
    needs = ("fundamentals",)

    def prefilter(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        if len(d) < 210 or not ok(r.hi252, r.sma50, r.sma200):
            return False
        off = 1 - r.close / r.hi252
        return p["off_high_min"] <= off <= p["off_high_max"] and r.sma200 < r.close < r.sma50

    def evaluate(self, d, symbol, ctx):
        p, r = self.params, d.iloc[-1]
        f = (ctx.get("fundamentals") or {}).get(symbol)
        if not f or not ok(r.atr, r.ema10, r.vol50):
            return None

        # ── "why it dipped": large gap-down on heavy volume
        w = d.iloc[-p["gap_lookback"]:]
        gaps = (w.open / d.close.shift(1).loc[w.index] - 1)
        heavy = w.volume >= p["gap_vol_mult"] * w.vol50
        gap_hit = bool(((gaps <= -p["gap_down"]) & heavy).any())
        if gap_hit and p["exclude_gap"]:
            return None

        # ── fundamentals: (label, passed or None if missing, display)
        g = f.get("fwd_eps_growth")
        peg_lim = p["peg_max_fast"] if (g is not None and g >= p["fast_growth"]) else p["peg_max"]
        fcf_y, fcf_m = f.get("fcf_yield"), f.get("fcf_margin")
        de, nde = f.get("de"), f.get("nd_ebitda")

        gmd, peg = f.get("gm_delta_pts"), f.get("peg")
        gmd_s = "n/a" if gmd is None else f"{gmd:+.1f} pt"
        peg_s = "n/a" if peg is None else f"{peg:.2f}"
        de_s = "n/a" if de is None else f"{de:.2f}"
        nde_s = "n/a" if nde is None else f"{nde:.1f}"

        def either(a, b):
            vals = [v for v in (a, b) if v is not None]
            return None if not vals else any(vals)

        checks = [
            ("ROIC", None if f.get("roic") is None else f["roic"] >= p["roic_min"],
             f"{f.get('roic_src') or 'ROIC'} {_pct(f.get('roic'))}"),
            ("Gross margin", None if gmd is None else gmd >= -p["gm_drop_max_pts"],
             f"Gross margin {_pct(f.get('gm'))} ({gmd_s} {f.get('gm_src') or ''})".replace(" )", ")")),
            ("FCF", either(None if fcf_y is None else fcf_y >= p["fcf_yield_min"],
                           None if fcf_m is None else fcf_m >= p["fcf_margin_min"]),
             f"FCF yield {_pct(fcf_y, 1)} · FCF margin {_pct(fcf_m)}"),
            ("Leverage", either(None if de is None else 0 <= de <= p["de_max"],
                                None if nde is None else nde <= p["nd_ebitda_max"]),
             f"D/E {de_s} · net debt/EBITDA {nde_s}"),
            ("Sales growth", None if f.get("sales_growth") is None else f["sales_growth"] >= p["sales_growth_min"],
             f"Sales {_pct(f.get('sales_growth'))} ({f.get('sales_growth_src') or 'n/a'})"),
            ("Fwd EPS growth", None if g is None else g >= p["fwd_eps_growth_min"], f"Fwd EPS growth {_pct(g)}"),
            ("PEG", None if peg is None else peg <= peg_lim, f"PEG {peg_s} (max {peg_lim:g})"),
        ]
        if any(c[1] is False for c in checks):
            return None
        missing = [c[0] for c in checks if c[1] is None]
        if len(missing) > p["max_missing"]:
            return None

        # ── timing trigger
        prev = d.iloc[-2]
        t_rsi = ok(prev.rsi2) and prev.rsi2 < p["rsi2_trigger"] and r.close > prev.high
        t_ema = r.close > r.ema10 and prev.close <= prev.ema10
        t_200 = bool(((d.low.iloc[-5:] - d.sma200.iloc[-5:]).abs() <= p["sma200_touch_atr"] * r.atr).any()) \
            and r.close > prev.high
        ready = bool(t_rsi or t_ema or t_200)
        fired = [n for n, v in (("RSI(2) turn", t_rsi), ("10 EMA reclaim", t_ema), ("200 SMA higher low", t_200)) if v]

        # ── score
        z = lambda v: 0 if v is None else v
        fund = (10 * scale(z(f.get("roic")), 0.15, 0.35) + 10 * scale(z(f.get("sales_growth")), 0.15, 0.40)
                + 8 * scale(z(g), 0.15, 0.40) + 8 * scale(f.get("peg") if f.get("peg") is not None else 9, 1.5, 0.8)
                + 7 * scale(z(fcf_y), 0.02, 0.06)
                + 4 * (scale(f["gm_delta_pts"], -1, 2) if f.get("gm_delta_pts") is not None else 0)
                + 3 * (scale(de, 0.8, 0) if de is not None and de >= 0 else scale(z(nde), 2, -1)))
        off = 1 - r.close / r.hi252
        rs = ctx.get("rs_pct", {}).get(symbol, 50)
        prox200 = (r.close - r.sma200) / r.atr
        entry_sc = (20 * ready + 10 * scale(rs, 30, 90) + 10 * (1 - min(abs(off - 0.20) / 0.10, 1))
                    + 10 * scale(prox200, 6, 0.5))
        score = fund + entry_sc

        stop = d.low.iloc[-10:].min()
        notes = [c[2] + ("" if c[1] is not False else " ✗") for c in checks]
        notes += [f"{off * 100:.0f}% off 52-week high", f"{prox200:.1f} ATR above the 200 SMA",
                  f"RS {rs:.0f}th pct",
                  ("Trigger: " + ", ".join(fired)) if ready else
                  f"No trigger yet — watch for a close above {max(prev.high, r.high):,.2f} or the 10 EMA ({r.ema10:,.2f})"]
        if missing:
            notes.append("Unavailable: " + ", ".join(missing))
        if gap_hit:
            notes.append("⚠ Recent gap-down on heavy volume")
        return Setup(
            score=score,
            entry=r.high if ready else max(r.high, r.ema10),
            entry_note="Buy-stop above today's high" if ready else "Not ready — alert above today's high / 10 EMA",
            stop_structure=stop, stop_note="Below the 10-day low",
            notes=notes, target_hint=r.sma50, target_note="Reclaim of 50 SMA",
            tags=["Ready" if ready else "Setting up"],
        )
