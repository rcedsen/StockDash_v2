"""Quality dip v2 — 'Pullback Swing Trading Screener v2.0'. Logic lives in core/qd2.py (shared with the
backtest); this file holds the plug-in metadata and turns engine rows into result cards."""
from core import qd2
from core.strategy import Strategy

P = qd2.PARAMS


def _pc(x, nd=0):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


class QualityDip(Strategy):
    key = "quality_dip"
    name = "Quality dip"
    description = "v2 · Russell 1000 profitable growers 10–30% off the high, below the 50 SMA, above a flat/rising 200 SMA."
    order = 50
    regime_sensitive = False
    universe = ["russell1000"]
    params = P
    hold = (P["hold_min"], P["hold_max"])
    target_r = 3.0
    atr_stop_mult = 2.0
    stop_rule = "Lowest low of the last 10 sessions − 0.1 × ATR(14); distance should be 1.5–2.5 ATR"
    full_report = True          # returns every constituent with a classification (see core/qd2.run_screen)
    needs = ("fundamentals",)
    rules = [
        "TECHNICAL · Close {off_high_min:.0%}–{off_high_max:.0%} below the 252-session high",
        "TECHNICAL · Close < 50 SMA and > 200 SMA",
        "TECHNICAL · 200 SMA ≥ its value {sma200_slope_days} sessions ago (flat or rising)",
        "FUNDAMENTAL · ROIC ≥ {roic_min:.0%} (computed NOPAT ÷ invested capital; proxies labeled and lower confidence)",
        "FUNDAMENTAL · Sales growth ≥ {sales_min:.0%} (TTM vs prior TTM when 8 quarters exist; basis shown)",
        "FUNDAMENTAL · Forward EPS growth ≥ {eps_min:.0%} — consensus FY+1 vs FY0 only (no GAAP/adjusted mixing)",
        "FUNDAMENTAL · Gross margin: stable/improving full credit · −{gm_stable_pts:g} to −{gm_fail_pts:g} pt reduced · worse fails",
        "FUNDAMENTAL · FCF yield ≥ {fcf_yield_min:.0%} OR FCF margin ≥ {fcf_margin_min:.0%} (TTM OCF − capex)",
        "FUNDAMENTAL · D/E ≤ {de_max:g} OR net debt/EBITDA ≤ {nd_ebitda_max:g}",
        "FUNDAMENTAL · Financials: FCF and leverage tests not applied (n/a, not 'missing')",
        "FUNDAMENTAL · PEG (fwd P/E ÷ fwd growth) ranks only; >{peg_max:g} is a warning; not computed on negative/near-zero EPS",
        "FUNDAMENTAL · Up to {max_unavailable} metric groups may be unavailable; every available minimum must pass",
        "EXCLUDE · Earnings within {earn_exclude} days · flag 8–{earn_elevated} days (elevated) and "
        "{earn_elevated}–{earn_event} days (event risk)",
        "EXCLUDE · Gap-down ≥ {gap_down:.0%} on ≥ {gap_vol_excl:g}× prior 20-day volume in {gap_lookback} sessions · "
        "{gap_vol_review:g}–{gap_vol_excl:g}× → manual review",
        "EXCLUDE · Confirmed guidance cut (not available on free data — never evaluated)",
        "WARN (−{warn_penalty:g} pts each) · EPS estimates cut ≥5% in 30 days · decelerating sales · operating margin down ≥{op_margin_drop_pts:g} pt",
    ]
    trigger = ("Four independent triggers, shown separately. A: RSI(2) < {rsi2_trigger} yesterday and close > "
               "yesterday's high. B: close reclaims the 10 EMA and closes up. C: a low within {sma200_touch_atr:g} ATR "
               "of the 200 SMA in the prior 5 sessions, then close > yesterday's high and > 200 SMA. D: green close "
               "above yesterday's high on ≥ {vol_trigger:g}× 20-day volume. Strength: A+D (15) › C+D (13) › A or C "
               "(11) › B (8) › D alone (5). A trigger alone is not a buy: risk and fundamentals must pass too. "
               "Entry = next session's open (the card estimates it with today's close).")
    score_parts = [
        "FUNDAMENTAL 40: ROIC 8 · sales 6 · EPS 6 · PEG 5 · FCF 5 · gross-margin trend 4 · leverage 3 · "
        "EPS revisions 3 — progressive (acceptable 40% · good 70% · excellent 100%), normalized over available metrics",
        "TECHNICAL 40: trigger strength 15 · pullback depth 7 (peak at 20% off high) · 200 SMA proximity in ATRs 5 · "
        "price structure (higher low) 5 · reversal candle 4 · bullish volume 4",
        "MARKET 20: RS vs SPY 10 · RS vs sector ETF 6 (20/60-day blend, Russell 1000 percentile) · regime 4",
        "Bands: ≥{class_highest} highest priority · {class_strong}–{class_highest} strong · {class_watch}–{class_strong} "
        "watchlist · below pass. Initial thresholds, not validated. A high score never overrides an exclusion.",
    ]
    best_when = "Broad pullbacks that drag quality growers down with the market"
    fails_when = "Company-specific re-ratings — estimate cuts and heavy-volume gaps are screened out or flagged"
    data = ("Daily OHLCV + fundamentals (Yahoo statements & consensus; Finnhub fallbacks), cached 7 days. "
            "Not point-in-time. Revenue-estimate and guidance revisions unavailable.")
    changed = "2026-11-05"

    def evaluate(self, d, symbol, ctx):            # not used: full_report strategies go through report()
        return None

    def report(self, frames, universe, ctx, settings, lookups, earnings_lookup, today):
        res = qd2.run_screen(frames, universe, ctx, lookups.get("fundamentals") or (lambda s: {}),
                             earnings_lookup, today, self.params)
        res["cards"] = {c: [self.card(r, i) for i, r in enumerate(
            [x for x in res["rows"] if x["classification"] == c], 1)]
            for c in ("Highest priority", "Strong", "Watchlist", "Pass", "Setting up", "Triggered — unacceptable risk")}
        return res

    @staticmethod
    def card(r: dict, rank: int) -> dict:
        rp, fe = r["risk"], r["fund_eval"]
        fs, ts, ms = r["fund_score"], r["tech_score"], r["mkt_score"]
        trig = [n for n, k in (("A RSI(2)", "trig_a"), ("B 10 EMA", "trig_b"), ("C 200 SMA", "trig_c"),
                               ("D volume", "trig_d")) if r[k]]
        notes = [f"Scores — fundamental {fs['score']:.1f}/40 · technical {ts['score']:.1f}/40 · market {ms['score']:.1f}/20 "
                 f"· coverage {r['coverage'] * 100:.0f}% · confidence {r['confidence']}",
                 "Triggers fired: " + (", ".join(trig) if trig else "none")]
        notes += [f"{g}: {fe['display'][g]}" + (" ✗" if fe["status"][g] == "fail" else "") for g in qd2.GROUPS]
        f = r["fund"]
        notes.append(f"PEG {f.get('peg'):.2f}" if f.get("peg") is not None else f"PEG n/a ({f.get('peg_note') or 'unavailable'})")
        notes.append(f"EPS revisions 30d: {_pc(f.get('eps_rev_30d'), 1)}")
        notes.append(f"RS vs SPY {r['rs_spy_pct'] or 0:.0f}th pct · vs sector "
                     + ("n/a" if r["rs_sector_pct"] is None else f"{r['rs_sector_pct']:.0f}th pct"))
        notes += ["⚠ " + w for w in r["warnings"]] + ["🔎 " + v for v in r["review"]]
        notes += [x for x in rp.get("flags", [])] + ["ℹ " + i for i in r["info"]]
        tags = [r["classification"]] + ([r["trig_label"]] if r["trig_label"] else []) + [f"conf. {r['confidence']}"]
        extra = [f"Stop distance <b>{rp['atr_stop_dist']:.2f} ATR</b> ({rp['atr_flag']}) · "
                 f"2R <b>{(rp.get('t2') or 0):,.2f}</b> (+{(rp.get('ret2_pct') or 0):.1f}%) · "
                 f"3R +{(rp.get('ret3_pct') or 0):.1f}%"]
        if rp.get("resistance") and rp["resistance"] < (rp.get("t3") or 0):
            extra.append(f"Resistance {rp['resistance_name']} <b>{rp['resistance']:,.2f}</b> "
                         f"→ {rp['avail_R']:.1f}R available")
        if r.get("earnings_flag"):
            extra.append("⚠ " + r["earnings_flag"])
        return dict(
            rank=rank, symbol=r["symbol"], name=r.get("name", ""), sector=r.get("sector") or "",
            sector_etf="", sector_state="—", sector_rs21=float("nan"), vs_sector21=float("nan"),
            score=round(r["total"], 1), price=r["close"], chg1=r.get("chg1") or 0.0, atr=r["atr"],
            entry=rp["entry"], entry_note="next session's open (estimated from today's close)",
            stop_structure=rp["stop"], stop_note="10-day low − 0.1 ATR",
            stop_atr=float("nan"), atr_mult=None,
            risk_struct_pct=rp.get("risk_pct") or float("nan"), risk_atr_pct=float("nan"),
            target_r=3.0, target=rp.get("t3") or float("nan"), target_hint=None, target_note="",
            hold_min=P["hold_min"], hold_max=P["hold_max"], time_stop=r["time_stop"], earn_in_hold=r["earn_in_hold"],
            notes=notes, tags=tags, rs_pct=r.get("rs_spy_pct"), earnings=r["earnings"],
            earnings_status="date" if r["earnings"] else r["earnings_status"], bar_date=r["bar_date"],
            extra=extra, classification=r["classification"],
        )
