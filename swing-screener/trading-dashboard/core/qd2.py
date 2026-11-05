"""Quality Dip v2 engine — the 'Pullback Swing Trading Screener v2.0' spec.

Pure functions (no Streamlit, no network) so every piece is unit-testable and the same code runs in the
live screener and in the backtest. All inputs are causal: indicators at session t use only data ≤ t.

Flow per ticker:  hard technical screen → gap check → fundamentals (survivors only) → earnings window →
                  fundamental qualification → triggers A–D → scores (40/40/20) → risk → classification.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd


# ───────────────────────── parameters
PARAMS = dict(
    # hard technical
    off_high_min=0.10, off_high_max=0.30, sma200_slope_days=20,
    # fundamental minimums
    roic_min=0.10, sales_min=0.10, eps_min=0.10,
    gm_stable_pts=1.0, gm_fail_pts=3.0,
    fcf_yield_min=0.01, fcf_margin_min=0.10,
    de_max=1.0, nd_ebitda_max=2.5,
    peg_max=2.0,                         # ranking + warning only, never a hard exclusion
    max_unavailable=2,                   # unavailable metric groups allowed
    # warnings / penalties
    eps_rev_material=-0.05, sales_decel_pts=5.0, op_margin_drop_pts=2.0, warn_penalty=2.0,
    # earnings (calendar days)
    earn_exclude=7, earn_elevated=14, earn_event=30,
    # gap-downs
    gap_down=0.05, gap_vol_excl=2.0, gap_vol_review=1.5, gap_lookback=15,
    # triggers
    rsi2_trigger=10, sma200_touch_atr=1.0, vol_trigger=1.2,
    # risk
    stop_lookback=10, stop_buffer_atr=0.1, atr_tight=1.5, atr_max=2.5, resistance_lookback=60,
    hold_min=10, hold_max=30,
    # classification
    class_highest=85, class_strong=75, class_watch=65,
    # variants (used by the backtest)
    use_sector_rs=True, require_volume=False,
)

# Version-1 thresholds, kept for "strict vs revised" comparisons
STRICT = dict(PARAMS, off_high_min=0.15, roic_min=0.15, sales_min=0.15, eps_min=0.15, gm_fail_pts=1.0,
              fcf_yield_min=0.02, fcf_margin_min=0.15, de_max=0.8, nd_ebitda_max=2.0, max_unavailable=1)

FINANCIAL_SECTORS = {"Financials"}
GROUPS = ["ROIC", "Sales growth", "Fwd EPS growth", "Gross margin", "FCF", "Leverage"]
TRIGGER_POINTS = {1: 15, 2: 13, 3: 11, 4: 8, 5: 5}
CLASS_ORDER = ["Highest priority", "Strong", "Watchlist", "Pass", "Setting up",
               "Triggered — unacceptable risk", "Excluded"]


def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def lin(x, x0, y0, x1, y1):
    """Linear interpolation clamped to [min(y0,y1), max(y0,y1)]."""
    if x is None:
        return None
    if x1 == x0:
        return y1
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    lo, hi = min(y0, y1), max(y0, y1)
    return max(lo, min(hi, y))


def progressive(x, acc, good, exc):
    """0 below acceptable · 0.4 at acceptable · 0.7 at good · 1.0 at excellent (linear between)."""
    if x is None:
        return None
    if x < acc:
        return 0.0
    if x < good:
        return lin(x, acc, 0.4, good, 0.7)
    if x < exc:
        return lin(x, good, 0.7, exc, 1.0)
    return 1.0


def bday_add(d: dt.date, n: int) -> dt.date:
    return pd.Timestamp(np.busday_offset(np.datetime64(d), n, roll="forward")).date()


def bday_before(d: dt.date) -> dt.date:
    return pd.Timestamp(np.busday_offset(np.datetime64(d), -1, roll="backward")).date()


# ───────────────────────── technical
def technicals(d: pd.DataFrame, p=PARAMS) -> dict:
    r = d.iloc[-1]
    hi = _f(r.hi252)
    close = float(r.close)
    s200, s200_ago = _f(r.sma200), _f(r.sma200_20ago)
    return dict(close=close, open=float(r.open), high=float(r.high), low=float(r.low),
                hi252=hi, off_high=(1 - close / hi) if hi else None,
                sma50=_f(r.sma50), sma200=s200, sma200_20ago=s200_ago,
                sma200_slope=((s200 / s200_ago - 1) if (s200 and s200_ago) else None),
                ema10=_f(r.ema10), atr=_f(r.atr), rsi2=_f(r.rsi2), vol20=_f(r.vol20),
                vol=float(r.vol_proj if "vol_proj" in d else r.volume),
                ret20=_f(r.ret20), ret60=_f(r.ret60), chg1=_f(r.chg1))


def hard_technical(t: dict, p=PARAMS) -> list[str]:
    """Returns failure reasons (empty = pass)."""
    fails = []
    if None in (t["hi252"], t["sma50"], t["sma200"], t["sma200_20ago"], t["atr"]):
        return ["insufficient price history for 52-week high / 200 SMA"]
    off = t["off_high"]
    if not (p["off_high_min"] <= off <= p["off_high_max"]):
        fails.append(f"{off * 100:.1f}% below 52-week high (needs {p['off_high_min'] * 100:.0f}–{p['off_high_max'] * 100:.0f}%)")
    if not t["close"] < t["sma50"]:
        fails.append("close not below 50 SMA")
    if not t["close"] > t["sma200"]:
        fails.append("close not above 200 SMA")
    if not t["sma200"] >= t["sma200_20ago"]:
        fails.append(f"200 SMA falling ({t['sma200_slope'] * 100:+.2f}% vs {p['sma200_slope_days']} sessions ago)")
    return fails


def gap_check(d: pd.DataFrame, p=PARAMS) -> tuple[str | None, str]:
    """('exclude' | 'review' | None, detail). Gap = open vs prior close; volume vs prior 20-day average."""
    n = p["gap_lookback"]
    w = d.iloc[-n:]
    prev_close = d.close.shift(1).loc[w.index]
    gap = w.open / prev_close - 1
    vr = w.volume / w.vol20
    worst, detail = None, ""
    for day in w.index:
        g, v = _f(gap.loc[day]), _f(vr.loc[day])
        if g is None or v is None or g > -p["gap_down"]:
            continue
        lvl = "exclude" if v >= p["gap_vol_excl"] else ("review" if v >= p["gap_vol_review"] else None)
        if lvl and (worst is None or lvl == "exclude"):
            worst = lvl
            detail = f"{g * 100:.1f}% gap-down on {v:.1f}× 20-day volume ({day:%b %d})"
    return worst, detail


def triggers(d: pd.DataFrame, p=PARAMS) -> dict:
    r, prev = d.iloc[-1], d.iloc[-2]
    vol = float(r.vol_proj if "vol_proj" in d else r.volume)
    v20 = _f(r.vol20)
    a = bool(_f(prev.rsi2) is not None and prev.rsi2 < p["rsi2_trigger"] and r.close > prev.high)
    b = bool(prev.close <= prev.ema10 and r.close > r.ema10 and r.close > prev.close)
    near = ((d.low.iloc[-6:-1] - d.sma200.iloc[-6:-1]).abs() <= p["sma200_touch_atr"] * r.atr).any()
    c = bool(near and r.close > prev.high and r.close > r.sma200)
    dd = bool(r.close > prev.high and r.close > r.open and v20 and vol >= p["vol_trigger"] * v20)
    if p.get("require_volume") and not dd:
        rank, label = None, ""
    elif a and dd:
        rank, label = 1, "RSI(2) reversal + volume"
    elif c and dd:
        rank, label = 2, "200 SMA reversal + volume"
    elif a or c:
        rank, label = 3, "RSI(2) reversal" if a else "200 SMA reversal"
    elif b:
        rank, label = 4, "10 EMA reclaim"
    elif dd:
        rank, label = 5, "Bullish volume reversal"
    else:
        rank, label = None, ""
    return dict(trig_a=a, trig_b=b, trig_c=c, trig_d=dd, trig_rank=rank, trig_label=label,
                trig_points=TRIGGER_POINTS.get(rank, 0))


# ───────────────────────── fundamentals
def fundamental_eval(f: dict | None, sector: str | None, p=PARAMS) -> dict:
    """Per-group status: pass / fail / unavailable / n/a, plus warnings and exclusion reasons."""
    f = f or {}
    fin = sector in FINANCIAL_SECTORS
    st, disp = {}, {}

    def put(g, ok, text):
        st[g] = "unavailable" if ok is None else ("pass" if ok else "fail")
        disp[g] = text

    roic = _f(f.get("roic"))
    put("ROIC", None if roic is None else roic >= p["roic_min"],
        "n/a" if roic is None else f"{roic * 100:.1f}% ({f.get('roic_src')})")
    sg = _f(f.get("sales_growth"))
    put("Sales growth", None if sg is None else sg >= p["sales_min"],
        "n/a" if sg is None else f"{sg * 100:.1f}% ({f.get('sales_growth_src')})")
    eg = _f(f.get("fwd_eps_growth"))
    put("Fwd EPS growth", None if eg is None else eg >= p["eps_min"],
        "n/a (not meaningful or mixed basis)" if eg is None else f"{eg * 100:.1f}% ({f.get('eps_basis')})")
    gmd = _f(f.get("gm_delta_pts"))
    put("Gross margin", None if gmd is None else gmd >= -p["gm_fail_pts"],
        "n/a" if gmd is None else f"{gmd:+.1f} pt ({f.get('gm_src')})")
    if fin:
        st["FCF"], disp["FCF"] = "n/a", "not applied to financials"
        st["Leverage"], disp["Leverage"] = "n/a", "not applied to financials"
    else:
        fy, fm = _f(f.get("fcf_yield")), _f(f.get("fcf_margin"))
        vals = [v for v in ((fy, p["fcf_yield_min"]), (fm, p["fcf_margin_min"])) if v[0] is not None]
        put("FCF", None if not vals else any(v >= m for v, m in vals),
            "n/a" if not vals else f"yield {'n/a' if fy is None else f'{fy * 100:.1f}%'} · margin "
                                    f"{'n/a' if fm is None else f'{fm * 100:.1f}%'}")
        de, nde = _f(f.get("de")), _f(f.get("nd_ebitda"))
        lv = [v for v in ((de, p["de_max"]), (nde, p["nd_ebitda_max"])) if v[0] is not None]
        put("Leverage", None if not lv else any(v <= m for v, m in lv),
            "n/a" if not lv else f"D/E {'n/a' if de is None else f'{de:.2f}'} · ND/EBITDA "
                                  f"{'n/a' if nde is None else f'{nde:.1f}'}")

    fails = [f"{g} {disp[g]}" for g in GROUPS if st[g] == "fail"]
    unavailable = [g for g in GROUPS if st[g] == "unavailable"]
    applicable = [g for g in GROUPS if st[g] != "n/a"]
    excl = list(fails)
    if len(unavailable) > p["max_unavailable"]:
        excl.append(f"{len(unavailable)} metric groups unavailable ({', '.join(unavailable)}); max {p['max_unavailable']}")
    if f.get("guidance_cut") is True:
        excl.append("confirmed guidance reduction")

    warns = []
    er = _f(f.get("eps_rev_30d"))
    if er is not None and er <= p["eps_rev_material"]:
        warns.append(f"EPS estimates cut {er * 100:.1f}% in 30 days")
    slq = _f(f.get("sales_growth_latest_q"))
    if slq is not None and sg is not None and slq < sg - p["sales_decel_pts"] / 100:
        warns.append(f"sales growth decelerating (latest qtr {slq * 100:.1f}% vs {sg * 100:.1f}%)")
    om = _f(f.get("op_margin_delta_pts"))
    if om is not None and om <= -p["op_margin_drop_pts"]:
        warns.append(f"operating margin down {om:.1f} pt Y/Y")
    if gmd is not None and -p["gm_fail_pts"] <= gmd < -p["gm_stable_pts"]:
        warns.append(f"gross margin down {gmd:.1f} pt (reduced credit)")
    peg = _f(f.get("peg"))
    if peg is not None and f.get("peg_meaningful") and peg > p["peg_max"]:
        warns.append(f"PEG {peg:.2f} above {p['peg_max']:g} (ranking penalty only)")
    info = []
    if f.get("roic_is_proxy"):
        info.append(f"ROIC is a proxy: {f.get('roic_src')}")
    for k in ("sales_growth_src", "gm_src"):
        if f.get(k) and "not TTM" in str(f.get(k)):
            info.append(f"{'Sales growth' if k.startswith('sales') else 'Gross margin'} basis: {f[k]}")
    if f.get("fcf_basis") and "not verified" in str(f.get("fcf_basis")):
        info.append(f"FCF basis: {f['fcf_basis']}")
    info.append("Revenue-estimate and guidance revisions unavailable on free data — not evaluated")
    info += list(f.get("notes") or [])
    return dict(status=st, display=disp, fails=fails, unavailable=unavailable, applicable=applicable,
                exclusions=excl, warnings=warns, info=info, financial=fin,
                completeness=(len(applicable) - len(unavailable)) / len(applicable) if applicable else 0.0)


def fundamental_score(f: dict | None, ev: dict, p=PARAMS) -> dict:
    f = f or {}
    st = ev["status"]
    comp = {}   # name: (fraction or None if unavailable, max points); n/a groups are skipped
    comp["ROIC"] = (progressive(_f(f.get("roic")), 0.10, 0.15, 0.25), 8)
    comp["Sales growth"] = (progressive(_f(f.get("sales_growth")), 0.10, 0.20, 0.30), 6)
    comp["Fwd EPS growth"] = (progressive(_f(f.get("fwd_eps_growth")), 0.10, 0.20, 0.30), 6)
    peg = _f(f.get("peg")) if f.get("peg_meaningful") else None
    comp["PEG"] = (None if peg is None else (1.0 if peg <= 0.8 else (lin(peg, 0.8, 1.0, 2.0, 0.1) if peg <= 2.0 else 0.0)), 5)
    if st.get("FCF") != "n/a":
        fy = progressive(_f(f.get("fcf_yield")), 0.01, 0.03, 0.05)
        fm = progressive(_f(f.get("fcf_margin")), 0.10, 0.20, 0.30)
        vals = [v for v in (fy, fm) if v is not None]
        comp["FCF"] = (max(vals) if vals else None, 5)
    gmd = _f(f.get("gm_delta_pts"))
    comp["Gross margin trend"] = (None if gmd is None else 1.0 if gmd >= -p["gm_stable_pts"]
                                  else lin(gmd, -p["gm_stable_pts"], 0.75, -p["gm_fail_pts"], 0.25)
                                  if gmd >= -p["gm_fail_pts"] else 0.0, 4)
    if st.get("Leverage") != "n/a":
        de, nde = _f(f.get("de")), _f(f.get("nd_ebitda"))
        vals = [v for v in ((lin(de, 0.25, 1.0, 1.0, 0.3) if de is not None and de >= 0 else None),
                            (lin(nde, 0.0, 1.0, 2.5, 0.3) if nde is not None and nde <= 2.5 else (0.0 if nde is not None else None)))
                if v is not None]
        comp["Leverage"] = (max(vals) if vals else None, 3)
    er = _f(f.get("eps_rev_30d"))
    comp["Estimate revisions"] = (lin(er, -0.05, 0.0, 0.05, 1.0) if er is not None else None, 3)

    avail = {k: v for k, v in comp.items() if v[0] is not None}
    max_avail = sum(m for _, m in avail.values())
    max_appl = sum(m for _, m in comp.values())
    earned = sum(v * m for v, m in avail.values())
    raw = 40 * earned / max_avail if max_avail else 0.0
    penalty = p["warn_penalty"] * len([w for w in ev["warnings"] if "reduced credit" not in w and "PEG" not in w])
    return dict(score=max(0.0, raw - penalty), raw=raw, penalty=penalty,
                coverage=max_avail / max_appl if max_appl else 0.0,
                parts={k: (None if v is None else round(v * m, 2), m) for k, (v, m) in comp.items()})


def technical_score(d: pd.DataFrame, t: dict, tr: dict, p=PARAMS) -> dict:
    r, prev = d.iloc[-1], d.iloc[-2]
    atr = t["atr"]
    parts = {}
    parts["Trigger strength"] = (tr["trig_points"] / 15, 15)
    off = t["off_high"]
    parts["Pullback depth"] = (max(0.3, 1 - 0.7 * abs(off - 0.20) / 0.10) if off is not None and
                               p["off_high_min"] <= off <= p["off_high_max"] else 0.0, 7)
    low5 = float(d.low.iloc[-5:].min())
    low_prior = float(d.low.iloc[-15:-5].min())
    if low5 > low_prior + 0.25 * atr:
        struct = 1.0
    elif low5 >= low_prior - 0.25 * atr:
        struct = 0.5
    else:
        struct = 0.0
    parts["Price structure"] = (struct, 5)
    dist = (t["close"] - t["sma200"]) / atr
    prox = 0.5 if dist < 0.5 else 1.0 if dist <= 3 else max(0.2, 1 - 0.8 * (dist - 3) / 5)
    if struct < 1.0:
        prox = min(prox, 0.6)            # nearness alone never earns full points
    parts["200 SMA proximity"] = (prox, 5)
    rng = float(r.high - r.low)
    green = r.close > r.open
    if rng > 0:
        clv, body = (r.close - r.low) / rng, abs(r.close - r.open) / rng
        candle = 0.5 * clv + (0.3 + 0.2 * body if green else 0.0)
    else:
        candle = 0.0
    parts["Reversal candle"] = (float(min(1.0, candle)), 4)
    vr = t["vol"] / t["vol20"] if t["vol20"] else None
    bull = green and r.close > prev.close
    parts["Bullish volume"] = ((lin(vr, 1.0, 0.2, 1.5, 1.0) if vr >= 1.0 else 0.0) if (bull and vr) else 0.0, 4)
    score = sum(v * m for v, m in parts.values())
    return dict(score=score, coverage=1.0, atr_dist_200=dist, vol_ratio=vr,
                parts={k: (round(v * m, 2), m) for k, (v, m) in parts.items()})


def market_score(sym: str, ctx: dict, p=PARAMS) -> dict:
    comp = {"RS vs SPY": (_pct(ctx.get("rs_spy_pct", {}).get(sym)), 10),
            "Regime": ((ctx.get("regime_points") / 4) if ctx.get("regime_points") is not None else None, 4)}
    if p.get("use_sector_rs", True):
        comp["RS vs sector"] = (_pct(ctx.get("rs_sector_pct", {}).get(sym)), 6)
    avail = {k: v for k, v in comp.items() if v[0] is not None}
    max_avail = sum(m for _, m in avail.values())
    max_appl = sum(m for _, m in comp.values())
    score = 20 * sum(v * m for v, m in avail.values()) / max_avail if max_avail else 0.0
    return dict(score=score, coverage=max_avail / max_appl if max_appl else 0.0,
                parts={k: (None if v is None else round(v * m, 2), m) for k, (v, m) in comp.items()})


def _pct(x):
    x = _f(x)
    return None if x is None else x / 100


# ───────────────────────── risk & earnings
def risk_plan(d: pd.DataFrame, entry: float, p=PARAMS) -> dict:
    r = d.iloc[-1]
    atr = float(r.atr)
    stop = float(d.low.iloc[-p["stop_lookback"]:].min()) - p["stop_buffer_atr"] * atr
    risk = entry - stop
    out = dict(entry=entry, stop=stop, risk_ps=risk, atr_stop_dist=risk / atr if atr else None,
               risk_pct=100 * risk / entry if entry else None, flags=[])
    if risk <= 0:
        out.update(t2=None, t3=None, ret2_pct=None, ret3_pct=None, risk_ok=False, atr_flag="invalid stop")
        return out
    out["t2"], out["t3"] = entry + 2 * risk, entry + 3 * risk
    out["ret2_pct"], out["ret3_pct"] = 200 * risk / entry, 300 * risk / entry
    a = out["atr_stop_dist"]
    out["atr_flag"] = ("too tight (<{:g} ATR)".format(p["atr_tight"]) if a < p["atr_tight"] else
                       "preferred" if a <= p["atr_max"] else "unfavorable (>{:g} ATR)".format(p["atr_max"]))
    out["risk_ok"] = a <= p["atr_max"]
    lvls = {"50 SMA": _f(r.sma50), f"{p['resistance_lookback']}-day high": float(d.high.iloc[-p["resistance_lookback"]:].max()),
            "52-week high": _f(r.hi252)}
    above = {k: v for k, v in lvls.items() if v is not None and v > entry * 1.005}
    if above:
        name, lvl = min(above.items(), key=lambda kv: kv[1])
        out["resistance"], out["resistance_name"] = lvl, name
        out["avail_R"] = (lvl - entry) / risk
        out["avail_pct"] = 100 * (lvl - entry) / entry
        if lvl < out["t3"]:
            out["flags"].append(f"3R ({out['t3']:,.2f}) is beyond {name} resistance at {lvl:,.2f} — "
                                f"available upside {out['avail_pct']:.1f}% ({out['avail_R']:.1f}R)")
    else:
        out["resistance"], out["resistance_name"], out["avail_R"], out["avail_pct"] = None, "", None, None
    out["flags"].append("stops can fill below the stop price on a gap")
    return out


def earnings_status(earn: dt.date | None, known: bool, today: dt.date, p=PARAMS) -> tuple[str, int | None]:
    """('exclude' | 'elevated' | 'event' | 'clear' | 'unknown', days)."""
    if earn is None:
        return ("clear" if known else "unknown"), None
    days = (earn - today).days
    if days < 0:
        return "clear", days
    if days <= p["earn_exclude"]:
        return "exclude", days
    if days <= p["earn_elevated"]:
        return "elevated", days
    if days <= p["earn_event"]:
        return "event", days
    return "clear", days


def classify(total: float, excluded: bool, triggered: bool, risk_ok: bool, p=PARAMS) -> str:
    if excluded:
        return "Excluded"
    if not triggered:
        return "Setting up"
    if not risk_ok:
        return "Triggered — unacceptable risk"
    if total >= p["class_highest"]:
        return "Highest priority"
    if total >= p["class_strong"]:
        return "Strong"
    if total >= p["class_watch"]:
        return "Watchlist"
    return "Pass"


def confidence(fund_cov, tech_cov, mkt_cov, proxy: bool) -> tuple[float, str]:
    cov = (40 * fund_cov + 40 * tech_cov + 20 * mkt_cov) / 100
    lvl = "High" if cov >= 0.9 else "Medium" if cov >= 0.7 else "Low"
    if proxy:
        lvl = {"High": "Medium", "Medium": "Low", "Low": "Low"}[lvl]
    return cov, lvl


# ───────────────────────── one ticker
def evaluate(sym: str, d: pd.DataFrame, sector: str | None, f: dict | None, ctx: dict,
             earn: tuple[dt.date | None, bool] | None, today: dt.date, p=PARAMS, entry: float | None = None,
             tech: dict | None = None) -> dict:
    """Full evaluation of a ticker that passed the hard technical screen and the gap exclusion."""
    t = tech or technicals(d, p)
    gap_lvl, gap_txt = gap_check(d, p)
    ev = fundamental_eval(f, sector, p)
    tr = triggers(d, p)
    fs, ts, ms = fundamental_score(f, ev, p), technical_score(d, t, tr, p), market_score(sym, ctx, p)
    total = fs["score"] + ts["score"] + ms["score"]
    cov, conf = confidence(fs["coverage"], ts["coverage"], ms["coverage"], bool((f or {}).get("roic_is_proxy")))
    e_date, e_known = earn if earn else (None, False)
    e_stat, e_days = earnings_status(e_date, e_known, today, p)
    rp = risk_plan(d, entry if entry is not None else t["close"], p)

    excl = list(ev["exclusions"])
    if gap_lvl == "exclude":
        excl.append(gap_txt)
    if e_stat == "exclude":
        excl.append(f"earnings in {e_days} days ({e_date:%b %d})")
    review = []
    if gap_lvl == "review":
        review.append("manual review: " + gap_txt)
    if rp["atr_flag"].startswith("unfavorable") or rp["atr_flag"] == "invalid stop":
        review.append(f"stop distance {rp['atr_flag']}")
    if e_stat == "unknown":
        review.append("earnings date unknown — check before entry")
    flags_earn = {"elevated": f"earnings in {e_days} days — elevated risk",
                  "event": f"earnings in {e_days} days — event risk inside the hold"}.get(e_stat)

    bar_date = d.index[-1].date()
    time_stop = bday_add(bar_date, p["hold_max"])
    earn_in_hold = bool(e_date and e_days is not None and e_days >= 0 and e_date <= time_stop)
    if earn_in_hold:
        time_stop = min(time_stop, bday_before(e_date))
    cls = classify(total, bool(excl), tr["trig_rank"] is not None, rp["risk_ok"], p)

    reasons = []
    if cls == "Excluded":
        reasons = excl
    elif cls == "Setting up":
        reasons = ["passes screens; no entry trigger yet"]
    elif cls.startswith("Triggered"):
        reasons = [f"trigger fired but stop distance {rp['atr_stop_dist']:.2f} ATR ({rp['atr_flag']})"]
    else:
        reasons = [f"{tr['trig_label']}; score {total:.0f}"]
    return dict(
        symbol=sym, sector=sector, bar_date=bar_date, **{k: t[k] for k in ("close", "off_high", "sma50", "sma200",
        "sma200_slope", "rsi2", "atr", "ema10", "chg1")},
        fund=f or {}, fund_eval=ev, fund_score=fs, tech_score=ts, mkt_score=ms, total=total,
        coverage=cov, confidence=conf, data_completeness=ev["completeness"], **tr, gap=gap_lvl, gap_detail=gap_txt,
        earnings=e_date, earnings_days=e_days, earnings_status=e_stat, earnings_flag=flags_earn,
        risk=rp, time_stop=time_stop, earn_in_hold=earn_in_hold, classification=cls,
        exclusions=excl, review=review, warnings=ev["warnings"], info=ev["info"], reasons=reasons,
        rs_spy_pct=ctx.get("rs_spy_pct", {}).get(sym), rs_sector_pct=ctx.get("rs_sector_pct", {}).get(sym),
        rs_spy_excess=ctx.get("rs_spy_excess", {}).get(sym), rs_sector_excess=ctx.get("rs_sector_excess", {}).get(sym),
    )


# ───────────────────────── universe run
def run_screen(frames: dict, universe: pd.DataFrame, ctx: dict, fund_lookup, earn_lookup,
               today: dt.date, p=PARAMS) -> dict:
    """fund_lookup(symbols) -> {sym: metrics}; earn_lookup(symbols) -> (dates, known, source).
    Returns dict(rows=[...one per constituent...], stats=...)."""
    sector_of = dict(zip(universe.symbol, universe.sector))
    name_of = dict(zip(universe.symbol, universe.name))
    rows, survivors, techs = [], [], {}
    for s in universe.symbol:
        d = frames.get(s)
        base = dict(symbol=s, name=name_of.get(s, ""), sector=sector_of.get(s))
        if d is None or len(d) < 30:
            rows.append(dict(base, classification="Excluded", exclusions=["no price data"], stage="data"))
            continue
        t = technicals(d, p)
        fails = hard_technical(t, p)
        if fails:
            rows.append(dict(base, **{k: t[k] for k in ("close", "off_high", "sma50", "sma200", "sma200_slope", "rsi2", "atr")},
                             bar_date=d.index[-1].date(), classification="Excluded", exclusions=fails, stage="technical"))
            continue
        g, gtxt = gap_check(d, p)
        if g == "exclude":
            rows.append(dict(base, **{k: t[k] for k in ("close", "off_high", "sma50", "sma200", "sma200_slope", "rsi2", "atr")},
                             bar_date=d.index[-1].date(), classification="Excluded", exclusions=[gtxt],
                             gap="exclude", gap_detail=gtxt, stage="gap"))
            continue
        survivors.append(s)
        techs[s] = t
    fund = fund_lookup(survivors) if survivors else {}
    dates, known, esrc = earn_lookup(survivors) if survivors else ({}, set(), "")
    for s in survivors:
        r = evaluate(s, frames[s], sector_of.get(s), fund.get(s), ctx, (dates.get(s), s in known), today, p,
                     tech=techs[s])
        r.update(name=name_of.get(s, ""), stage="full")
        rows.append(r)
    order = {c: i for i, c in enumerate(CLASS_ORDER)}
    rows.sort(key=lambda r: (order.get(r["classification"], 99), -(r.get("total") or 0), r["symbol"]))
    stats = dict(universe=len(universe), priced=sum(1 for r in rows if r.get("stage") != "data"),
                 technical_pass=len(survivors) + sum(1 for r in rows if r.get("stage") == "gap"),
                 evaluated=len(survivors), fundamentals_returned=sum(1 for s in survivors if fund.get(s)),
                 earnings_source=esrc,
                 counts={c: sum(1 for r in rows if r["classification"] == c) for c in CLASS_ORDER})
    return dict(rows=rows, stats=stats)


# ───────────────────────── flat CSV
CSV_COLUMNS = [
    "ticker", "company", "sector", "close", "pct_below_52w_high", "sma50", "sma200", "sma200_slope_20d_pct", "rsi2",
    "atr14", "roic_pct", "roic_basis", "sales_growth_pct", "sales_growth_basis", "fwd_eps_growth_pct", "eps_basis",
    "peg", "peg_note", "fcf_yield_pct", "fcf_margin_pct", "fcf_basis", "gross_margin_trend_pts", "gross_margin_basis",
    "debt_equity", "net_debt_ebitda", "eps_revision_30d_pct", "rs_vs_spy_pct", "rs_vs_sector_pct",
    "fundamental_score", "technical_score", "market_score", "total_score", "fundamental_coverage_pct",
    "score_coverage_pct", "confidence", "data_completeness_pct",
    "trigger_a_rsi2", "trigger_b_10ema", "trigger_c_200sma", "trigger_d_volume", "trigger_best",
    "earnings_date", "earnings_days", "earnings_flag", "gap_flag", "fundamental_warnings",
    "entry_est", "structural_stop", "risk_per_share", "risk_pct", "atr_stop_distance", "atr_stop_flag",
    "target_2r", "target_3r", "return_2r_pct", "return_3r_pct", "nearest_resistance", "available_r",
    "time_stop", "classification", "reasons", "review", "stage", "bar_date",
]


def _r(x, n=2, mul=1.0):
    x = _f(x)
    return None if x is None else round(x * mul, n)


def to_csv_rows(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        f, rp = r.get("fund") or {}, r.get("risk") or {}
        fs, ts, ms = r.get("fund_score") or {}, r.get("tech_score") or {}, r.get("mkt_score") or {}
        full = r.get("stage") == "full"
        out.append({
            "ticker": r["symbol"], "company": r.get("name"), "sector": r.get("sector"), "close": _r(r.get("close")),
            "pct_below_52w_high": _r(r.get("off_high"), 1, 100), "sma50": _r(r.get("sma50")), "sma200": _r(r.get("sma200")),
            "sma200_slope_20d_pct": _r(r.get("sma200_slope"), 2, 100), "rsi2": _r(r.get("rsi2"), 1), "atr14": _r(r.get("atr")),
            "roic_pct": _r(f.get("roic"), 1, 100), "roic_basis": f.get("roic_src"),
            "sales_growth_pct": _r(f.get("sales_growth"), 1, 100), "sales_growth_basis": f.get("sales_growth_src"),
            "fwd_eps_growth_pct": _r(f.get("fwd_eps_growth"), 1, 100), "eps_basis": f.get("eps_basis"),
            "peg": _r(f.get("peg")), "peg_note": f.get("peg_note"),
            "fcf_yield_pct": _r(f.get("fcf_yield"), 2, 100), "fcf_margin_pct": _r(f.get("fcf_margin"), 1, 100),
            "fcf_basis": f.get("fcf_basis"), "gross_margin_trend_pts": _r(f.get("gm_delta_pts"), 2),
            "gross_margin_basis": f.get("gm_src"), "debt_equity": _r(f.get("de")), "net_debt_ebitda": _r(f.get("nd_ebitda")),
            "eps_revision_30d_pct": _r(f.get("eps_rev_30d"), 1, 100),
            "rs_vs_spy_pct": _r(r.get("rs_spy_pct"), 0), "rs_vs_sector_pct": _r(r.get("rs_sector_pct"), 0),
            "fundamental_score": _r(fs.get("score"), 1), "technical_score": _r(ts.get("score"), 1),
            "market_score": _r(ms.get("score"), 1), "total_score": _r(r.get("total"), 1),
            "fundamental_coverage_pct": _r(fs.get("coverage"), 0, 100), "score_coverage_pct": _r(r.get("coverage"), 0, 100),
            "confidence": r.get("confidence"), "data_completeness_pct": _r(r.get("data_completeness"), 0, 100),
            "trigger_a_rsi2": r.get("trig_a"), "trigger_b_10ema": r.get("trig_b"), "trigger_c_200sma": r.get("trig_c"),
            "trigger_d_volume": r.get("trig_d"), "trigger_best": r.get("trig_label"),
            "earnings_date": r.get("earnings"), "earnings_days": r.get("earnings_days"), "earnings_flag": r.get("earnings_flag"),
            "gap_flag": r.get("gap_detail") or None, "fundamental_warnings": " | ".join(r.get("warnings") or []) or None,
            "entry_est": _r(rp.get("entry")), "structural_stop": _r(rp.get("stop")), "risk_per_share": _r(rp.get("risk_ps")),
            "risk_pct": _r(rp.get("risk_pct"), 2), "atr_stop_distance": _r(rp.get("atr_stop_dist")),
            "atr_stop_flag": rp.get("atr_flag"), "target_2r": _r(rp.get("t2")), "target_3r": _r(rp.get("t3")),
            "return_2r_pct": _r(rp.get("ret2_pct"), 1), "return_3r_pct": _r(rp.get("ret3_pct"), 1),
            "nearest_resistance": _r(rp.get("resistance")), "available_r": _r(rp.get("avail_R"), 1),
            "time_stop": r.get("time_stop"), "classification": r["classification"],
            "reasons": " | ".join(r.get("reasons") or r.get("exclusions") or []),
            "review": " | ".join(r.get("review") or []) or None,
            "stage": {"data": "no data", "technical": "failed technical screen (fundamentals not fetched)",
                      "gap": "gap exclusion (fundamentals not fetched)", "full": "fully evaluated"}.get(r.get("stage")),
            "bar_date": r.get("bar_date"),
        })
        if not full:
            out[-1]["reasons"] = " | ".join(r.get("exclusions") or [])
    return pd.DataFrame(out, columns=CSV_COLUMNS)


def _reason_key(reason: str) -> str:
    """Group exclusion reasons: drop the numbers and parenthetical detail."""
    import re
    if "below 52-week high" in reason:
        return "not 10–30% below 52-week high"
    if reason.startswith("earnings in"):
        return "earnings within 7 days"
    if "gap-down" in reason:
        return "heavy-volume gap-down"
    if "metric groups unavailable" in reason:
        return "too many unavailable fundamentals"
    base = reason.split(" (")[0]
    for g in GROUPS:
        if base.startswith(g):
            return f"failed {g}"
    return re.sub(r"[-+]?\d[\d.,]*%?", "#", base).strip()


def report_markdown(res: dict, ctx: dict, top: int = 25) -> str:
    """Readable ranked + exclusion report."""
    rows, st = res["rows"], res["stats"]
    lines = [f"# Quality Dip v2 — screening report ({rows[0]['bar_date'] if rows and rows[0].get('bar_date') else ''})", "",
             f"Universe {st['universe']} · priced {st['priced']} · passed technical {st['technical_pass']} · "
             f"fully evaluated {st['evaluated']} · fundamentals returned {st['fundamentals_returned']} · "
             f"market regime: {ctx.get('regime_label', 'n/a')}", ""]
    for c in CLASS_ORDER:
        lines.append(f"- {c}: {st['counts'].get(c, 0)}")
    for title, cls in [("1. Highest-priority candidates", ["Highest priority"]), ("2. Strong candidates", ["Strong"]),
                       ("Watchlist (triggered, 65–74)", ["Watchlist"]), ("Pass (triggered, below 65)", ["Pass"]),
                       ("3. Setting up (no entry trigger)", ["Setting up"]),
                       ("4. Triggered with unacceptable risk", ["Triggered — unacceptable risk"])]:
        sel = [r for r in rows if r["classification"] in cls][:top]
        lines += ["", f"## {title}", ""]
        if not sel:
            lines.append("None.")
            continue
        lines.append("| Ticker | Score | F/T/M | Trigger | Entry | Stop | ATR dist | 2R | 3R | Earnings | Notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            rp = r["risk"]
            notes = "; ".join((r.get("warnings") or []) + (r.get("review") or []) + rp.get("flags", [])[:-1])
            lines.append(f"| {r['symbol']} | {r['total']:.0f} ({r['confidence']}) | {r['fund_score']['score']:.0f}/"
                         f"{r['tech_score']['score']:.0f}/{r['mkt_score']['score']:.0f} | {r['trig_label'] or '—'} | "
                         f"{rp['entry']:,.2f} | {rp['stop']:,.2f} | {rp['atr_stop_dist']:.2f} | "
                         f"{(rp.get('t2') or 0):,.2f} | {(rp.get('t3') or 0):,.2f} | "
                         f"{r['earnings'] or '—'} | {notes or '—'} |")
    ex = [r for r in rows if r["classification"] == "Excluded"]
    lines += ["", f"## 5. Excluded ({len(ex)})", ""]
    by = {}
    for r in ex:
        for reason in r.get("exclusions") or ["?"]:
            key = _reason_key(reason)
            by[key] = by.get(key, 0) + 1
    lines.append("Exclusion reasons (a ticker can have several):")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        lines.append(f"- {k}: {v}")
    lines += ["", "Full per-ticker reasons are in the CSV.",
              "", "_Score bands are initial ranking thresholds, not statistically validated predictions._"]
    return "\n".join(lines)
