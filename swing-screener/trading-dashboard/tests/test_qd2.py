"""Quality dip v2 engine: scoring, missing data, earnings, gaps, triggers, stops/targets, normalization,
sizing, and a small known sample (synthetic universe)."""
import datetime as dt

import pandas as pd
import pytest

from core import fundamentals, indicators, qd2, sizing
from tests.synthetic import GOOD_FUND, fake_fundamentals, market
from tests.test_pipeline import END, _setup, no_earnings

P = qd2.PARAMS


def raw(sym="QDIP"):
    return market(END)[sym][["open", "high", "low", "close", "volume"]].copy()


def ind(df):
    df = df.copy()
    df["vol_proj"] = df.volume
    return indicators.add_all(df)


def set_bar(df, i, o, h, l, c, v=None):
    idx = df.index[i]
    df.loc[idx, ["open", "high", "low", "close"]] = [o, h, l, c]
    if v is not None:
        df.loc[idx, "volume"] = v


# ───────────────────────── scoring functions
def test_progressive_and_lin():
    assert qd2.progressive(0.05, .10, .15, .25) == 0.0
    assert qd2.progressive(0.10, .10, .15, .25) == pytest.approx(0.4)
    assert qd2.progressive(0.15, .10, .15, .25) == pytest.approx(0.7)
    assert qd2.progressive(0.20, .10, .15, .25) == pytest.approx(0.85)
    assert qd2.progressive(0.40, .10, .15, .25) == 1.0
    assert qd2.progressive(None, .10, .15, .25) is None
    assert qd2.lin(5, 0, 0, 10, 1) == pytest.approx(0.5) and qd2.lin(20, 0, 0, 10, 1) == 1


def _fs(f, sector="Information Technology", p=P):
    ev = qd2.fundamental_eval(f, sector, p)
    return ev, qd2.fundamental_score(f, ev, p)


def test_fundamental_score_full_and_peg_curve():
    ev, fs = _fs(dict(GOOD_FUND))
    assert not ev["exclusions"] and fs["coverage"] == 1.0 and 0 < fs["score"] <= 40
    _, cheap = _fs(dict(GOOD_FUND, peg=0.7))
    _, dear = _fs(dict(GOOD_FUND, peg=1.9))
    assert cheap["parts"]["PEG"][0] == 5 and dear["parts"]["PEG"][0] < 1


def test_normalization_over_available_metrics():
    f = dict(GOOD_FUND)
    _, full = _fs(f)
    g = dict(f, roic=None, roic_src=None, sales_growth=None)
    ev, part = _fs(g)
    assert ev["unavailable"] == ["ROIC", "Sales growth"] and not ev["exclusions"]       # 2 allowed
    assert part["coverage"] == pytest.approx((40 - 14) / 40)
    earned = sum(v for k, (v, m) in part["parts"].items() if v is not None)
    assert part["score"] == pytest.approx(40 * earned / 26)
    ev3, _ = _fs(dict(g, fcf_yield=None, fcf_margin=None))
    assert any("unavailable" in e for e in ev3["exclusions"])                            # 3 → excluded


def test_failed_vs_unavailable_and_financials():
    ev, _ = _fs(dict(GOOD_FUND, sales_growth=0.05))
    assert ev["status"]["Sales growth"] == "fail" and ev["exclusions"]
    ev, fs = _fs(dict(GOOD_FUND, fcf_yield=None, fcf_margin=None, de=None, nd_ebitda=None), sector="Financials")
    assert ev["status"]["FCF"] == ev["status"]["Leverage"] == "n/a" and not ev["unavailable"] and not ev["exclusions"]
    assert "FCF" not in fs["parts"] and fs["coverage"] == 1.0


def test_gross_margin_bands():
    ev, fs = _fs(dict(GOOD_FUND, gm_delta_pts=-0.5))
    assert ev["status"]["Gross margin"] == "pass" and fs["parts"]["Gross margin trend"][0] == 4
    ev, fs = _fs(dict(GOOD_FUND, gm_delta_pts=-2.0))
    assert ev["status"]["Gross margin"] == "pass" and 0 < fs["parts"]["Gross margin trend"][0] < 4
    assert any("reduced credit" in w for w in ev["warnings"])
    ev, _ = _fs(dict(GOOD_FUND, gm_delta_pts=-3.5))
    assert ev["status"]["Gross margin"] == "fail"


def test_warnings_penalize_but_peg_never_excludes():
    ev, fs = _fs(dict(GOOD_FUND, peg=2.6))
    assert not ev["exclusions"] and any("PEG" in w for w in ev["warnings"]) and fs["penalty"] == 0
    ev, fs = _fs(dict(GOOD_FUND, eps_rev_30d=-0.08, sales_growth_latest_q=0.12, op_margin_delta_pts=-3))
    assert len(ev["warnings"]) == 3 and fs["penalty"] == 3 * P["warn_penalty"]
    ev, _ = _fs(dict(GOOD_FUND, guidance_cut=True))
    assert "confirmed guidance reduction" in ev["exclusions"]


def test_proxy_lowers_confidence():
    assert qd2.confidence(1, 1, 1, False)[1] == "High"
    assert qd2.confidence(1, 1, 1, True)[1] == "Medium"
    assert qd2.confidence(0.5, 1, 1, False)[1] == "Medium"


# ───────────────────────── fundamentals.compute (no network)
def _stmt(rows: dict, n=8):
    cols = pd.date_range(end="2026-09-30", periods=n, freq="QE")[::-1]
    return pd.DataFrame({c: {k: v[i] for k, v in rows.items()} for i, c in enumerate(cols)})


def test_compute_metrics_consistent_basis():
    q_inc = _stmt({"Total Revenue": [130, 125, 120, 115, 110, 105, 100, 95],
                   "Gross Profit": [78, 75, 72, 69, 64, 61, 58, 55],
                   "Operating Income": [30, 29, 28, 27, 22, 21, 20, 19],
                   "EBITDA": [36, 35, 34, 33, 28, 27, 26, 25], "Tax Rate For Calcs": [.2] * 8})
    q_bs = _stmt({"Total Debt": [100] * 8, "Stockholders Equity": [400] * 8, "Cash And Cash Equivalents": [50] * 8})
    q_cf = _stmt({"Free Cash Flow": [25, 24, 23, 22, 20, 19, 18, 17]})
    est = pd.DataFrame({"avg": {"0q": 1, "+1q": 1, "0y": 5.0, "+1y": 6.0}})
    trend = pd.DataFrame({"current": {"0y": 5.0, "+1y": 6.0}, "30daysAgo": {"0y": 5.0, "+1y": 5.7}})
    info = dict(currentPrice=120.0, marketCap=4000.0, trailingEps=4.5, forwardEps=6.5)
    f = fundamentals.compute(info, q_inc, q_bs, q_cf, None, est, trend)
    assert f["roic"] == pytest.approx(114 * 0.8 / 450) and not f["roic_is_proxy"]
    assert f["sales_growth"] == pytest.approx(490 / 410 - 1) and f["sales_growth_src"] == "TTM vs prior TTM"
    assert f["gm_delta_pts"] == pytest.approx(100 * (294 / 490 - 238 / 410))
    assert f["fcf_ttm"] == 94 and f["fcf_yield"] == pytest.approx(94 / 4000) and "GAAP" in f["fcf_basis"]
    assert f["fwd_eps_growth"] == pytest.approx(0.2) and "consensus" in f["eps_basis"]
    assert f["fwd_pe"] == pytest.approx(20) and f["peg"] == pytest.approx(1.0) and f["peg_meaningful"]
    assert f["eps_rev_30d"] == pytest.approx(0.3 / 5.7)
    assert f["fwd_eps_growth_mixed"] == pytest.approx(6.5 / 4.5 - 1)      # kept, but not used
    assert f["de"] == pytest.approx(0.25) and f["nd_ebitda"] == pytest.approx(50 / 138)


def test_compute_refuses_mixed_basis_and_bad_peg():
    info = dict(currentPrice=50.0, marketCap=1000.0, trailingEps=-0.5, forwardEps=1.0, returnOnEquity=0.2)
    f = fundamentals.compute(info, None, None, None, None, None, None)
    assert f["fwd_eps_growth"] is None and f["peg"] is None and f["peg_meaningful"] is False
    assert "trailing EPS negative" in f["peg_note"]
    assert f["roic_src"] == "ROE — ROIC proxy" and f["roic_is_proxy"]
    assert f["rev_rev_30d"] is None and f["guidance_cut"] is None      # blocked on free data


# ───────────────────────── earnings & gaps
def test_earnings_windows():
    t = dt.date(2026, 11, 5)
    assert qd2.earnings_status(t + dt.timedelta(3), True, t)[0] == "exclude"
    assert qd2.earnings_status(t + dt.timedelta(7), True, t)[0] == "exclude"
    assert qd2.earnings_status(t + dt.timedelta(10), True, t)[0] == "elevated"
    assert qd2.earnings_status(t + dt.timedelta(20), True, t)[0] == "event"
    assert qd2.earnings_status(t + dt.timedelta(40), True, t)[0] == "clear"
    assert qd2.earnings_status(None, True, t)[0] == "clear"
    assert qd2.earnings_status(None, False, t)[0] == "unknown"


def test_earnings_exclusion_and_time_stop_in_evaluate():
    frames, uni, ctx = _setup()
    d = frames["QDIP"]
    e = lambda days: (END + dt.timedelta(days), True)
    r = qd2.evaluate("QDIP", d, "Information Technology", GOOD_FUND, ctx, e(5), END)
    assert r["classification"] == "Excluded" and any("earnings in 5 days" in x for x in r["exclusions"])
    r = qd2.evaluate("QDIP", d, "Information Technology", GOOD_FUND, ctx, e(20), END)
    assert r["classification"] != "Excluded" and r["earn_in_hold"]
    assert r["time_stop"] < END + dt.timedelta(20) and "event risk" in r["earnings_flag"]
    r = qd2.evaluate("QDIP", d, "Information Technology", GOOD_FUND, ctx, (None, True), END)
    assert r["time_stop"] == qd2.bday_add(d.index[-1].date(), P["hold_max"])     # never extended


def _gap(mult, pct=0.92):
    df = raw()
    i = -8
    prev_c = df.close.iloc[i - 1]
    o = prev_c * pct
    set_bar(df, i, o, o * 1.004, o * 0.99, o * 1.001, v=1e6 * mult)
    return ind(df)


def test_gap_down_levels():
    assert qd2.gap_check(_gap(2.5), P)[0] == "exclude"
    assert qd2.gap_check(_gap(1.7), P)[0] == "review"
    assert qd2.gap_check(_gap(1.2), P)[0] is None
    assert qd2.gap_check(_gap(3.0, pct=0.97), P)[0] is None          # −3% gap is not a gap-down event


# ───────────────────────── triggers
def test_trigger_a_rsi2_reversal():
    tr = qd2.triggers(ind(raw()), P)
    assert tr["trig_a"] and not tr["trig_d"] and tr["trig_rank"] == 3 and tr["trig_points"] == 11


def test_trigger_d_volume_and_priority():
    df = raw()
    df.iloc[-1, df.columns.get_loc("volume")] = 1.6e6
    tr = qd2.triggers(ind(df), P)
    assert tr["trig_a"] and tr["trig_d"] and tr["trig_rank"] == 1 and tr["trig_points"] == 15


def test_trigger_b_10ema_reclaim():
    d0 = ind(raw())
    df = raw()
    c = float(d0.ema10.iloc[-2]) * 1.02
    set_bar(df, -1, c * 0.985, c * 1.002, c * 0.98, c)
    tr = qd2.triggers(ind(df), P)
    assert tr["trig_b"]


def test_trigger_c_200sma_support():
    d0 = ind(raw())
    df = raw()
    s200, atr = float(d0.sma200.iloc[-3]), float(d0.atr.iloc[-3])
    lo = s200 + 0.3 * atr
    set_bar(df, -3, lo * 1.01, lo * 1.015, lo, lo * 1.008)
    set_bar(df, -2, lo * 1.008, lo * 1.012, lo * 1.002, lo * 1.006)
    c = lo * 1.03
    set_bar(df, -1, lo * 1.01, c * 1.001, lo * 1.008, c)
    tr = qd2.triggers(ind(df), P)
    assert tr["trig_c"] and tr["trig_rank"] in (2, 3)


def test_no_trigger_and_require_volume_variant():
    df = raw()
    p2 = float(df.close.iloc[-2])
    set_bar(df, -1, p2 * 1.0, p2 * 1.002, p2 * 0.985, p2 * 0.99)      # red, below prior high
    tr = qd2.triggers(ind(df), P)
    assert not any(tr[k] for k in ("trig_a", "trig_c", "trig_d")) and tr["trig_rank"] in (None, 4)
    tr = qd2.triggers(ind(raw()), dict(P, require_volume=True))
    assert tr["trig_rank"] is None                                         # A without volume no longer counts


# ───────────────────────── risk, targets, classification, sizing
def test_risk_plan_numbers_and_flags():
    d = ind(raw())
    entry = float(d.close.iloc[-1])
    rp = qd2.risk_plan(d, entry, P)
    stop = float(d.low.iloc[-10:].min()) - 0.1 * float(d.atr.iloc[-1])
    assert rp["stop"] == pytest.approx(stop)
    risk = entry - stop
    assert rp["t2"] == pytest.approx(entry + 2 * risk) and rp["t3"] == pytest.approx(entry + 3 * risk)
    assert rp["atr_stop_dist"] == pytest.approx(risk / float(d.atr.iloc[-1]))
    assert rp["risk_pct"] == pytest.approx(100 * risk / entry)
    assert rp["resistance"] is not None and rp["avail_R"] == pytest.approx((rp["resistance"] - entry) / risk)
    tight = qd2.risk_plan(d, stop + 1.0 * float(d.atr.iloc[-1]), P)
    wide = qd2.risk_plan(d, stop + 3.0 * float(d.atr.iloc[-1]), P)
    assert tight["atr_flag"].startswith("too tight") and tight["risk_ok"]
    assert wide["atr_flag"].startswith("unfavorable") and not wide["risk_ok"]
    assert not qd2.risk_plan(d, stop - 1, P)["risk_ok"]


def test_classification_gates():
    assert qd2.classify(99, True, True, True) == "Excluded"                # score never overrides exclusion
    assert qd2.classify(99, False, False, True) == "Setting up"
    assert qd2.classify(99, False, True, False) == "Triggered — unacceptable risk"
    assert qd2.classify(85, False, True, True) == "Highest priority"
    assert qd2.classify(80, False, True, True) == "Strong"
    assert qd2.classify(70, False, True, True) == "Watchlist"
    assert qd2.classify(50, False, True, True) == "Pass"


def test_position_sizing():
    s = sizing.position_size(100_000, 0.5, 50.0, 47.0)
    assert s["shares"] == 166 and s["dollar_risk"] == 500 and s["actual_risk"] == pytest.approx(498)
    c = sizing.position_size(100_000, 0.5, 50.0, 49.9, max_position_pct=10)
    assert c["capped"] and c["shares"] == 200
    assert sizing.position_size(100_000, 0.5, 50.0, 51.0)["shares"] == 0
    assert sizing.position_size(0, 0.5, 50.0, 47.0)["shares"] == 0


# ───────────────────────── small known sample
def test_run_screen_known_sample_reports_every_constituent():
    frames, uni, ctx = _setup()
    res = qd2.run_screen(frames, uni, ctx, fake_fundamentals, no_earnings, END)
    rows = {r["symbol"]: r for r in res["rows"]}
    assert set(rows) == set(uni.symbol)                                   # nothing hidden
    q = rows["QDIP"]
    assert q["stage"] == "full" and q["trig_a"] and q["classification"] not in ("Excluded", "Setting up")
    for s, r in rows.items():
        if r["classification"] == "Excluded":
            assert r["exclusions"], s                                        # every exclusion has a reason
    csv = qd2.to_csv_rows(res["rows"])
    assert list(csv.columns) == qd2.CSV_COLUMNS and len(csv) == len(uni)
    assert csv.set_index("ticker").loc["MOMO", "stage"].startswith("failed technical")
    md = qd2.report_markdown(res, ctx)
    assert "Highest-priority" in md and "Excluded" in md


def test_fundamental_fail_excludes_in_run():
    frames, uni, ctx = _setup()
    bad = lambda syms: {s: dict(GOOD_FUND, roic=0.06) for s in syms}
    res = qd2.run_screen(frames, uni, ctx, bad, no_earnings, END)
    q = next(r for r in res["rows"] if r["symbol"] == "QDIP")
    assert q["classification"] == "Excluded" and any(x.startswith("ROIC") for x in q["exclusions"])


def test_volume_confirmation_raises_rank():
    frames, uni, ctx = _setup()
    df = raw()
    df.iloc[-1, df.columns.get_loc("volume")] = 1.8e6
    f2 = dict(frames, QDIP=ind(df))
    base = next(r for r in qd2.run_screen(frames, uni, ctx, fake_fundamentals, no_earnings, END)["rows"] if r["symbol"] == "QDIP")
    vol = next(r for r in qd2.run_screen(f2, uni, ctx, fake_fundamentals, no_earnings, END)["rows"] if r["symbol"] == "QDIP")
    assert vol["trig_rank"] == 1 and vol["tech_score"]["score"] > base["tech_score"]["score"]


def test_strict_preset_is_stricter():
    assert qd2.STRICT["off_high_min"] > P["off_high_min"] and qd2.STRICT["roic_min"] > P["roic_min"]
    t = {"hi252": 100, "close": 88, "off_high": 0.12, "sma50": 95, "sma200": 80, "sma200_20ago": 79, "atr": 2,
         "sma200_slope": 0.01}
    assert not qd2.hard_technical(t, P) and qd2.hard_technical(t, qd2.STRICT)
    t2 = dict(t, sma200_20ago=81, sma200_slope=-0.01)
    assert any("200 SMA falling" in x for x in qd2.hard_technical(t2, P))
