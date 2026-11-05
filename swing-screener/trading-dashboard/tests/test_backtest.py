"""Backtest module: execution rules, point-in-time lookups, metrics, walk-forward, no look-ahead."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from core import backtest, qd2
from core.settings import SECTOR_ETF
from tests.synthetic import GOOD_FUND, _ohlcv


def bars(rows, start="2026-01-05"):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range(start, periods=len(rows)))


# ───────────────────────── execution rules
def test_same_bar_stop_first_and_targets():
    r = backtest.simulate_trade(bars([(100, 112, 94, 105)] + [(105, 106, 104, 105)] * 40), 100, 95, 110, 30, slip_bps=0)
    assert r["reason"] == "stop" and r["R"] == pytest.approx(-1)
    r = backtest.simulate_trade(bars([(100, 101, 99, 100), (101, 111, 100, 110)] + [(110,) * 4] * 40), 100, 95, 110, 30, slip_bps=0)
    assert r["reason"] == "target" and r["R"] == pytest.approx(2) and r["sessions"] == 2


def test_gap_through_stop_fills_at_open():
    r = backtest.simulate_trade(bars([(100, 101, 99, 100), (92, 93, 90, 91)] + [(91,) * 4] * 40), 100, 95, 110, 30, slip_bps=0)
    assert r["reason"] == "stop" and r["R"] == pytest.approx(-8 / 5)


def test_time_stop_earnings_exit_and_open_trades():
    flat = [(100, 101, 99, 100.5)] * 35
    r = backtest.simulate_trade(bars(flat), 100, 95, 115, 30, slip_bps=0)
    assert r["reason"] == "time" and r["sessions"] == 30 and r["R"] == pytest.approx(0.1)
    b = bars(flat)
    eb = b.index[9].date()
    r = backtest.simulate_trade(b, 100, 95, 115, 30, exit_before=eb, slip_bps=0)
    assert r["reason"] == "earnings exit" and r["exit_date"] == eb and r["sessions"] == 10
    assert backtest.simulate_trade(bars(flat[:12]), 100, 95, 115, 30, slip_bps=0) == dict(open=True)


def test_costs_reduce_R():
    b = bars([(100, 101, 99, 100), (101, 111, 100, 110)] + [(110,) * 4] * 40)
    free = backtest.simulate_trade(b, 100, 95, 110, 30, slip_bps=0, commission_ps=0)["R"]
    paid = backtest.simulate_trade(b, 100, 95, 110, 30, slip_bps=10, commission_ps=0.01)["R"]
    assert paid < free


# ───────────────────────── point-in-time lookups
def test_pit_fundamentals_and_earnings():
    f = backtest.PITFundamentals(pd.DataFrame([dict(asof_date="2025-02-01", symbol="X", roic=0.1),
                                               dict(asof_date="2025-05-01", symbol="X", roic=0.2)]))
    assert f.get("X", pd.Timestamp("2025-01-15")) is None
    assert f.get("X", pd.Timestamp("2025-03-01"))["roic"] == 0.1
    assert f.get("X", pd.Timestamp("2025-05-01"))["roic"] == 0.2
    e = backtest.Earnings(pd.DataFrame(dict(symbol=["X", "X"], date=["2025-02-10", "2025-05-10"])))
    assert e.next_after("X", dt.date(2025, 2, 11)) == dt.date(2025, 5, 10)


# ───────────────────────── metrics & walk-forward
def test_metrics_math():
    t = pd.DataFrame(dict(R=[2.0, -1.0, 3.0, -1.0, -1.0], sessions=[5, 3, 10, 2, 4], reason=["target", "stop", "target", "stop", "stop"],
                          exit_date=pd.date_range("2025-01-01", periods=5).date, entry_date=pd.date_range("2025-01-01", periods=5).date))
    m = backtest.metrics(t)
    assert m["trades"] == 5 and m["win_rate"] == 40 and m["expectancy_R"] == pytest.approx(0.4)
    assert m["profit_factor"] == pytest.approx(5 / 3) and m["max_drawdown_R"] == pytest.approx(2.0)
    assert m["avg_win_R"] == 2.5 and m["avg_loss_R"] == -1


def test_walk_forward_selects_on_past_only():
    days = pd.bdate_range("2024-01-01", periods=400).date

    def mk(early, late):
        return pd.DataFrame(dict(signal_date=days, R=[early if i < 200 else late for i in range(400)],
                                 sessions=5, reason="x", exit_date=days, entry_date=days))
    res = {"A": dict(trades=mk(1.0, -1.0)), "B": dict(trades=mk(-0.5, 2.0))}
    wf = backtest.walk_forward(res, folds=4, min_trades=20)
    assert wf.iloc[0].chosen == "A"                      # only fold 0 (A good) is visible
    assert wf.iloc[0].oos_expectancy_R == pytest.approx(1.0)
    assert wf.iloc[-1].chosen == "A" or wf.iloc[-1].chosen == "B"


# ───────────────────────── end-to-end on a synthetic panel
def panel(n=420):
    idx = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n)
    r = np.random.default_rng(3)
    out = {"SPY": _ohlcv(400 * np.exp(0.0005 * t + np.cumsum(r.normal(0, 0.002, n))), seed=1)}
    for j, e in enumerate(SECTOR_ETF.values()):
        out[e] = _ohlcv(50 * np.exp(0.0003 * t + np.cumsum(r.normal(0, 0.003, n))), seed=10 + j)
    c = 50 * np.exp(0.003 * t)
    k = 330                                             # dip, reversal at k, then rally
    peak = c[k - 26]
    c[k - 25:k] = peak * np.linspace(0.99, 0.80, 25)
    c[k] = c[k - 1] * 1.03
    c[k + 1:] = c[k] * np.exp(0.006 * np.arange(1, n - k))
    v = np.full(n, 1e6)
    v[k] = 1.8e6
    out["QQ"] = _ohlcv(c, v, rng=0.004, seed=7)
    for d in out.values():
        d.index = idx
    return out, idx[k]


def test_backtest_end_to_end_and_no_lookahead():
    prices, sig_day = panel()
    frames = backtest.prepare(prices)
    uni = pd.DataFrame(dict(symbol=["QQ"], name=["QQ"], sector=["Information Technology"]))
    fund = pd.DataFrame([dict(asof_date="2024-01-01", symbol="QQ", **{k: GOOD_FUND[k] for k in backtest.FUND_COLS if k in GOOD_FUND})])
    res = backtest.run(frames, universe=uni, fundamentals=fund, classes=tuple(qd2.CLASS_ORDER[:4]), slip_bps=5)
    t = res["trades"]
    assert len(t) >= 1 and res["survivorship_bias"] and res["fundamentals_used"]
    first = t.iloc[0]
    assert first.signal_date == sig_day.date()
    nxt = frames["QQ"].index[frames["QQ"].index.get_loc(sig_day) + 1]
    assert first.entry_date == nxt.date()
    assert first.entry == pytest.approx(float(frames["QQ"].loc[nxt, "open"]) * 1.0005)      # next open + slippage
    assert first.reason == "target"
    # no look-ahead: truncating the data right after the signal leaves the signal-day evaluation unchanged
    cut = {s: d[d.index <= nxt] for s, d in prices.items()}
    f2 = backtest.prepare(cut)
    ctx = backtest.ctx_for(sig_day, backtest.build_daily_context(frames, frames["SPY"].index, ["QQ"]), {"QQ"},
                           {"QQ": "Information Technology"})
    ctx2 = backtest.ctx_for(sig_day, backtest.build_daily_context(f2, f2["SPY"].index, ["QQ"]), {"QQ"},
                            {"QQ": "Information Technology"})
    a = qd2.evaluate("QQ", frames["QQ"].loc[:sig_day], "Information Technology", GOOD_FUND, ctx, None, sig_day.date())
    b = qd2.evaluate("QQ", f2["QQ"].loc[:sig_day], "Information Technology", GOOD_FUND, ctx2, None, sig_day.date())
    assert a["total"] == pytest.approx(b["total"]) and a["classification"] == b["classification"]
    assert a["risk"]["stop"] == pytest.approx(b["risk"]["stop"])


def test_technical_only_mode_is_labeled():
    prices, _ = panel()
    frames = backtest.prepare(prices)
    uni = pd.DataFrame(dict(symbol=["QQ"], name=["QQ"], sector=["Information Technology"]))
    res = backtest.run(frames, universe=uni, classes=tuple(qd2.CLASS_ORDER[:4]))
    assert res["fundamentals_used"] is False and res["earnings_used"] is False
    md = backtest.summary_markdown({"revised_3R": res}, backtest.walk_forward({"revised_3R": res}))
    assert "NOT USED" in md and "Survivorship bias: YES" in md
