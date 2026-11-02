import datetime as dt

import pandas as pd

from core import context, outcomes, pipeline
from core.settings import DEFAULTS
from core.strategy import discover
from tests.synthetic import SyntheticProvider, universe

END = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).date()


def _setup():
    p = SyntheticProvider(END)
    hist = p.history(list(p.data), END - dt.timedelta(days=500), END + dt.timedelta(days=1))
    frames = pipeline.build_frames(hist, {}, None, live=False)
    uni = universe()
    ctx = context.build_context(frames, uni, None, DEFAULTS)
    return frames, uni, ctx


def no_earnings(syms):
    return {s: None for s in syms}, {s: "clear" for s in syms}, "test"


def test_discovery():
    strats, errs = discover()
    assert not errs, errs
    assert [s.key for s in strats] == ["breakout", "pullback", "oversold", "momentum"]


def test_each_strategy_finds_its_planted_setup():
    frames, uni, ctx = _setup()
    expect = {"breakout": "BRKO", "pullback": "PULL", "oversold": "OVSD", "momentum": "MOMO"}
    strats, _ = discover()
    for s in strats:
        res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, no_earnings)
        syms = [r["symbol"] for r in res["results"]]
        assert expect[s.key] in syms, (s.key, syms)
        for r in res["results"]:
            assert r["stop_structure"] < r["entry"], r
            assert r["target_2r"] > r["entry"]
            assert 0 <= r["score"] <= 100


def test_earnings_exclusion():
    frames, uni, ctx = _setup()
    s = next(x for x in discover()[0] if x.key == "momentum")
    in_win = lambda syms: ({x: None for x in syms}, {x: "in_window" if x == "MOMO" else "clear" for x in syms}, "t")
    res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, in_win)
    assert "MOMO" not in [r["symbol"] for r in res["results"]]
    assert res["excluded_earnings"] >= 1


def test_live_volume_projection():
    from zoneinfo import ZoneInfo
    p = SyntheticProvider(END)
    hist = {"X": p.data["BRKO"].iloc[:-1]}
    bar = p.data["BRKO"].iloc[-1].copy()
    bar["volume"] = 1e6
    asof = dt.datetime.combine(bar.name.date(), dt.time(12, 30), ZoneInfo("America/New_York"))
    f = pipeline.build_frames(hist, {"X": bar}, asof, live=True)["X"]
    assert f.vol_proj.iloc[-1] > 2e6       # 1M by 12:30 ≈ 2.3M+ full-day


def _bars(rows, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_outcome_rules():
    # entry 100, stop 95, target 110
    tgt = _bars([(99, 101, 98, 100), (101, 111, 100, 109)])
    assert outcomes.score_signal(100, 95, 110, tgt, 5)["outcome"] == "target"
    stp = _bars([(99, 101, 98, 100), (99, 100, 94, 95)])
    r = outcomes.score_signal(100, 95, 110, stp, 5)
    assert r["outcome"] == "stop" and abs(r["outcome_r"] + 1) < 1e-9
    gap = _bars([(99, 101, 98, 100), (93, 94, 90, 92)])
    assert outcomes.score_signal(100, 95, 110, gap, 5)["outcome_r"] < -1      # gap through stop
    none = _bars([(99, 99.5, 97, 98)] * 5)
    assert outcomes.score_signal(100, 95, 110, none, 5)["outcome"] == "no trigger"
    assert outcomes.score_signal(100, 95, 110, none.iloc[:2], 5) is None       # still open
    tm = _bars([(99, 101, 98, 100)] + [(100, 102, 99, 102.5)] * 4)
    r = outcomes.score_signal(100, 95, 110, tm, 5)
    assert r["outcome"] == "time" and abs(r["outcome_r"] - 0.5) < 1e-9
