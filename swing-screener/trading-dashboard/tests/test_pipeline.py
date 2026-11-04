import datetime as dt

import pandas as pd

from core import context, outcomes, pipeline
from core.settings import DEFAULTS
from core.strategy import discover
from tests.synthetic import SyntheticProvider, fake_fundamentals, universe

END = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).date()


def _setup():
    p = SyntheticProvider(END)
    hist = p.history(list(p.data), END - dt.timedelta(days=500), END + dt.timedelta(days=1))
    frames = pipeline.build_frames(hist, {}, None, live=False)
    uni = universe()
    ctx = context.build_context(frames, uni, None, DEFAULTS)
    return frames, uni, ctx


def no_earnings(syms):
    return {s: None for s in syms}, set(syms), "test"


LOOKUPS = {"fundamentals": fake_fundamentals}
TODAY = END


def _strat(key):
    return next(x for x in discover()[0] if x.key == key)


def test_discovery():
    strats, errs = discover()
    assert not errs, errs
    assert [s.key for s in strats] == ["breakout", "pullback", "oversold", "momentum", "quality_dip"]
    for s in strats:
        d = s.describe()
        assert s.hold[0] <= s.hold[1] and s.target_r > 0 and d["rules"] and d["score_parts"]
        assert not any("{" in r for r in d["rules"] + d["score_parts"] + [d["trigger"]]), d["name"]


def test_each_strategy_finds_its_planted_setup():
    frames, uni, ctx = _setup()
    expect = {"breakout": "BRKO", "pullback": "PULL", "oversold": "OVSD", "momentum": "MOMO", "quality_dip": "QDIP"}
    for s in discover()[0]:
        res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, no_earnings, LOOKUPS, TODAY)
        syms = [r["symbol"] for r in res["results"]]
        assert expect[s.key] in syms, (s.key, syms)
        for r in res["results"]:
            assert r["stop_structure"] < r["entry"], r
            assert r["target"] > r["entry"]
            assert 0 <= r["score"] <= 100
            assert (r["hold_min"], r["hold_max"]) == s.hold
            assert r["time_stop"] == pipeline.bday_add(r["bar_date"], s.hold[1])


def test_quality_dip_details():
    frames, uni, ctx = _setup()
    s = _strat("quality_dip")
    res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, no_earnings, LOOKUPS, TODAY)
    q = next(r for r in res["results"] if r["symbol"] == "QDIP")
    assert q["tags"] == ["Ready"]                         # turn-up day after an RSI(2) washout
    assert [r["symbol"] for r in res["results"]] == ["QDIP"]   # others fail fundamentals
    assert "fundamentals" in res["extra_note"]
    # fundamentals only requested for prefilter survivors
    asked = []
    spy = lambda syms: (asked.extend(syms), fake_fundamentals(syms))[1]
    pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, no_earnings, {"fundamentals": spy}, TODAY)
    assert "QDIP" in asked and len(asked) < len(uni) / 2


def test_quality_dip_rejects_failing_metric_and_gap():
    frames, uni, ctx = _setup()
    s = _strat("quality_dip")
    bad = lambda syms: {x: dict(fake_fundamentals([x])[x], peg=2.4) for x in syms}
    res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, no_earnings, {"fundamentals": bad}, TODAY)
    assert "QDIP" not in [r["symbol"] for r in res["results"]]
    # plant a heavy-volume 8% gap-down 10 sessions ago
    from core import indicators
    d = frames["QDIP"][["open", "high", "low", "close", "volume"]].copy()
    i = d.index[-10]
    d.loc[i, ["open", "high", "low", "close"]] = d.loc[i, ["open", "high", "low", "close"]] * 0.92
    d.loc[i, "high"] = max(d.loc[i, "open"], d.loc[i, "close"]) * 1.002
    d.loc[i, "volume"] = 4e6
    d["vol_proj"] = d.volume
    f2 = dict(frames, QDIP=indicators.add_all(d))
    res = pipeline.run_strategy(s, f2, uni, ctx, DEFAULTS, no_earnings, LOOKUPS, TODAY)
    assert "QDIP" not in [r["symbol"] for r in res["results"]]


def test_earnings_exclusion_and_time_stop_cap():
    import datetime as dt
    frames, uni, ctx = _setup()
    s = _strat("momentum")
    near = lambda syms: ({x: (TODAY + dt.timedelta(days=3)) if x == "MOMO" else None for x in syms}, set(syms), "t")
    res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, near, LOOKUPS, TODAY)
    assert "MOMO" not in [r["symbol"] for r in res["results"]]
    assert res["excluded_earnings"] >= 1
    # a report inside the 20-day hold but outside the 7-day exclusion → kept, time stop moved before it
    later = TODAY + dt.timedelta(days=14)
    mid = lambda syms: ({x: later if x == "MOMO" else None for x in syms}, set(syms), "t")
    res = pipeline.run_strategy(s, frames, uni, ctx, DEFAULTS, mid, LOOKUPS, TODAY)
    m = next(r for r in res["results"] if r["symbol"] == "MOMO")
    assert m["earn_in_hold"] and m["time_stop"] < later


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
