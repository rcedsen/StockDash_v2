"""Synthetic market used by the tests: random-walk filler plus one planted setup per strategy."""
import datetime as dt

import numpy as np
import pandas as pd

from core.data import Provider
from core.settings import SECTOR_ETF

N = 320


def _ohlcv(close, vol=None, rng=0.008, seed=0):
    r = np.random.default_rng(seed)
    close = np.asarray(close, float)
    op = np.r_[close[0], close[:-1]] * (1 + r.normal(0, rng / 3, len(close)))
    hi = np.maximum(op, close) * (1 + np.abs(r.normal(0, rng / 2, len(close))))
    lo = np.minimum(op, close) * (1 - np.abs(r.normal(0, rng / 2, len(close))))
    v = np.full(len(close), 1e6) * (1 + r.normal(0, 0.1, len(close))) if vol is None else vol
    return pd.DataFrame(dict(open=op, high=hi, low=lo, close=close, volume=v))


def market(end: dt.date):
    idx = pd.bdate_range(end=end, periods=N)
    out = {}
    r = np.random.default_rng(1)
    # filler: driftless random walks
    for i in range(40):
        c = 100 * np.exp(np.cumsum(r.normal(0, 0.012, N)))
        out[f"R{i:02d}"] = _ohlcv(c, seed=i)
    t = np.arange(N)
    out["SPY"] = _ohlcv(500 * np.exp(0.0004 * t + np.cumsum(r.normal(0, 0.003, N))), seed=99)
    for j, etf in enumerate(SECTOR_ETF.values()):
        out[etf] = _ohlcv(80 * np.exp(0.0003 * t + np.cumsum(r.normal(0, 0.004, N))), seed=200 + j)

    # Breakout: trend, 25-day tight base, then a high-volume close above the base
    c = 50 * np.exp(0.002 * t)
    c[-26:-1] = c[-27] * (1 + 0.004 * np.sin(np.arange(25)))
    c[-1] = c[-27] * 1.03
    v = np.full(N, 1e6); v[-1] = 2.6e6
    out["BRKO"] = _ohlcv(c, v, rng=0.004, seed=11)

    # Pullback: steady uptrend, last 6 bars ease back to the 20 EMA on light volume, green last bar
    c = 40 * np.exp(0.003 * t)
    c[-6:] = c[-7] * np.array([0.99, 0.978, 0.965, 0.952, 0.942, 0.948])
    v = np.full(N, 1e6); v[-6:] = 6e5
    df = _ohlcv(c, v, rng=0.006, seed=12)
    df.loc[df.index[-1], "open"] = c[-2] * 0.998
    out["PULL"] = df

    # Oversold: strong uptrend, then three hard down days (still far above the 200 SMA)
    c = 30 * np.exp(0.0035 * t)
    c[-3:] = c[-4] * np.array([0.965, 0.93, 0.90])
    out["OVSD"] = _ohlcv(c, rng=0.006, seed=13)

    # Momentum: smooth, strong uptrend at its highs
    c = 20 * np.exp(0.005 * t + np.cumsum(r.normal(0, 0.002, N)))
    out["MOMO"] = _ohlcv(c, rng=0.006, seed=14)

    # Quality dip: long uptrend, ~20% orderly decline over 25 sessions (no gap), then a turn-up day
    c = 60 * np.exp(0.003 * t)
    peak = c[-27]
    c[-26:] = peak
    c[-26:-1] = peak * np.linspace(0.99, 0.80, 25)
    c[-1] = c[-2] * 1.03
    out["QDIP"] = _ohlcv(c, rng=0.004, seed=15)

    for s, df in out.items():
        df.index = idx
    return out


def universe():
    syms = [f"R{i:02d}" for i in range(40)] + ["BRKO", "PULL", "OVSD", "MOMO", "QDIP"]
    secs = list(SECTOR_ETF)
    return pd.DataFrame(dict(symbol=syms, name=syms, sector=[secs[i % len(secs)] for i in range(len(syms))]))


GOOD_FUND = dict(roic=0.24, roic_src="ROI TTM", roe=0.3, sales_growth=0.22, sales_growth_src="TTM Y/Y",
                 gm=0.62, gm_delta_pts=0.8, gm_src="TTM vs prior TTM", fcf_yield=0.028, fcf_margin=0.21,
                 de=0.35, nd_ebitda=0.4, fwd_eps_growth=0.24, fwd_pe=28.0, peg=28.0 / 24, error=None)


def fake_fundamentals(symbols, key=None):
    """Only QDIP has fundamentals that pass; everything else fails sales growth."""
    return {s: dict(GOOD_FUND, symbol=s) if s == "QDIP" else dict(GOOD_FUND, symbol=s, sales_growth=0.05)
            for s in symbols}


class SyntheticProvider(Provider):
    name = "synthetic"
    live = False

    def __init__(self, end):
        self.data = market(end)

    def history(self, symbols, start, end):
        return {s: d[(d.index >= pd.Timestamp(start)) & (d.index < pd.Timestamp(end))]
                for s, d in self.data.items() if s in symbols}

    def today(self, symbols):
        return {}, None, None
