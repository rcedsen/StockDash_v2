"""Shared indicator library. Computed once per ticker and reused by every strategy.
Add a column here and every strategy can use it."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def trend_r2(c: pd.Series, n: int = 63) -> float:
    """R² of a straight-line fit to log price over n bars: 1 = perfectly smooth trend."""
    y = np.log(c.tail(n).values)
    if len(y) < n or not np.isfinite(y).all():
        return np.nan
    x = np.arange(n)
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r) if np.polyfit(x, y, 1)[0] > 0 else 0.0


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d.close, d.high, d.low
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    d["ema10"] = c.ewm(span=10, adjust=False).mean()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi2"] = rsi(c, 2)
    d["rsi14"] = rsi(c, 14)
    # volume: vol_proj = full-day equivalent (only differs from volume on a live partial bar)
    if "vol_proj" not in d:
        d["vol_proj"] = d.volume
    d["vol50"] = d.volume.shift(1).rolling(50).mean()          # prior 50 completed days
    d["hi20_prior"] = h.shift(1).rolling(20).max()              # 20-day high excluding today
    d["lo20_prior"] = l.shift(1).rolling(20).min()
    d["hi10"] = h.rolling(10).max()
    d["hi252"] = h.rolling(252, min_periods=200).max()
    d["lo50"] = l.rolling(50).min()
    d["ret21"] = c / c.shift(21) - 1
    d["ret63"] = c / c.shift(63) - 1
    d["chg1"] = c / c.shift(1) - 1
    return d
