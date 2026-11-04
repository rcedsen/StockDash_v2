"""Market data providers behind one interface.

Every provider returns bars as {symbol: DataFrame[open, high, low, close, volume]} indexed by
session date (DatetimeIndex, no tz). Add a provider by subclassing Provider and adding it to
get_provider().

- YFinanceProvider: no key needed (unofficial Yahoo data). Used until Alpaca keys are set.
- AlpacaProvider:   free plan. Daily history from the consolidated SIP feed (free except the
                    latest 15 min), live price from the IEX feed.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
COLS = ["open", "high", "low", "close", "volume"]


# ───────────────────────── market clock helpers
def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def market_open(t: dt.datetime | None = None) -> bool:
    t = t or now_et()
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 570 <= m < 960  # 9:30–16:00 ET (exchange holidays are handled by "no bar today")


# Approximate cumulative share of a full day's volume traded by each minute after the open.
# U-shaped intraday profile; used to project partial-day volume to a full-day equivalent.
_PACE = [(0, 0.0), (30, 0.13), (60, 0.21), (90, 0.27), (120, 0.33), (180, 0.43), (240, 0.52),
         (300, 0.61), (330, 0.66), (360, 0.76), (375, 0.84), (390, 1.0)]


def volume_fraction(t: dt.datetime) -> float:
    m = (t.hour * 60 + t.minute) - 570
    if m >= 390:
        return 1.0
    if m <= 5:
        return 0.04
    xs, ys = zip(*_PACE)
    return float(np.interp(m, xs, ys))


# ───────────────────────── interface
class Provider:
    name = "base"
    live = False  # True when today() returns real-time prices

    def history(self, symbols: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        """Completed daily bars from start up to and excluding end."""
        raise NotImplementedError

    def today(self, symbols: list[str]) -> tuple[dict[str, pd.Series], dt.datetime | None, dt.datetime | None]:
        """Today's (possibly partial) bar per symbol.
        Returns (bars, price_asof, volume_asof). Empty dict before the open / on holidays."""
        raise NotImplementedError


def _std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[COLS].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


# ───────────────────────── Yahoo (no key)
class YFinanceProvider(Provider):
    name = "Yahoo (yfinance)"
    live = True

    @staticmethod
    def _y(s: str) -> str:
        return s.replace(".", "-")

    def _download(self, symbols, **kw) -> dict[str, pd.DataFrame]:
        import yfinance as yf
        out = {}
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i + 200]
            ymap = {self._y(s): s for s in chunk}
            raw = yf.download(list(ymap), interval="1d", auto_adjust=True, group_by="ticker",
                              threads=True, progress=False, **kw)
            if raw is None or raw.empty:
                continue
            for ys, s in ymap.items():
                try:
                    df = raw[ys] if isinstance(raw.columns, pd.MultiIndex) else raw
                    df = _std(df)
                    if len(df):
                        out[s] = df
                except Exception:
                    pass
        return out

    def history(self, symbols, start, end):
        return self._download(symbols, start=start.isoformat(), end=end.isoformat())

    def today(self, symbols):
        t = now_et()
        bars = self._download(symbols, period="1d")
        today = pd.Timestamp(t.date())
        res = {s: df.iloc[-1] for s, df in bars.items() if df.index[-1] == today}
        return res, (t if res else None), (t if res else None)


# ───────────────────────── Alpaca (free plan)
class AlpacaProvider(Provider):
    name = "Alpaca"
    live = True
    URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, key: str, secret: str):
        self.h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _bars(self, symbols, start: dt.datetime, end: dt.datetime) -> dict[str, list]:
        out: dict[str, list] = {}
        for i in range(0, len(symbols), 100):
            p = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": "1Day",
                 "start": start.isoformat(), "end": end.isoformat(), "limit": 10000,
                 "adjustment": "all", "feed": "sip"}
            while True:
                r = requests.get(f"{self.URL}/bars", headers=self.h, params=p, timeout=30)
                r.raise_for_status()
                j = r.json()
                for s, rows in (j.get("bars") or {}).items():
                    out.setdefault(s, []).extend(rows)
                if not j.get("next_page_token"):
                    break
                p["page_token"] = j["next_page_token"]
        return out

    @staticmethod
    def _frame(rows) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(pd.to_datetime(df.t, utc=True).dt.tz_convert(ET).dt.date.values)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return _std(df)

    def history(self, symbols, start, end):
        s = dt.datetime.combine(start, dt.time(0), ET)
        e = dt.datetime.combine(end, dt.time(0), ET) - dt.timedelta(seconds=1)
        return {sym: self._frame(rows) for sym, rows in self._bars(symbols, s, e).items() if rows}

    def today(self, symbols):
        t = now_et()
        vol_asof = t - dt.timedelta(minutes=16)            # free SIP data must be >15 min old
        start = dt.datetime.combine(t.date(), dt.time(0), ET)
        if vol_asof <= start.replace(hour=9, minute=30):
            return {}, None, None
        raw = self._bars(symbols, start, vol_asof)
        bars = {s: self._frame(r).iloc[-1] for s, r in raw.items() if r}
        bars = {s: b for s, b in bars.items() if b.name == pd.Timestamp(t.date())}
        # live last price from IEX (real-time on the free plan)
        for i in range(0, len(symbols), 200):
            try:
                r = requests.get(f"{self.URL}/snapshots", headers=self.h, timeout=30,
                                 params={"symbols": ",".join(symbols[i:i + 200]), "feed": "iex"})
                r.raise_for_status()
                for s, snap in r.json().items():
                    px = ((snap or {}).get("latestTrade") or {}).get("p")
                    if px and s in bars:
                        b = bars[s].copy()
                        b["close"] = px
                        b["high"] = max(b["high"], px)
                        b["low"] = min(b["low"], px)
                        bars[s] = b
            except Exception:
                pass
        return bars, (t if bars else None), (vol_asof if bars else None)


def get_provider(secret) -> Provider:
    k, s = secret("ALPACA_API_KEY"), secret("ALPACA_SECRET_KEY")
    if k and s:
        return AlpacaProvider(k, s)
    return YFinanceProvider()
