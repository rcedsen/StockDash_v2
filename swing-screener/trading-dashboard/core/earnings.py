"""Upcoming earnings dates. Finnhub (one call for the whole window) if FINNHUB_API_KEY is set,
otherwise yfinance per ticker (only for the shortlisted candidates)."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import requests


def _finnhub(key: str, start: dt.date, end: dt.date) -> dict[str, dt.date]:
    r = requests.get("https://finnhub.io/api/v1/calendar/earnings", timeout=20,
                     params={"from": start.isoformat(), "to": end.isoformat(), "token": key})
    r.raise_for_status()
    out = {}
    for e in r.json().get("earningsCalendar", []):
        d = dt.date.fromisoformat(e["date"])
        s = e["symbol"].replace("-", ".")
        out[s] = min(out.get(s, d), d)
    return out


def _yf_one(sym: str, today: dt.date):
    import yfinance as yf
    try:
        cal = yf.Ticker(sym.replace(".", "-")).calendar or {}
        dates = cal.get("Earnings Date") or []
        fut = sorted(d for d in dates if isinstance(d, dt.date) and d >= today)
        return sym, (fut[0] if fut else None)
    except Exception:
        return sym, None


def next_earnings(symbols: list[str], today: dt.date, window_days: int, finnhub_key: str | None):
    """Returns (dates, status, source). status[sym] in {'in_window','clear','unknown'}."""
    end = today + dt.timedelta(days=window_days)
    status, dates = {}, {}
    if finnhub_key:
        try:
            cal = _finnhub(finnhub_key, today, end)
            for s in symbols:
                dates[s] = cal.get(s)
                status[s] = "in_window" if s in cal else "clear"
            return dates, status, "Finnhub"
        except Exception:
            pass
    with ThreadPoolExecutor(8) as ex:
        for s, d in ex.map(lambda x: _yf_one(x, today), symbols):
            dates[s] = d
            status[s] = "unknown" if d is None else ("in_window" if d <= end else "clear")
    return dates, status, "Yahoo"
