"""Upcoming earnings dates. Finnhub (one call for the whole horizon) if FINNHUB_API_KEY is set,
otherwise yfinance per ticker (only for shortlisted candidates).

next_earnings() returns (dates, known, source):
  dates[sym] = next report date or None
  known      = symbols whose answer is reliable (a None in `known` means "no report in the horizon")
The pipeline decides exclusion / time-stop capping per strategy from these dates."""
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


def next_earnings(symbols: list[str], today: dt.date, horizon_days: int, finnhub_key: str | None):
    end = today + dt.timedelta(days=horizon_days)
    if finnhub_key:
        try:
            cal = _finnhub(finnhub_key, today, end)
            return {s: cal.get(s) for s in symbols}, set(symbols), "Finnhub"
        except Exception:
            pass
    dates, known = {}, set()
    with ThreadPoolExecutor(8) as ex:
        for s, d in ex.map(lambda x: _yf_one(x, today), symbols):
            dates[s] = d
            if d is not None:
                known.add(s)
    return dates, known, "Yahoo"
