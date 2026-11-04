"""Fundamentals for strategies that declare needs = ("fundamentals",).

Source: Yahoo via yfinance (no key) for most metrics; Finnhub (free key) adds ROI TTM, used as the
ROIC figure. Without Finnhub, ROE is used as the ROIC proxy (paired with the leverage filter).
Fetched only for tickers that pass a strategy's price prefilter, and cached for a week by the app.

Every metric is a fraction (0.15 = 15%) or None when unavailable."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

_fh_lock = threading.Lock()
_fh_last = [0.0]


def _num(x):
    try:
        x = float(x)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame, *names) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index(ascending=False).dropna()
    return None


def _finnhub_metrics(sym: str, key: str) -> dict:
    with _fh_lock:                       # free plan: 60 calls/min
        wait = 1.05 - (time.time() - _fh_last[0])
        if wait > 0:
            time.sleep(wait)
        _fh_last[0] = time.time()
    r = requests.get("https://finnhub.io/api/v1/stock/metric", timeout=20,
                     params={"symbol": sym, "metric": "all", "token": key})
    r.raise_for_status()
    return (r.json() or {}).get("metric") or {}


def fetch_one(sym: str, finnhub_key: str | None = None) -> dict:
    import yfinance as yf
    out = dict(symbol=sym, roic=None, roic_src=None, roe=None, sales_growth=None, sales_growth_src=None,
               gm=None, gm_delta_pts=None, gm_src=None, fcf_yield=None, fcf_margin=None, de=None,
               nd_ebitda=None, fwd_eps_growth=None, fwd_pe=None, peg=None, error=None)
    try:
        t = yf.Ticker(sym.replace(".", "-"))
        info = t.get_info() or {}
        mcap, rev_ttm = _num(info.get("marketCap")), _num(info.get("totalRevenue"))
        fcf = _num(info.get("freeCashflow"))
        if fcf is not None and mcap:
            out["fcf_yield"] = fcf / mcap
        if fcf is not None and rev_ttm:
            out["fcf_margin"] = fcf / rev_ttm
        de = _num(info.get("debtToEquity"))           # Yahoo reports this as a percentage
        out["de"] = de / 100 if de is not None else None
        debt, cash, ebitda = (_num(info.get(k)) for k in ("totalDebt", "totalCash", "ebitda"))
        if ebitda and ebitda > 0:
            out["nd_ebitda"] = ((debt or 0) - (cash or 0)) / ebitda
        out["roe"] = _num(info.get("returnOnEquity"))
        fe, te = _num(info.get("forwardEps")), _num(info.get("trailingEps"))
        if fe is not None and te and te > 0:
            out["fwd_eps_growth"] = fe / te - 1
        out["fwd_pe"] = _num(info.get("forwardPE"))
        g = out["fwd_eps_growth"]
        if out["fwd_pe"] and g and g > 0:
            out["peg"] = out["fwd_pe"] / (g * 100)

        q = t.quarterly_income_stmt
        rev = _row(q, "Total Revenue", "Operating Revenue")
        gp = _row(q, "Gross Profit")
        if rev is not None and len(rev) >= 8:
            out["sales_growth"] = rev.iloc[:4].sum() / rev.iloc[4:8].sum() - 1
            out["sales_growth_src"] = "TTM Y/Y"
        elif rev is not None and len(rev) >= 5:
            out["sales_growth"] = rev.iloc[0] / rev.iloc[4] - 1
            out["sales_growth_src"] = "latest qtr Y/Y"
        elif _num(info.get("revenueGrowth")) is not None:
            out["sales_growth"] = _num(info.get("revenueGrowth"))
            out["sales_growth_src"] = "latest qtr Y/Y (Yahoo)"
        if rev is not None and gp is not None:
            m = (gp / rev).dropna()
            if len(m) >= 8:
                now = gp.iloc[:4].sum() / rev.iloc[:4].sum()
                prev = gp.iloc[4:8].sum() / rev.iloc[4:8].sum()
                out.update(gm=now, gm_delta_pts=100 * (now - prev), gm_src="TTM vs prior TTM")
            elif len(m) >= 5:
                out.update(gm=m.iloc[0], gm_delta_pts=100 * (m.iloc[0] - m.iloc[4]), gm_src="latest qtr vs year-ago qtr")
        if out["gm"] is None and _num(info.get("grossMargins")) is not None:
            out["gm"] = _num(info.get("grossMargins"))
    except Exception as e:
        out["error"] = f"Yahoo: {type(e).__name__}"

    if finnhub_key:
        try:
            m = _finnhub_metrics(sym, finnhub_key)
            roi = _num(m.get("roiTTM"))
            if roi is not None:
                out["roic"], out["roic_src"] = roi / 100, "ROI TTM (Finnhub)"
            if out["sales_growth_src"] != "TTM Y/Y" and _num(m.get("revenueGrowthTTMYoy")) is not None:
                out["sales_growth"] = _num(m.get("revenueGrowthTTMYoy")) / 100
                out["sales_growth_src"] = "TTM Y/Y (Finnhub)"
        except Exception:
            pass
    if out["roic"] is None and out["roe"] is not None:
        out["roic"], out["roic_src"] = out["roe"], "ROE (proxy)"
    return out


def fetch_many(symbols: list[str], finnhub_key: str | None = None) -> dict[str, dict]:
    with ThreadPoolExecutor(6) as ex:
        return {d["symbol"]: d for d in ex.map(lambda s: fetch_one(s, finnhub_key), symbols)}
