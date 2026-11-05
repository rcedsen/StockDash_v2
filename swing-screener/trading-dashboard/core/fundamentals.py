"""Fundamentals for strategies that declare needs = ("fundamentals",).

Sources (free): Yahoo via yfinance (statements, consensus estimates, estimate trend); Finnhub (free key)
as a fallback for ROI and TTM sales growth. Fetched only for tickers that pass a strategy's price
prefilter and cached for a week by the app.

Every ratio is a fraction (0.15 = 15%) or None when unavailable. Each metric carries a `*_basis` label
so results never mix accounting definitions silently:
  - ROIC      = NOPAT (TTM operating income × (1 − tax rate)) ÷ (total debt + equity − cash), latest quarter.
                Fallbacks are labeled and flagged roic_is_proxy=True (Finnhub ROI, then ROE).
  - EPS growth= consensus next-FY EPS ÷ consensus current-FY EPS − 1 (same analyst-adjusted basis).
                Yahoo forward EPS vs GAAP trailing EPS is a MIXED basis: kept as fwd_eps_growth_mixed,
                never used for pass/fail.
  - FCF       = TTM operating cash flow − capex from the GAAP cash-flow statement.
  - Point-in-time history is NOT available from these sources (values are as of today).
Blocked on free data: revenue-estimate revisions, guidance revisions (always None)."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

_fh_lock = threading.Lock()
_fh_last = [0.0]

FIELDS = ["roic", "roic_src", "roic_is_proxy", "roe", "sales_growth", "sales_growth_src", "sales_growth_latest_q",
          "gm", "gm_prior", "gm_delta_pts", "gm_src", "op_margin_delta_pts", "fcf_ttm", "fcf_basis", "fcf_yield",
          "fcf_margin", "de", "nd_ebitda", "leverage_note", "eps_cy", "eps_ny", "eps_basis", "fwd_eps_growth",
          "fwd_eps_growth_mixed", "fwd_pe", "peg", "peg_meaningful", "peg_note", "trailing_eps", "price",
          "eps_rev_30d", "rev_rev_30d", "guidance_cut", "notes", "error"]


def _num(x):
    try:
        x = float(x)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame, *names) -> pd.Series | None:
    """Statement row, newest first, NaNs dropped."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s.index = pd.to_datetime(s.index)
            s = s.sort_index(ascending=False).dropna()
            return s if len(s) else None
    return None


def _ttm(s: pd.Series | None, offset: int = 0):
    if s is None or len(s) < offset + 4:
        return None
    return float(s.iloc[offset:offset + 4].sum())


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


def _safe(fn):
    try:
        v = fn()
        return v if v is not None else None
    except Exception:
        return None


def compute(info: dict, q_inc, q_bs, q_cf, a_inc, est, trend, fh: dict | None = None) -> dict:
    """Pure function: raw Yahoo/Finnhub inputs → metrics dict. Unit-testable without the network."""
    out = {k: None for k in FIELDS}
    out["notes"] = []
    out["roic_is_proxy"] = False
    info = info or {}
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    out["price"] = price
    mcap = _num(info.get("marketCap"))

    # ── revenue / sales growth
    rev_q = _row(q_inc, "Total Revenue", "Operating Revenue")
    rev_ttm = _ttm(rev_q)
    if rev_q is not None and len(rev_q) >= 8:
        out["sales_growth"], out["sales_growth_src"] = rev_ttm / _ttm(rev_q, 4) - 1, "TTM vs prior TTM"
    elif fh and _num(fh.get("revenueGrowthTTMYoy")) is not None:
        out["sales_growth"], out["sales_growth_src"] = _num(fh["revenueGrowthTTMYoy"]) / 100, "TTM Y/Y (Finnhub)"
    else:
        rev_a = _row(a_inc, "Total Revenue", "Operating Revenue")
        if rev_a is not None and len(rev_a) >= 2:
            out["sales_growth"], out["sales_growth_src"] = rev_a.iloc[0] / rev_a.iloc[1] - 1, "latest FY vs prior FY (not TTM)"
        elif rev_q is not None and len(rev_q) >= 5:
            out["sales_growth"], out["sales_growth_src"] = rev_q.iloc[0] / rev_q.iloc[4] - 1, "latest qtr Y/Y (not TTM)"
    if rev_q is not None and len(rev_q) >= 5:
        out["sales_growth_latest_q"] = rev_q.iloc[0] / rev_q.iloc[4] - 1

    # ── gross margin TTM vs prior-year TTM
    gp_q = _row(q_inc, "Gross Profit")
    if rev_q is not None and gp_q is not None and len(rev_q) >= 8 and len(gp_q) >= 8:
        now, prev = _ttm(gp_q) / rev_ttm, _ttm(gp_q, 4) / _ttm(rev_q, 4)
        out.update(gm=now, gm_prior=prev, gm_src="TTM vs prior TTM")
    else:
        gp_a, rev_a = _row(a_inc, "Gross Profit"), _row(a_inc, "Total Revenue", "Operating Revenue")
        if gp_a is not None and rev_a is not None and len(gp_a) >= 2 and len(rev_a) >= 2:
            out.update(gm=gp_a.iloc[0] / rev_a.iloc[0], gm_prior=gp_a.iloc[1] / rev_a.iloc[1],
                       gm_src="latest FY vs prior FY (not TTM)")
        elif gp_q is not None and rev_q is not None and len(gp_q) >= 5 and len(rev_q) >= 5:
            out.update(gm=gp_q.iloc[0] / rev_q.iloc[0], gm_prior=gp_q.iloc[4] / rev_q.iloc[4],
                       gm_src="latest qtr vs year-ago qtr (not TTM)")
    if out["gm"] is not None and out["gm_prior"] is not None:
        out["gm_delta_pts"] = 100 * (out["gm"] - out["gm_prior"])

    # ── operating margin trend (profitability warning)
    op_q = _row(q_inc, "Operating Income", "EBIT")
    if op_q is not None and rev_q is not None and len(op_q) >= 5 and len(rev_q) >= 5:
        out["op_margin_delta_pts"] = 100 * (op_q.iloc[0] / rev_q.iloc[0] - op_q.iloc[4] / rev_q.iloc[4])

    # ── balance sheet
    debt = _row(q_bs, "Total Debt")
    eq = _row(q_bs, "Stockholders Equity", "Common Stock Equity")
    cash = _row(q_bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    debt0 = float(debt.iloc[0]) if debt is not None else _num(info.get("totalDebt"))
    eq0 = float(eq.iloc[0]) if eq is not None else None
    cash0 = float(cash.iloc[0]) if cash is not None else _num(info.get("totalCash"))

    # ── ROIC (computed), else labeled fallbacks
    op_ttm = _ttm(op_q)
    tax_r = _row(q_inc, "Tax Rate For Calcs")
    tax = float(tax_r.iloc[0]) if tax_r is not None else None
    if tax is None:
        pre, prov = _ttm(_row(q_inc, "Pretax Income")), _ttm(_row(q_inc, "Tax Provision"))
        tax = (prov / pre) if (pre and prov is not None and pre > 0) else 0.21
    tax = min(max(tax, 0.0), 0.35)
    if op_ttm is not None and debt0 is not None and eq0 is not None:
        ic = debt0 + eq0 - (cash0 or 0)
        if ic > 0:
            out["roic"], out["roic_src"] = op_ttm * (1 - tax) / ic, "ROIC (NOPAT TTM ÷ invested capital)"
        else:
            out["notes"].append("ROIC not meaningful (invested capital ≤ 0)")
    if out["roic"] is None and fh and _num(fh.get("roiTTM")) is not None:
        out["roic"], out["roic_src"], out["roic_is_proxy"] = _num(fh["roiTTM"]) / 100, "ROI TTM (Finnhub) — ROIC proxy", True
    out["roe"] = _num(info.get("returnOnEquity"))
    if out["roic"] is None and out["roe"] is not None:
        out["roic"], out["roic_src"], out["roic_is_proxy"] = out["roe"], "ROE — ROIC proxy", True

    # ── leverage
    if debt0 is not None and eq0 is not None:
        if eq0 > 0:
            out["de"] = debt0 / eq0
        else:
            out["leverage_note"] = "negative equity — D/E not meaningful"
    elif _num(info.get("debtToEquity")) is not None:
        out["de"] = _num(info["debtToEquity"]) / 100          # Yahoo reports a percentage
    ebitda_ttm = _ttm(_row(q_inc, "EBITDA", "Normalized EBITDA")) or _num(info.get("ebitda"))
    if ebitda_ttm and ebitda_ttm > 0 and debt0 is not None:
        out["nd_ebitda"] = (debt0 - (cash0 or 0)) / ebitda_ttm

    # ── free cash flow (GAAP OCF − capex)
    fcf_q = _row(q_cf, "Free Cash Flow")
    if fcf_q is not None and len(fcf_q) >= 4:
        out["fcf_ttm"], out["fcf_basis"] = _ttm(fcf_q), "TTM OCF − capex (GAAP cash-flow statement)"
    else:
        ocf, capex = _row(q_cf, "Operating Cash Flow"), _row(q_cf, "Capital Expenditure")
        if ocf is not None and capex is not None and len(ocf) >= 4 and len(capex) >= 4:
            out["fcf_ttm"], out["fcf_basis"] = _ttm(ocf) + _ttm(capex), "TTM OCF − capex (GAAP cash-flow statement)"
        elif _num(info.get("freeCashflow")) is not None:
            out["fcf_ttm"], out["fcf_basis"] = _num(info["freeCashflow"]), "Yahoo summary FCF (definition not verified)"
    if out["fcf_ttm"] is not None:
        if mcap:
            out["fcf_yield"] = out["fcf_ttm"] / mcap
        rev_base = rev_ttm or _num(info.get("totalRevenue"))
        if rev_base:
            out["fcf_margin"] = out["fcf_ttm"] / rev_base

    # ── EPS: consensus FY0 → FY+1 (same basis)
    te = _num(info.get("trailingEps"))
    out["trailing_eps"] = te
    if isinstance(est, pd.DataFrame) and not est.empty and "avg" in est.columns:
        cy = _num(est["avg"].get("0y")) if "0y" in est.index else None
        ny = _num(est["avg"].get("+1y")) if "+1y" in est.index else None
        out["eps_cy"], out["eps_ny"] = cy, ny
        if cy is not None and ny is not None and cy > 0:
            out["fwd_eps_growth"], out["eps_basis"] = ny / cy - 1, "consensus FY+1 vs FY0 (analyst-adjusted)"
    fe = _num(info.get("forwardEps"))
    if fe is not None and te and te > 0:
        out["fwd_eps_growth_mixed"] = fe / te - 1
        if out["fwd_eps_growth"] is None:
            out["notes"].append("Only forward (adjusted) vs trailing (GAAP) EPS available — mixed basis, not used")

    # ── forward P/E and PEG (only when meaningful)
    ny, cy, g = out["eps_ny"], out["eps_cy"], out["fwd_eps_growth"]
    if price and ny and ny > 0:
        out["fwd_pe"] = price / ny
    reasons = []
    if not (cy and cy > 0 and price and price / cy <= 150):
        reasons.append("current-FY EPS negative or near zero")
    if not (te and te > 0 and price and price / te <= 150):
        reasons.append("trailing EPS negative or near zero")
    if not (g and g > 0):
        reasons.append("no positive forward growth")
    if out["fwd_pe"] is None:
        reasons.append("no forward P/E")
    if reasons:
        out["peg_meaningful"], out["peg_note"] = False, "; ".join(reasons)
    else:
        out["peg"], out["peg_meaningful"] = out["fwd_pe"] / (g * 100), True

    # ── estimate revisions (EPS only — revenue revisions and guidance are unavailable on free data)
    if isinstance(trend, pd.DataFrame) and not trend.empty and {"current", "30daysAgo"} <= set(trend.columns):
        for per in ("+1y", "0y"):
            if per in trend.index:
                cur, ago = _num(trend.loc[per, "current"]), _num(trend.loc[per, "30daysAgo"])
                if cur is not None and ago:
                    out["eps_rev_30d"] = (cur - ago) / abs(ago)
                    break
    return out


def fetch_one(sym: str, finnhub_key: str | None = None) -> dict:
    import yfinance as yf
    t = yf.Ticker(sym.replace(".", "-"))
    fh = None
    if finnhub_key:
        fh = _safe(lambda: _finnhub_metrics(sym, finnhub_key))
    try:
        out = compute(_safe(t.get_info), _safe(lambda: t.quarterly_income_stmt), _safe(lambda: t.quarterly_balance_sheet),
                      _safe(lambda: t.quarterly_cashflow), _safe(lambda: t.income_stmt),
                      _safe(lambda: t.earnings_estimate), _safe(lambda: t.eps_trend), fh)
    except Exception as e:
        out = {k: None for k in FIELDS}
        out.update(notes=[], roic_is_proxy=False, error=f"{type(e).__name__}: {e}")
    out["symbol"] = sym
    return out


def fetch_many(symbols: list[str], finnhub_key: str | None = None) -> dict[str, dict]:
    with ThreadPoolExecutor(6) as ex:
        return {d["symbol"]: d for d in ex.map(lambda s: fetch_one(s, finnhub_key), symbols)}
