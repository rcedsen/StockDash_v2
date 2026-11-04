"""Ticker universes. Each source is a function returning a DataFrame[symbol, name, sector].
To add a universe, write a function and register it in SOURCES (and optionally FALLBACKS).
A strategy picks its universe with `universe = [...]`; None = the app default in settings."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (trading-dashboard; personal use)"}
FALLBACK = Path(__file__).resolve().parent.parent / "data" / "universe_fallback.csv"

# iShares sector labels → GICS names used everywhere else (sector ETF mapping)
_SECTOR_FIX = {"Communication": "Communication Services", "Information Tech": "Information Technology"}
# iShares writes share classes without a dot
_CLASS_FIX = {"BRKB": "BRK.B", "BFB": "BF.B", "BFA": "BF.A", "LENB": "LEN.B", "HEIA": "HEI.A",
              "UHALB": "UHAL.B", "FOXA": "FOXA", "NWSA": "NWSA", "GOOGL": "GOOGL", "CWENA": "CWEN.A"}


def _wiki_table(url: str) -> pd.DataFrame:
    html = requests.get(url, headers=UA, timeout=20).text
    return pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]


def _col(t: pd.DataFrame, *cands):
    for c in t.columns:
        if any(k.lower() in str(c).lower() for k in cands):
            return c
    return None


def sp500() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return pd.DataFrame({"symbol": t["Symbol"], "name": t["Security"], "sector": t["GICS Sector"]})


def ndx100() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100")
    sym = "Ticker" if "Ticker" in t.columns else "Symbol"
    name = "Company" if "Company" in t.columns else t.columns[0]
    sec = next((c for c in t.columns if "Sector" in str(c)), None)
    return pd.DataFrame({"symbol": t[sym], "name": t[name], "sector": t[sec] if sec else None})


def sp400() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    sym, name, sec = _col(t, "symbol", "ticker"), _col(t, "security", "company"), _col(t, "gics sector", "sector")
    return pd.DataFrame({"symbol": t[sym], "name": t[name], "sector": t[sec] if sec else None})


def russell1000() -> pd.DataFrame:
    """Holdings of the iShares Russell 1000 ETF (IWB), published daily."""
    url = ("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWB_holdings&dataType=fund")
    txt = requests.get(url, headers=UA, timeout=30).text
    lines = txt.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip('﻿"').startswith("Ticker"))
    t = pd.read_csv(io.StringIO("\n".join(lines[start:])), on_bad_lines="skip")
    if "Asset Class" in t.columns:
        t = t[t["Asset Class"].astype(str).str.strip().eq("Equity")]
    t = t[t["Ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}")]
    sym = t["Ticker"].astype(str).str.strip().map(lambda s: _CLASS_FIX.get(s, s))
    sec = t["Sector"].astype(str).str.strip().map(lambda s: _SECTOR_FIX.get(s, s)) if "Sector" in t else None
    out = pd.DataFrame({"symbol": sym.values, "name": t["Name"].astype(str).str.title().values,
                        "sector": sec.values if sec is not None else None})
    if len(out) < 800:   # sanity check — a layout change would return a partial list
        raise ValueError(f"only {len(out)} holdings parsed")
    return out


SOURCES = {"sp500": sp500, "ndx100": ndx100, "sp400": sp400, "russell1000": russell1000}
FALLBACKS = {"russell1000": ["sp500", "sp400"]}   # used if the primary source fails


def load_universe(names: list[str]) -> tuple[pd.DataFrame, str]:
    """Returns (universe, note). Falls back to a bundled large-cap list if every web list fails."""
    frames, failed, used = [], [], []
    for n in names:
        try:
            frames.append(SOURCES[n]())
            used.append(n)
        except Exception as e:  # network / page layout change
            failed.append(f"{n} ({type(e).__name__})")
            for fb in FALLBACKS.get(n, []):
                if fb in used or fb in names:
                    continue
                try:
                    frames.append(SOURCES[fb]())
                    used.append(f"{fb} (fallback for {n})")
                except Exception as e2:
                    failed.append(f"{fb} ({type(e2).__name__})")
    if not frames:
        fb = pd.read_csv(FALLBACK)
        return fb, "Using bundled fallback list (~70 large caps) — live index lists failed: " + ", ".join(failed)
    u = pd.concat(frames, ignore_index=True)
    u["symbol"] = u.symbol.astype(str).str.strip().str.upper().str.replace("-", ".", regex=False)
    u["sector"] = u.sector.where(u.sector.notna() & (u.sector.astype(str) != "nan"), None)
    # earlier sources win for name/sector (S&P GICS labels are listed first)
    u = u.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    note = f"{len(u)} tickers from {', '.join(used)}" + (f" · failed: {', '.join(failed)}" if failed else "")
    return u, note
