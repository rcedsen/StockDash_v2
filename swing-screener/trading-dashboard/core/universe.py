"""Ticker universe. Each source is a function returning a DataFrame[symbol, name, sector].
To add a universe (e.g. Russell 1000), write a function and register it in SOURCES."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (trading-dashboard; personal use)"}
FALLBACK = Path(__file__).resolve().parent.parent / "data" / "universe_fallback.csv"


def _wiki_table(url: str) -> pd.DataFrame:
    html = requests.get(url, headers=UA, timeout=20).text
    return pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]


def sp500() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return pd.DataFrame({"symbol": t["Symbol"], "name": t["Security"], "sector": t["GICS Sector"]})


def ndx100() -> pd.DataFrame:
    t = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100")
    sym = "Ticker" if "Ticker" in t.columns else "Symbol"
    name = "Company" if "Company" in t.columns else t.columns[0]
    sec = next((c for c in t.columns if "Sector" in str(c)), None)
    return pd.DataFrame({"symbol": t[sym], "name": t[name], "sector": t[sec] if sec else None})


SOURCES = {"sp500": sp500, "ndx100": ndx100}


def load_universe(names: list[str]) -> tuple[pd.DataFrame, str]:
    """Returns (universe, note). Falls back to a bundled large-cap list if the web lists fail."""
    frames, failed = [], []
    for n in names:
        try:
            frames.append(SOURCES[n]())
        except Exception as e:  # network / page layout change
            failed.append(f"{n} ({type(e).__name__})")
    if not frames:
        fb = pd.read_csv(FALLBACK)
        return fb, "Using bundled fallback list (~70 large caps) — live index lists failed: " + ", ".join(failed)
    u = pd.concat(frames, ignore_index=True)
    u["symbol"] = u.symbol.astype(str).str.strip().str.upper().str.replace("-", ".", regex=False)
    # S&P sectors win over NDX labels when a ticker is in both
    u = u.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    note = f"{len(u)} tickers" + (f" · failed: {', '.join(failed)}" if failed else "")
    return u, note
