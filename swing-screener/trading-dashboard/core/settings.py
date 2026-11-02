"""App-wide settings. Change values here (or override in .streamlit/secrets.toml under [settings])."""
from __future__ import annotations

DEFAULTS = {
    # Universe sources (see core/universe.py). Add "russell1000" etc. later.
    "universe": ["sp500", "ndx100"],
    # Daily history to load (calendar days). 252-day high + 200 SMA need ~300 trading days.
    "history_days": 460,
    # Results
    "top_n": 10,
    # Earnings exclusion: drop tickers reporting within this many calendar days
    "earnings_window_days": 7,
    # ATR stop multiple (entry - k * ATR14)
    "atr_stop_mult": 1.5,
    # Outcome tracking
    "outcome_hold_days": 5,      # trading days a signal is tracked
    "outcome_target_r": 2.0,     # target used to score outcomes (R multiple of structure stop)
    # Regime / context
    "regime_symbol": "SPY",
    "vix_warn_level": 25.0,
}

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def get_settings() -> dict:
    s = dict(DEFAULTS)
    try:
        import streamlit as st
        over = st.secrets.get("settings", {})
        s.update(dict(over))
    except Exception:
        pass
    return s


def secret(name: str, default=None):
    """Read a top-level secret without crashing when secrets.toml is absent."""
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default
