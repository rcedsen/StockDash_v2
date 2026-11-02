"""Shared screening pipeline: bars → indicators → strategy → rank → earnings filter → top N → enrich.
Every strategy goes through this, so warnings, earnings exclusion and card fields apply to all."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import indicators
from .data import volume_fraction
from .settings import SECTOR_ETF
from .strategy import Strategy


def build_frames(history: dict[str, pd.DataFrame], today_bars: dict[str, pd.Series],
                 vol_asof: dt.datetime | None, live: bool) -> dict[str, pd.DataFrame]:
    """Merge cached history with today's (partial) bar and compute indicators."""
    frames = {}
    frac = volume_fraction(vol_asof) if (live and vol_asof) else 1.0
    for s, h in history.items():
        d = h.copy()
        b = today_bars.get(s)
        if b is not None:
            d = d[d.index < b.name]
            d.loc[b.name] = b[["open", "high", "low", "close", "volume"]].astype(float)
        d["vol_proj"] = d.volume.astype(float)
        if b is not None and frac < 1.0:
            d.iloc[-1, d.columns.get_loc("vol_proj")] = float(b["volume"]) / max(frac, 0.04)
        if len(d) >= 30:
            frames[s] = indicators.add_all(d)
    return frames


def run_strategy(strategy: Strategy, frames: dict[str, pd.DataFrame], universe: pd.DataFrame,
                 ctx: dict, settings: dict, earnings_lookup) -> dict:
    """earnings_lookup(symbols) -> (dates, status, source)."""
    sector_of = dict(zip(universe.symbol, universe.sector))
    name_of = dict(zip(universe.symbol, universe.name))
    hits, errors = [], 0
    for s in universe.symbol:
        d = frames.get(s)
        if d is None:
            continue
        try:
            st = strategy.evaluate(d, s, ctx)
        except Exception:
            errors += 1
            continue
        if st is not None and np.isfinite(st.score):
            hits.append((s, st))
    hits.sort(key=lambda x: x[1].score, reverse=True)
    n_qual = len(hits)

    shortlist = hits[: settings["top_n"] * 3]
    dates, status, src = earnings_lookup([s for s, _ in shortlist]) if shortlist else ({}, {}, "")
    kept = [(s, st) for s, st in shortlist if status.get(s) != "in_window"]
    n_earn = len(shortlist) - len(kept)

    rows = []
    k = settings["atr_stop_mult"]
    for rank, (s, st) in enumerate(kept[: settings["top_n"]], start=1):
        r = frames[s].iloc[-1]
        sec = sector_of.get(s) or ""
        sinfo = ctx.get("sectors", {}).get(sec, {})
        etf = SECTOR_ETF.get(sec)
        etf_ret = sinfo.get("ret21", np.nan)
        stop_atr = st.entry - k * r.atr
        risk = st.entry - st.stop_structure
        rows.append(dict(
            rank=rank, symbol=s, name=name_of.get(s, ""), sector=sec, sector_etf=etf or "",
            sector_state=sinfo.get("state", "—"), sector_rs21=sinfo.get("rs21", np.nan),
            vs_sector21=(r.ret21 - etf_ret) if np.isfinite(etf_ret) else np.nan,
            score=round(float(st.score), 1), price=float(r.close), chg1=float(r.chg1),
            atr=float(r.atr), entry=float(st.entry), entry_note=st.entry_note,
            stop_structure=float(st.stop_structure), stop_note=st.stop_note,
            stop_atr=float(stop_atr),
            risk_struct_pct=100 * risk / st.entry if st.entry else np.nan,
            risk_atr_pct=100 * (st.entry - stop_atr) / st.entry if st.entry else np.nan,
            target_2r=float(st.entry + settings["outcome_target_r"] * risk) if risk > 0 else np.nan,
            target_hint=st.target_hint, target_note=st.target_note,
            notes=st.notes, rs_pct=ctx.get("rs_pct", {}).get(s, np.nan),
            earnings=dates.get(s), earnings_status=status.get(s, "unknown"),
            bar_date=frames[s].index[-1].date(),
        ))
    return dict(results=rows, qualified=n_qual, excluded_earnings=n_earn, errors=errors, earnings_source=src)
