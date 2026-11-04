"""Shared screening pipeline:
bars → indicators → prefilter → (extra data for survivors) → strategy → rank → earnings rules → top N → card fields.
Every strategy goes through this, so warnings, earnings rules, hold periods and card fields apply to all."""
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


def bday_add(d: dt.date, n: int) -> dt.date:
    return pd.Timestamp(np.busday_offset(np.datetime64(d), n, roll="forward")).date()


def run_strategy(strategy: Strategy, frames: dict[str, pd.DataFrame], universe: pd.DataFrame,
                 ctx: dict, settings: dict, earnings_lookup, lookups: dict | None = None,
                 today: dt.date | None = None) -> dict:
    """earnings_lookup(symbols) -> (dates, known, source).
    lookups = {"fundamentals": fn(symbols) -> {sym: dict}, ...} for strategies that declare `needs`."""
    today = today or dt.date.today()
    lookups = lookups or {}
    sector_of = dict(zip(universe.symbol, universe.sector))
    name_of = dict(zip(universe.symbol, universe.name))

    # 1. cheap price-only prefilter
    cands, errors = [], 0
    for s in universe.symbol:
        d = frames.get(s)
        if d is None:
            continue
        try:
            if strategy.prefilter(d, s, ctx):
                cands.append(s)
        except Exception:
            errors += 1

    # 2. extra data only for survivors
    sctx = dict(ctx)
    extra_note = ""
    for need in strategy.needs:
        fn = lookups.get(need)
        sctx[need] = fn(cands) if (fn and cands) else {}
        extra_note = f"{len(cands)} passed the price prefilter → {need} checked"

    # 3. full rules
    hits = []
    for s in cands:
        try:
            st = strategy.evaluate(frames[s], s, sctx)
        except Exception:
            errors += 1
            continue
        if st is not None and np.isfinite(st.score):
            hits.append((s, st))
    hits.sort(key=lambda x: x[1].score, reverse=True)
    n_qual = len(hits)

    # 4. earnings: exclude reports inside the exclusion window; cap the time stop before later ones
    excl = strategy.earnings_exclude_days or settings["earnings_window_days"]
    shortlist = hits[: settings["top_n"] * 3]
    dates, known, src = earnings_lookup([s for s, _ in shortlist]) if shortlist else ({}, set(), "")
    excl_end = today + dt.timedelta(days=excl)
    kept = [(s, st) for s, st in shortlist if not (dates.get(s) and dates[s] <= excl_end)]
    n_earn = len(shortlist) - len(kept)

    rows = []
    k = strategy.atr_stop_mult or settings["atr_stop_mult"]
    for rank, (s, st) in enumerate(kept[: settings["top_n"]], start=1):
        f = frames[s]
        r = f.iloc[-1]
        bar_date = f.index[-1].date()
        sec = sector_of.get(s) or ""
        sinfo = ctx.get("sectors", {}).get(sec, {})
        etf = SECTOR_ETF.get(sec)
        etf_ret = sinfo.get("ret21", np.nan)
        stop_atr = st.entry - k * r.atr
        risk = st.entry - st.stop_structure

        hold_min, hold_max = strategy.hold
        time_stop = bday_add(bar_date, hold_max)
        earn = dates.get(s)
        earn_in_hold = bool(earn and earn <= time_stop)
        if earn_in_hold:
            time_stop = pd.Timestamp(np.busday_offset(np.datetime64(earn), -1, roll="backward")).date()
        earn_status = "date" if earn else ("clear" if s in known else "unknown")

        rows.append(dict(
            rank=rank, symbol=s, name=name_of.get(s, ""), sector=sec, sector_etf=etf or "",
            sector_state=sinfo.get("state", "—"), sector_rs21=sinfo.get("rs21", np.nan),
            vs_sector21=(r.ret21 - etf_ret) if np.isfinite(etf_ret) else np.nan,
            score=round(float(st.score), 1), price=float(r.close), chg1=float(r.chg1),
            atr=float(r.atr), entry=float(st.entry), entry_note=st.entry_note,
            stop_structure=float(st.stop_structure), stop_note=st.stop_note,
            stop_atr=float(stop_atr), atr_mult=k,
            risk_struct_pct=100 * risk / st.entry if st.entry else np.nan,
            risk_atr_pct=100 * (st.entry - stop_atr) / st.entry if st.entry else np.nan,
            target_r=strategy.target_r,
            target=float(st.entry + strategy.target_r * risk) if risk > 0 else np.nan,
            target_hint=st.target_hint, target_note=st.target_note,
            hold_min=hold_min, hold_max=hold_max, time_stop=time_stop, earn_in_hold=earn_in_hold,
            notes=st.notes, tags=st.tags, rs_pct=ctx.get("rs_pct", {}).get(s, np.nan),
            earnings=earn, earnings_status=earn_status, bar_date=bar_date,
        ))
    return dict(results=rows, qualified=n_qual, excluded_earnings=n_earn, errors=errors,
                earnings_source=src, prefiltered=len(cands), extra_note=extra_note, earnings_excl_days=excl)
