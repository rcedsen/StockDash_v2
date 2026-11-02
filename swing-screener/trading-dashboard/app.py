"""Swing Screener — Streamlit entry point (phase 1: screeners only).
Layout is mobile-first: one column, strategy picker on top, one Run button, result cards."""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import streamlit as st

from core import context as cctx
from core import earnings, outcomes, pipeline
from core.data import ET, get_provider, market_open, now_et
from core.settings import SECTOR_ETF, get_settings, secret
from core.storage import SheetLog
from core.strategy import discover
from core.universe import load_universe

st.set_page_config(page_title="Swing Screener", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 760px;}
.card-h {font-size: 1.05rem; font-weight: 700; margin-bottom: 0;}
.card-sub {color: var(--text-color); opacity: .65; font-size: .82rem; margin: 0 0 .35rem 0;}
.kv {font-size: .92rem; line-height: 1.45; margin: 0;}
.kv b {font-variant-numeric: tabular-nums;}
.pill {display:inline-block; padding: 0 .45rem; border-radius: 999px; font-size: .75rem;
       border: 1px solid rgba(128,128,128,.35); margin-left: .25rem;}
.up {color: #16a34a;} .dn {color: #dc2626;}
</style>""", unsafe_allow_html=True)

S = get_settings()
PROVIDER = get_provider(secret)
EXTRA = [S["regime_symbol"]] + list(SECTOR_ETF.values())


# ───────────────────────── cached loaders
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def c_universe(names: tuple):
    return load_universe(list(names))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def c_history(provider_name: str, symbols: tuple, start: dt.date, end: dt.date):
    return PROVIDER.history(list(symbols), start, end)


@st.cache_data(ttl=60, show_spinner=False)
def c_today(provider_name: str, symbols: tuple, minute_key: str):
    return PROVIDER.today(list(symbols))


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def c_vix():
    try:
        import yfinance as yf
        v = yf.download("^VIX", period="3mo", interval="1d", progress=False, auto_adjust=True)
        c = v["Close"]
        return (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def c_earnings(symbols: tuple, today: dt.date, window: int):
    return earnings.next_earnings(list(symbols), today, window, secret("FINNHUB_API_KEY"))


def expected_session(t: dt.datetime) -> dt.date:
    """Most recent session that should have data (ignores exchange holidays)."""
    d = t.date()
    if t.weekday() < 5 and (t.hour * 60 + t.minute) >= 570:
        return d
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def load_market():
    """Returns a dict with frames, context and data stamps. Reused across strategies for 60 s."""
    t = now_et()
    uni, uni_note = c_universe(tuple(S["universe"]))
    syms = tuple(sorted(set(uni.symbol) | set(EXTRA)))
    start = t.date() - dt.timedelta(days=S["history_days"])
    hist = c_history(PROVIDER.name, syms, start, t.date())
    today_bars, px_asof, vol_asof = c_today(PROVIDER.name, syms, t.strftime("%Y%m%d%H%M"))
    key = (PROVIDER.name, len(hist), str(px_asof)[:16], t.date())
    cache = st.session_state.get("market")
    if cache and cache["key"] == key:
        return cache
    live = market_open(t) and bool(today_bars)
    frames = pipeline.build_frames(hist, today_bars, vol_asof, live)
    ctx = cctx.build_context(frames, uni, c_vix(), S)
    spy = frames.get(S["regime_symbol"])
    last_bar = spy.index[-1].date() if spy is not None else None
    exp = expected_session(t)
    if last_bar is None or last_bar < exp:
        ctx["warnings"].insert(0, f"⚠️ Data may be stale: latest bar {last_bar}, expected {exp}"
                                  " (ignore on exchange holidays).")
    if live:
        mode = f"Live · price {px_asof.astimezone(ET):%H:%M} ET"
        if vol_asof and vol_asof != px_asof:
            mode += f" · volume {vol_asof.astimezone(ET):%H:%M} ET (projected to full day)"
        elif vol_asof:
            mode += " · volume projected to full day"
    else:
        mode = f"Close of {last_bar:%a %b %d}" if last_bar else "No data"
    m = dict(key=key, frames=frames, ctx=ctx, universe=uni, uni_note=uni_note, mode=mode, live=live,
             last_bar=last_bar, loaded=t, missing=len(set(uni.symbol) - set(frames)))
    st.session_state["market"] = m
    return m


# ───────────────────────── rendering helpers
def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def pct(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f'<span class="{"up" if x >= 0 else "dn"}">{x * 100:+.1f}%</span>'


def render_card(r: dict):
    earn = r["earnings"]
    if earn:
        earn_txt = f"{earn:%b %d}"
    elif r["earnings_status"] == "clear":
        earn_txt = f"none within {S['earnings_window_days']}d"
    else:
        earn_txt = "unknown — check before entry"
    sector_pill = f'<span class="pill">{r["sector_etf"]} {r["sector_state"]}</span>' if r["sector_etf"] else ""
    with st.container(border=True):
        st.markdown(
            f'<p class="card-h">{r["rank"]} · {r["symbol"]} <span class="pill">score {r["score"]:.0f}</span></p>'
            f'<p class="card-sub">{r["name"]} · ${fmt(r["price"])} {pct(r["chg1"])} {sector_pill}</p>'
            f'<p class="kv">Entry <b>{fmt(r["entry"])}</b> · {r["entry_note"]}</p>'
            f'<p class="kv">Stop <b>{fmt(r["stop_structure"])}</b> structure (−{fmt(r["risk_struct_pct"], 1)}%)'
            f' · <b>{fmt(r["stop_atr"])}</b> {S["atr_stop_mult"]}×ATR (−{fmt(r["risk_atr_pct"], 1)}%)</p>'
            f'<p class="kv">2R <b>{fmt(r["target_2r"])}</b>'
            + (f' · {r["target_note"]} <b>{fmt(r["target_hint"])}</b>' if r["target_hint"] else "")
            + f'</p><p class="kv">Earnings: {earn_txt}</p>',
            unsafe_allow_html=True)
        with st.expander("Why it qualified"):
            for n in r["notes"]:
                st.markdown(f"- {n}")
            st.markdown(f"- {r['stop_note']}")
            vs = r["vs_sector21"]
            if r["sector"]:
                st.markdown(f"- Sector: {r['sector']} ({r['sector_etf']}, {r['sector_state']}, "
                            f"1M vs SPY {fmt(100 * r['sector_rs21'], 1)}%)")
            if vs is not None and np.isfinite(vs):
                st.markdown(f"- Stock vs sector, 1M: {vs * 100:+.1f}%")
            st.caption(f"Bar date {r['bar_date']} · ATR {fmt(r['atr'])}")


def render_context(m, strategy):
    ctx = m["ctx"]
    spy, vix = ctx.get("spy", {}), ctx.get("vix")
    bits = []
    if spy:
        bits.append(f"SPY {fmt(spy['close'])} {pct(spy['chg1'])} "
                    f"{'above' if ctx['regime_ok'] else 'below'} 50 SMA")
    if np.isfinite(ctx.get("breadth", np.nan)):
        bits.append(f"Breadth {ctx['breadth']:.0f}%")
    if vix:
        bits.append(f"VIX {vix['level']:.1f} ({vix['chg5']:+.1f} 5d)")
    st.markdown(f'<p class="card-sub">{" · ".join(bits)}</p>', unsafe_allow_html=True)
    for w in ctx["warnings"]:
        regime = "50-day SMA" in w
        if regime and strategy.regime_sensitive:
            st.warning(w)
        elif regime:
            st.caption(w)
        else:
            st.warning(w)


# ───────────────────────── app
strategies, strat_errors = discover()
log = SheetLog(secret)

st.markdown("### 📈 Swing Screener")
t = now_et()
st.caption(f"{t:%a %b %d · %H:%M} ET · market {'open' if market_open(t) else 'closed'} · data: {PROVIDER.name}")

tab_screen, tab_track, tab_status = st.tabs(["Screen", "Tracker", "Status"])

with tab_screen:
    if not strategies:
        st.error("No strategies found in /strategies.")
        st.stop()
    names = [s.name for s in strategies]
    pick = st.segmented_control("Strategy", names, default=st.session_state.get("pick", names[0]),
                                key="pick_ctl", label_visibility="collapsed")
    pick = pick or names[0]
    st.session_state["pick"] = pick
    strat = next(s for s in strategies if s.name == pick)
    st.caption(strat.description)

    if st.button(f"Run {strat.name}", type="primary", width="stretch"):
        t0 = time.time()
        with st.spinner("Loading market data…" if "market" not in st.session_state else "Screening…"):
            m = load_market()
            today = now_et().date()
            res = pipeline.run_strategy(
                strat, m["frames"], m["universe"], m["ctx"], S,
                lambda syms: c_earnings(tuple(syms), today, S["earnings_window_days"]))
        res.update(strategy=strat.key, name=strat.name, mode=m["mode"], secs=time.time() - t0,
                   ran=now_et())
        st.session_state.setdefault("results", {})[strat.key] = res
        if log.on:
            run_id = f"{res['ran']:%Y%m%d-%H%M%S}-{strat.key}"
            sigs = [dict(r, run_id=run_id, run_time_et=f"{res['ran']:%Y-%m-%d %H:%M}", strategy=strat.key,
                         data_mode="live" if m["live"] else "close", regime_ok=m["ctx"].get("regime_ok"),
                         status="open") for r in res["results"]]
            try:
                log.log_run(dict(run_id=run_id, run_time_et=f"{res['ran']:%Y-%m-%d %H:%M}", strategy=strat.key,
                                 provider=PROVIDER.name, data_mode=m["mode"], data_asof=str(m["last_bar"]),
                                 universe_n=len(m["frames"]), qualified=res["qualified"],
                                 excluded_earnings=res["excluded_earnings"], shown=len(res["results"]),
                                 warnings=m["ctx"]["warnings"]), sigs)
                res["logged"] = True
            except Exception as e:
                res["log_error"] = f"{type(e).__name__}: {e}"

    res = st.session_state.get("results", {}).get(strat.key)
    if res:
        m = st.session_state["market"]
        st.markdown(f"**{res['mode']}**")
        render_context(m, strat)
        st.caption(f"{res['qualified']} qualified · {res['excluded_earnings']} dropped for earnings ≤"
                   f"{S['earnings_window_days']}d ({res['earnings_source'] or '—'}) · showing top "
                   f"{len(res['results'])} · ran {res['ran']:%H:%M} ET in {res['secs']:.0f}s"
                   + (" · logged" if res.get("logged") else ""))
        if res.get("log_error"):
            st.caption(f"Log failed: {res['log_error']}")
        if not res["results"]:
            st.info("No tickers meet this strategy's rules right now.")
        for r in res["results"]:
            render_card(r)
    else:
        st.info("Pick a strategy and tap Run. Nothing is screened until you do.")

with tab_track:
    if not log.on:
        st.info(log.error)
    else:
        st.caption(f"Signals are scored after {S['outcome_hold_days']} trading days: fill on entry trigger, "
                   f"structure stop, {S['outcome_target_r']:g}R target, otherwise exit at the close.")
        if st.button("Update outcomes", width="stretch"):
            with st.spinner("Scoring open signals…"):
                sig = log.read_signals()
                open_ = sig[sig.status.isin(["", "open"])]
                if open_.empty:
                    st.info("No open signals.")
                else:
                    start = pd.to_datetime(open_.bar_date).min().date() - dt.timedelta(days=3)
                    hist = PROVIDER.history(sorted(set(open_.symbol)), start, now_et().date())
                    sig, n = outcomes.update_outcomes(sig, hist, S["outcome_hold_days"])
                    log.write_signals(sig)
                    st.success(f"Scored {n} signal(s).")
        try:
            sig = log.read_signals()
            summ = outcomes.summarize(sig)
            if not summ.empty:
                st.markdown("**By strategy**")
                st.dataframe(summ, hide_index=True, width="stretch")
                st.markdown("**Recent signals**")
                cols = ["run_time_et", "strategy", "symbol", "score", "entry", "stop_structure", "outcome", "outcome_r"]
                st.dataframe(sig[cols].tail(30).iloc[::-1], hide_index=True, width="stretch")
            else:
                st.caption("No signals logged yet.")
        except Exception as e:
            st.error(f"Could not read the log: {e}")

with tab_status:
    keys = {"Alpaca": bool(secret("ALPACA_API_KEY")), "Finnhub": bool(secret("FINNHUB_API_KEY")),
            "Google Sheets": log.on}
    st.markdown("**Connections**")
    for k, v in keys.items():
        st.markdown(f"- {k}: {'✅ connected' if v else '— not set'}")
    if not keys["Alpaca"]:
        st.caption("Without Alpaca keys the app uses Yahoo data (no key, unofficial, may rate-limit).")
    m = st.session_state.get("market")
    if m:
        st.markdown("**Data**")
        st.markdown(f"- Universe: {m['uni_note']}\n- Loaded: {len(m['frames'])} symbols "
                    f"({m['missing']} universe tickers missing)\n- Latest bar: {m['last_bar']}\n- Mode: {m['mode']}")
    st.markdown("**Strategies loaded**")
    for s in strategies:
        st.markdown(f"- {s.name} (`{s.key}`) — params: `{s.params}`")
    for e in strat_errors:
        st.error(f"Strategy failed to load — {e}")
    st.markdown("**Settings**")
    st.json(S, expanded=False)
    if st.button("Clear data cache", width="stretch"):
        st.cache_data.clear()
        st.session_state.pop("market", None)
        st.session_state.pop("results", None)
        st.rerun()
