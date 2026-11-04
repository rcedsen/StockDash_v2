"""Swing Screener — Streamlit entry point (phase 1: screeners only).
Layout is mobile-first: one column, strategy picker on top, one Run button, result cards."""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import streamlit as st

from core import context as cctx
from core import earnings, fundamentals, outcomes, pipeline
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
.tag-ready {background: rgba(34,197,94,.18); border-color: rgba(34,197,94,.6);}
.def-h {font-weight: 700; margin: .6rem 0 .1rem 0; font-size: .92rem;}
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
def c_earnings(symbols: tuple, today: dt.date, horizon: int):
    return earnings.next_earnings(list(symbols), today, horizon, secret("FINNHUB_API_KEY"))


FUND_TTL = 7 * 86400


@st.cache_resource
def fund_store() -> dict:
    """{symbol: (fetched_epoch, metrics)} shared by all sessions until the app restarts."""
    return {}


def get_fundamentals(symbols: list[str]) -> dict:
    store, now = fund_store(), time.time()
    need = [s for s in symbols if s not in store or now - store[s][0] > FUND_TTL]
    if need:
        for s, v in fundamentals.fetch_many(need, secret("FINNHUB_API_KEY")).items():
            store[s] = (now, v)
    return {s: store[s][1] for s in symbols if s in store}


def expected_session(t: dt.datetime) -> dt.date:
    """Most recent session that should have data (ignores exchange holidays)."""
    d = t.date()
    if t.weekday() < 5 and (t.hour * 60 + t.minute) >= 570:
        return d
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def uni_key(strategy) -> tuple:
    return tuple(strategy.universe or S["universe"])


def load_market(names: tuple):
    """Frames, context and data stamps for one universe. Reused across strategies for 60 s."""
    t = now_et()
    uni, uni_note = c_universe(names)
    syms = tuple(sorted(set(uni.symbol) | set(EXTRA)))
    start = t.date() - dt.timedelta(days=S["history_days"])
    hist = c_history(PROVIDER.name, syms, start, t.date())
    today_bars, px_asof, vol_asof = c_today(PROVIDER.name, syms, t.strftime("%Y%m%d%H%M"))
    key = (PROVIDER.name, len(hist), str(px_asof)[:16], t.date())
    markets = st.session_state.setdefault("markets", {})
    cache = markets.get(names)
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
             last_bar=last_bar, loaded=t, missing=len(set(uni.symbol) - set(frames)), names=names)
    markets[names] = m
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
    if earn and r["earn_in_hold"]:
        earn_txt = f"⚠ {earn:%b %d} — inside the hold, time stop moved before it"
    elif earn:
        earn_txt = f"{earn:%b %d}"
    elif r["earnings_status"] == "clear":
        earn_txt = "none inside the hold"
    else:
        earn_txt = "unknown — check before entry"
    sector_pill = f'<span class="pill">{r["sector_etf"]} {r["sector_state"]}</span>' if r["sector_etf"] else ""
    tag_pills = "".join(f'<span class="pill {"tag-ready" if t == "Ready" else ""}">{t}</span>' for t in r["tags"])
    with st.container(border=True):
        st.markdown(
            f'<p class="card-h">{r["rank"]} · {r["symbol"]} <span class="pill">score {r["score"]:.0f}</span>{tag_pills}</p>'
            f'<p class="card-sub">{r["name"]} · ${fmt(r["price"])} {pct(r["chg1"])} {sector_pill}</p>'
            f'<p class="kv">Entry <b>{fmt(r["entry"])}</b> · {r["entry_note"]}</p>'
            f'<p class="kv">Stop <b>{fmt(r["stop_structure"])}</b> structure (−{fmt(r["risk_struct_pct"], 1)}%)'
            f' · <b>{fmt(r["stop_atr"])}</b> {r["atr_mult"]:g}×ATR (−{fmt(r["risk_atr_pct"], 1)}%)</p>'
            f'<p class="kv">Target <b>{fmt(r["target"])}</b> ({r["target_r"]:g}R)'
            + (f' · {r["target_note"]} <b>{fmt(r["target_hint"])}</b>' if r["target_hint"] else "")
            + f'</p><p class="kv">Hold {r["hold_min"]}–{r["hold_max"]} days · time stop <b>{r["time_stop"]:%a %b %d}</b></p>'
            f'<p class="kv">Earnings: {earn_txt}</p>',
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


GLOSSARY = {
    "ATR(14)": "Average True Range over 14 sessions (Wilder smoothing) — the stock's typical daily move. "
               "Stops, base width and stretch are measured in ATRs so they scale with volatility.",
    "SMA / EMA": "Simple / exponential moving average of daily closes (10, 20, 50, 200 sessions).",
    "RSI(2) / RSI(14)": "Wilder RSI over 2 or 14 sessions. RSI(2) < 10 = short-term washed out.",
    "RS percentile": "Rank of (50% × 1-month + 50% × 3-month return minus SPY's) across the universe; "
                     "90 = stronger than 90% of the universe.",
    "Base width": "High–low range of the 20 sessions before today, in ATRs. Smaller = tighter base.",
    "Projected volume": "During market hours, today's volume so far ÷ the typical share of a day's volume "
                        "traded by this time, so it can be compared with the 50-day average.",
    "R": "Risk unit = entry − structure stop. 2R target = entry + 2 × that distance.",
    "Time stop": "Exit date if neither stop nor target has hit: the strategy's max hold in trading days, "
                 "moved to the session before earnings when a report falls inside the hold.",
    "Breadth": "% of the universe closing above its 50-day SMA.",
    "Sector state": "Leading = sector ETF above its 20 & 50 SMA and beating SPY over 1 month; "
                    "Lagging = below its 50 SMA and trailing SPY; otherwise Neutral.",
    "PEG": "Forward P/E ÷ forward EPS growth (in %). Below 1.5 = cheap for its growth.",
    "FCF yield / margin": "Trailing free cash flow ÷ market cap / ÷ revenue.",
}

CARD_LEGEND = [
    "**Header** — rank, ticker, score (0–100, ranking only), badges such as Ready / Setting up",
    "**Sub-line** — name, last price, today's % change, sector ETF and its state",
    "**Entry** — the trigger price and how to use it (usually a buy-stop above today's high)",
    "**Stop** — structure stop and ATR stop, each with % risk from entry; use whichever fits your sizing",
    "**Target** — the strategy's R-multiple target, plus a chart objective when the strategy has one",
    "**Hold / time stop** — the strategy's expected hold and the latest exit date",
    "**Earnings** — next report date; ⚠ when it falls inside the hold",
    "**Why it qualified** — the values behind each rule and score component",
]


def render_definitions(strats):
    st.caption("Generated from the strategy files — when a parameter changes, this page changes with it.")
    st.markdown("**At a glance**")
    st.dataframe(pd.DataFrame([dict(Strategy=s.name, Hold=f"{s.hold[0]}–{s.hold[1]}d", Target=f"{s.target_r:g}R",
                                    **{"ATR stop": f"{s.atr_stop_mult:g}×"},
                                    Regime="warn" if s.regime_sensitive else "info",
                                    Universe="R1000" if "russell1000" in (s.universe or S["universe"]) else "S&P+NDX",
                                    Data="price + fund." if "fundamentals" in s.needs else "price")
                               for s in strats]), hide_index=True, width="stretch")
    for s in strats:
        d = s.describe()
        with st.expander(f"{d['name']} · {d['hold'][0]}–{d['hold'][1]} days"):
            st.markdown(f"{d['description']}")
            st.markdown('<p class="def-h">Timeframe & plan</p>', unsafe_allow_html=True)
            excl = d["earnings_exclude_days"] or S["earnings_window_days"]
            st.markdown(
                f"- Universe: {' + '.join(d['universe'] or S['universe'])}\n"
                f"- Timeframe: {d['timeframe']}\n- Hold: {d['hold'][0]}–{d['hold'][1]} trading days "
                f"(time stop at {d['hold'][1]})\n- Target: {d['target_r']:g}R\n"
                f"- Stops: structure — {s.stop_rule or 'see card'}; ATR — entry − {d['atr_stop_mult']:g} × ATR(14)\n"
                f"- Earnings: excluded if reporting within {excl} days; later reports inside the hold move the time stop\n"
                f"- Regime warning (SPY < 50 SMA): {'shown prominently' if d['regime_sensitive'] else 'informational only'}")
            st.markdown('<p class="def-h">Rules (all must pass)</p>', unsafe_allow_html=True)
            st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(d["rules"], 1)) or "—")
            if d["trigger"]:
                st.markdown('<p class="def-h">Entry trigger</p>', unsafe_allow_html=True)
                st.markdown(d["trigger"])
            st.markdown('<p class="def-h">Score (ranking)</p>', unsafe_allow_html=True)
            st.markdown("\n".join(f"- {x}" for x in d["score_parts"]) or "—")
            if d["best_when"] or d["fails_when"]:
                st.markdown('<p class="def-h">Market fit</p>', unsafe_allow_html=True)
                st.markdown(f"- Works best: {d['best_when']}\n- Struggles: {d['fails_when']}")
            st.markdown('<p class="def-h">Data</p>', unsafe_allow_html=True)
            st.markdown(f"{d['data']}  \nParameters: `{d['params']}`  \nLast changed: {d['changed'] or '—'}")

    st.markdown("**Applies to every strategy**")
    with st.expander("Universe & data"):
        loaded = st.session_state.get("markets", {})
        st.markdown(f"- Default universe: {' + '.join(S['universe'])} (a strategy can set its own — see its section)"
                    + "".join(f"\n- Loaded {' + '.join(k)}: {v['uni_note']}" for k, v in loaded.items()) +
                    f"\n- Price data: {PROVIDER.name}; daily bars, ~{S['history_days']} calendar days of history"
                    "\n- During market hours today's partial bar is added and volume is projected to a full day"
                    "\n- Every result shows whether it ran on live data or the last close, with timestamps")
    with st.expander("Exclusions, ranking & warnings"):
        st.markdown(
            f"- Results ranked by score; top {S['top_n']} shown\n"
            f"- Earnings within each strategy's exclusion window (default {S['earnings_window_days']} days) → removed\n"
            "- Tickers without enough history for a rule's indicators are skipped\n"
            "- Warnings: SPY below its 50 SMA · breadth < 40% · VIX ≥ "
            f"{S['vix_warn_level']:g} or up 4+ pts in 5 days · latest bar older than the expected session")
    with st.expander("Glossary"):
        st.markdown("\n".join(f"- **{k}** — {v}" for k, v in GLOSSARY.items()))
    with st.expander("Result card legend"):
        st.markdown("\n".join(f"- {x}" for x in CARD_LEGEND))
    with st.expander("Outcome scoring (Tracker)"):
        st.markdown("\n".join(f"- {x}" for x in outcomes.RULES))
    with st.expander("Change log"):
        rows = sorted(((s.changed or "—", s.name) for s in strats), reverse=True)
        st.markdown("\n".join(f"- {c} · {n}" for c, n in rows))


# ───────────────────────── app
strategies, strat_errors = discover()
log = SheetLog(secret)

st.markdown("### 📈 Swing Screener")
t = now_et()
st.caption(f"{t:%a %b %d · %H:%M} ET · market {'open' if market_open(t) else 'closed'} · data: {PROVIDER.name}")

# earnings are looked up far enough ahead to cover the longest hold (trading days → calendar days)
EARN_HORIZON = max([S["earnings_window_days"]] + [int(s.hold[1] * 1.45) + 4 for s in strategies])

tab_screen, tab_defs, tab_track, tab_status = st.tabs(["Screen", "Screening Definitions", "Tracker", "Status"])

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
    st.caption(f"{strat.description} · hold {strat.hold[0]}–{strat.hold[1]} days · {strat.target_r:g}R target")

    if st.button(f"Run {strat.name}", type="primary", width="stretch"):
        t0 = time.time()
        ukey = uni_key(strat)
        first = ukey not in st.session_state.get("markets", {})
        msg = f"Loading {' + '.join(ukey)} market data…" if first else "Screening…"
        if strat.needs:
            msg += " (fundamentals are fetched on the first run of the week — up to a minute)"
        with st.spinner(msg):
            m = load_market(ukey)
            today = now_et().date()
            res = pipeline.run_strategy(
                strat, m["frames"], m["universe"], m["ctx"], S,
                lambda syms: c_earnings(tuple(sorted(syms)), today, EARN_HORIZON),
                lookups={"fundamentals": get_fundamentals}, today=today)
        res.update(strategy=strat.key, name=strat.name, mode=m["mode"], secs=time.time() - t0, uni=ukey,
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
        m = st.session_state["markets"][res["uni"]]
        st.markdown(f"**{res['mode']}** · {len(m['frames'])} symbols ({' + '.join(res['uni'])})")
        render_context(m, strat)
        st.caption((f"{res['extra_note']} · " if res.get("extra_note") else "")
                   + f"{res['qualified']} qualified · {res['excluded_earnings']} dropped for earnings ≤"
                   f"{res['earnings_excl_days']}d ({res['earnings_source'] or '—'}) · showing top "
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

with tab_defs:
    render_definitions(strategies)

with tab_track:
    if not log.on:
        st.info(log.error)
    else:
        st.caption("Each signal is scored over its own strategy's max hold and target — rules on the "
                   "Screening Definitions tab.")
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
    for k, m in st.session_state.get("markets", {}).items():
        st.markdown(f"**Data · {' + '.join(k)}**")
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
        st.session_state.pop("markets", None)
        st.session_state.pop("results", None)
        st.rerun()
