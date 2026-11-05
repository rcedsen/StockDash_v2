"""Run the Quality dip v2 screen on the full Russell 1000 from the command line.

    python run_quality_dip.py --out results/            # Yahoo data, no keys needed
    FINNHUB_API_KEY=... python run_quality_dip.py       # adds Finnhub earnings calendar + ROI fallback

Writes results/quality_dip_v2_<date>.csv (every constituent) and .md (ranked + exclusion report).
Same engine as the app (core/qd2.py)."""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from core import context, earnings, fundamentals, pipeline, qd2
from core.data import YFinanceProvider, now_et
from core.settings import DEFAULTS, SECTOR_ETF
from core.universe import load_universe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--universe", default="russell1000")
    args = ap.parse_args()
    key = os.environ.get("FINNHUB_API_KEY")
    today = now_et().date()

    uni, note = load_universe(args.universe.split(","))
    print("Universe:", note)
    syms = sorted(set(uni.symbol) | {"SPY"} | set(SECTOR_ETF.values()))
    prov = YFinanceProvider()
    hist = prov.history(syms, today - dt.timedelta(days=DEFAULTS["history_days"]), today + dt.timedelta(days=1))
    print(f"Prices: {len(hist)} of {len(syms)} symbols")
    frames = pipeline.build_frames(hist, {}, None, live=False)
    ctx = context.build_context(frames, uni, None, DEFAULTS)
    res = qd2.run_screen(frames, uni, ctx, lambda s: fundamentals.fetch_many(s, key),
                         lambda s: earnings.next_earnings(s, today, 47, key), today)
    spy = frames.get("SPY")
    day = spy.index[-1].date() if spy is not None else today
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qd2.to_csv_rows(res["rows"]).to_csv(out / f"quality_dip_v2_{day}.csv", index=False)
    (out / f"quality_dip_v2_{day}.md").write_text(qd2.report_markdown(res, ctx))
    print(res["stats"])
    print(f"Wrote {out}/quality_dip_v2_{day}.csv and .md (data through {day})")


if __name__ == "__main__":
    main()
