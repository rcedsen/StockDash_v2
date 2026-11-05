"""Quality dip v2 backtest (see core/backtest.py for rules and input schemas).

With your own point-in-time data (recommended):
    python backtest_quality_dip.py --prices prices.csv --membership membership.csv \\
        --fundamentals fundamentals.csv --earnings earnings.csv --out bt/

Quick technical-only run on today's Russell 1000 list (survivorship bias and no fundamentals — labeled as such):
    python backtest_quality_dip.py --download 5 --out bt/
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from core import backtest
from core.settings import SECTOR_ETF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices")
    ap.add_argument("--membership")
    ap.add_argument("--fundamentals")
    ap.add_argument("--earnings")
    ap.add_argument("--download", type=int, help="years of Yahoo prices for today's Russell 1000 list")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--commission", type=float, default=0.005, help="$ per share per side")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", default="bt")
    a = ap.parse_args()

    universe = None
    if a.prices:
        prices = backtest.load_prices_csv(a.prices)
    elif a.download:
        from core.data import YFinanceProvider
        from core.universe import load_universe
        universe, note = load_universe(["russell1000"])
        print("Universe (today's list — survivorship bias):", note)
        syms = sorted(set(universe.symbol) | {"SPY"} | set(SECTOR_ETF.values()))
        end = dt.date.today() + dt.timedelta(days=1)
        prices = YFinanceProvider().history(syms, end - dt.timedelta(days=365 * a.download + 400), end)
    else:
        ap.error("give --prices or --download")
    frames = backtest.prepare(prices)
    rd = lambda p: pd.read_csv(p) if p else None
    kw = dict(universe=universe, membership=rd(a.membership), fundamentals=rd(a.fundamentals),
              earnings=rd(a.earnings), start=a.start, end=a.end, slip_bps=a.slippage_bps, commission_ps=a.commission)
    results = backtest.run_variants(frames, **kw)
    wf = backtest.walk_forward(results, folds=a.folds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, r in results.items():
        r["trades"].to_csv(out / f"trades_{n}.csv", index=False)
    wf.to_csv(out / "walk_forward.csv", index=False)
    (out / "summary.md").write_text(backtest.summary_markdown(results, wf))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
