"""Strategy template + auto-discovery.

To ADD a strategy: copy strategies/_template.py to strategies/<name>.py and edit it.
To REMOVE one: delete its file, or set `enabled = False` in it.
No other file needs to change — the picker, the Screening Definitions tab, card hold periods and
outcome scoring are all built from the attributes below.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Setup:
    score: float                  # 0–100, used to rank results
    entry: float                  # suggested entry trigger price
    entry_note: str               # e.g. "Buy-stop above today's high"
    stop_structure: float         # stop based on price structure
    stop_note: str                # e.g. "Below the 5-day swing low"
    notes: list[str] = field(default_factory=list)   # why it qualified / score drivers
    target_hint: float | None = None                 # optional strategy-specific objective
    target_note: str = ""
    tags: list[str] = field(default_factory=list)    # short badges shown on the card, e.g. "Ready"


class _SafeDict(dict):
    def __missing__(self, k):
        return "{" + k + "}"


class Strategy:
    # ── identity
    key = ""                 # short id, used in logs (keep stable once logging starts)
    name = ""                # shown in the picker
    description = ""         # one line shown under the picker
    enabled = True           # False hides it from the app
    order = 100              # position in the picker (lower = first)
    params: dict = {}
    universe: list | None = None   # e.g. ["sp500", "ndx100", "russell1000"]; None = app default

    # ── trade plan (drives cards, time stop and outcome scoring)
    timeframe = "Daily bars"
    hold = (2, 5)            # (min, max) trading days; max = time stop
    target_r = 2.0           # profit objective as a multiple of structure-stop risk
    atr_stop_mult = 1.5      # ATR stop = entry − k × ATR(14)
    stop_rule = ""           # plain-English structure stop, for the definitions tab
    earnings_exclude_days = None   # None = app default; earnings inside the hold cap the time stop
    regime_sensitive = True  # True = show the SPY-below-50-SMA warning prominently

    # ── documentation (rendered on the Screening Definitions tab; {param} placeholders are filled in)
    rules: list[str] = []
    score_parts: list[str] = []
    trigger = ""
    best_when = ""
    fails_when = ""
    data = "Daily price & volume"
    changed = ""             # date the rules/params last changed (YYYY-MM-DD)

    # ── extra data the strategy needs, fetched only for tickers that pass prefilter()
    needs: tuple = ()        # e.g. ("fundamentals",)

    def prefilter(self, d: pd.DataFrame, symbol: str, ctx: dict) -> bool:
        """Cheap price-only check run before any extra data is fetched."""
        return True

    def evaluate(self, d: pd.DataFrame, symbol: str, ctx: dict) -> Setup | None:
        """d = one ticker's daily bars with indicators (last row = today / latest session).
        Return a Setup if the ticker qualifies, else None."""
        raise NotImplementedError

    def describe(self) -> dict:
        p = _SafeDict(self.params)   # rules may use format specs, e.g. {near_high:.0%}

        def f(s):
            try:
                return s.format_map(p)
            except (ValueError, KeyError, IndexError):
                return s
        return dict(name=self.name, key=self.key, description=self.description, timeframe=self.timeframe,
                    hold=self.hold, target_r=self.target_r, atr_stop_mult=self.atr_stop_mult,
                    regime_sensitive=self.regime_sensitive, rules=[f(r) for r in self.rules],
                    score_parts=[f(s) for s in self.score_parts], trigger=f(self.trigger),
                    best_when=self.best_when, fails_when=self.fails_when, data=self.data,
                    changed=self.changed, params=dict(self.params), universe=self.universe,
                    earnings_exclude_days=self.earnings_exclude_days)


def discover() -> tuple[list[Strategy], list[str]]:
    import strategies as pkg
    found, errors = [], []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"strategies.{m.name}")
            mod = importlib.reload(mod)
            for obj in vars(mod).values():
                if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy \
                        and obj.__module__ == mod.__name__ and obj.enabled:
                    found.append(obj())
        except Exception as e:
            errors.append(f"{m.name}: {type(e).__name__}: {e}")
    found.sort(key=lambda s: (s.order, s.name))
    return found, errors
