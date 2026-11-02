"""Strategy template + auto-discovery.

To ADD a strategy: copy strategies/_template.py to strategies/<name>.py and edit it.
To REMOVE one: delete its file, or set `enabled = False` in it.
No other file needs to change — the app builds its strategy picker from this registry.
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


class Strategy:
    key = ""                 # short id, used in logs (keep stable once logging starts)
    name = ""                # shown in the picker
    description = ""         # one line shown under the picker
    enabled = True           # False hides it from the app
    regime_sensitive = True  # True = show the SPY-below-50-SMA warning prominently
    order = 100              # position in the picker (lower = first)
    params: dict = {}

    def evaluate(self, d: pd.DataFrame, symbol: str, ctx: dict) -> Setup | None:
        """d = one ticker's daily bars with indicators (last row = today / latest session).
        Return a Setup if the ticker qualifies, else None."""
        raise NotImplementedError


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
