"""Small helpers shared by strategies (files starting with _ are not loaded as strategies)."""
import math


def scale(x, lo, hi) -> float:
    """Map x linearly onto 0..1 between lo and hi (hi may be < lo for 'lower is better')."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def ok(*vals) -> bool:
    """True if every value is a finite number."""
    try:
        return all(v is not None and math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False
