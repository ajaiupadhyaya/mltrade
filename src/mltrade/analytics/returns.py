"""Return-series primitives shared across the analytics layer.

Pure, deterministic helpers operating on per-session simple returns.  All public
helpers round float outputs to a fixed precision so repeated calls on identical
inputs are byte-identical.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

PERIODS_PER_YEAR: float = 252.0
ROUND_PLACES: int = 10

# A return series may arrive as a plain sequence or a numpy array.
ReturnsLike = Sequence[float] | npt.NDArray[np.float64]


def _round(x: float) -> float:
    return round(x, ROUND_PLACES)


def as_array(returns: ReturnsLike) -> npt.NDArray[np.float64]:
    """Return ``returns`` as a 1-D float64 array (copy, finite-checked)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must be finite")
    return arr


def equity_to_returns(
    equity: Sequence[float], *, initial: float
) -> list[float]:
    """Convert an equity curve to per-period simple returns.

    The first return is measured against ``initial`` (the pre-curve NAV), so the
    output has the same length as ``equity``.
    """
    if initial <= 0:
        raise ValueError("initial equity must be positive")
    out: list[float] = []
    prev = float(initial)
    for value in equity:
        out.append(_round(float(value) / prev - 1.0) if prev > 0 else 0.0)
        prev = float(value)
    return out


def cumulative_growth(returns: Sequence[float], *, initial: float = 1.0) -> list[float]:
    """Compound ``returns`` into a growth curve starting from ``initial``."""
    out: list[float] = []
    level = float(initial)
    for r in returns:
        level *= 1.0 + float(r)
        out.append(_round(level))
    return out


def annualized_return(returns: ReturnsLike) -> float:
    """Geometric annualised return."""
    arr = as_array(returns)
    n = arr.size
    if n == 0:
        return 0.0
    growth = float(np.prod(1.0 + arr))
    if growth <= 0.0:
        return -1.0
    return _round(growth ** (PERIODS_PER_YEAR / n) - 1.0)


def annualized_volatility(returns: ReturnsLike) -> float:
    """Annualised standard deviation of returns (sample, ddof=1)."""
    arr = as_array(returns)
    if arr.size < 2:
        return 0.0
    return _round(float(np.std(arr, ddof=1)) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: ReturnsLike, *, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (arithmetic mean convention, rf per period)."""
    arr = as_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - risk_free
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0:
        return 0.0
    return _round(float(np.mean(excess)) / sd * np.sqrt(PERIODS_PER_YEAR))


def per_period_sharpe(returns: ReturnsLike, *, risk_free: float = 0.0) -> float:
    """Non-annualised (per-observation) Sharpe ratio — used by DSR/PSR."""
    arr = as_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - risk_free
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(excess)) / sd


def align_by_session[T](
    a_sessions: Sequence[T],
    a_returns: Sequence[float],
    b_sessions: Sequence[T],
    b_returns: Sequence[float],
) -> tuple[list[T], list[float], list[float]]:
    """Inner-join two return series on their session keys (ascending)."""
    a_map = dict(zip(a_sessions, a_returns, strict=True))
    b_map = dict(zip(b_sessions, b_returns, strict=True))
    common = sorted(set(a_map) & set(b_map), key=lambda s: str(s))
    return (
        common,
        [a_map[s] for s in common],
        [b_map[s] for s in common],
    )
