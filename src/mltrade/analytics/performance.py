"""Performance and risk statistics computed from a per-session return series.

Institutional-grade summary: geometric/annualised return and volatility, Sharpe,
Sortino, Calmar, drawdown depth/duration/recovery, return-distribution moments,
and historical / Cornish-Fisher tail risk (VaR, CVaR).  Also exposes the chart
series the dashboard renders (drawdown/underwater curve, rolling Sharpe, return
histogram, monthly-return table).

All functions are pure and deterministic; float outputs are rounded for
byte-stable reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from scipy import stats  # type: ignore[import-untyped]

from mltrade.analytics.returns import (
    PERIODS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    as_array,
    sharpe_ratio,
)

ROUND_PLACES = 10


def _round(x: float) -> float:
    return round(x, ROUND_PLACES)


class PerformanceStats(BaseModel):
    """Scalar performance and risk summary for one return series."""

    model_config = ConfigDict(frozen=True)

    n_sessions: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    time_to_recovery: int | None
    skewness: float
    excess_kurtosis: float
    best_day: float
    worst_day: float
    positive_fraction: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    cornish_fisher_var_95: float


def sortino_ratio(
    returns: Sequence[float], *, risk_free: float = 0.0, target: float = 0.0
) -> float:
    """Annualised Sortino ratio (downside deviation about ``target``)."""
    arr = as_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - risk_free
    downside = np.minimum(arr - target, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0.0:
        return 0.0
    return _round(float(np.mean(excess)) / dd * np.sqrt(PERIODS_PER_YEAR))


def drawdown_series(returns: Sequence[float]) -> list[float]:
    """Per-session drawdown from the running peak (<= 0)."""
    arr = as_array(returns)
    if arr.size == 0:
        return []
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return [_round(float(x)) for x in dd]


def _drawdown_array(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def max_drawdown(returns: Sequence[float]) -> float:
    arr = as_array(returns)
    if arr.size == 0:
        return 0.0
    return _round(float(np.min(_drawdown_array(arr))))


def max_drawdown_duration(returns: Sequence[float]) -> int:
    """Longest underwater stretch in sessions (peak → return to that peak)."""
    arr = as_array(returns)
    if arr.size == 0:
        return 0
    dd = _drawdown_array(arr)
    longest = 0
    current = 0
    for value in dd:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def time_to_recovery(returns: Sequence[float]) -> int | None:
    """Sessions from the deepest trough until equity reclaims its prior peak.

    Returns ``None`` if the maximum-drawdown episode never fully recovered
    within the sample.
    """
    arr = as_array(returns)
    if arr.size == 0:
        return None
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    trough = int(np.argmin(dd))
    peak_value = peak[trough]
    for i in range(trough + 1, arr.size):
        if equity[i] >= peak_value:
            return i - trough
    return None


def historical_var(returns: Sequence[float], *, level: float = 0.95) -> float:
    """Historical Value-at-Risk as a positive loss at ``level`` confidence."""
    arr = as_array(returns)
    if arr.size == 0:
        return 0.0
    q = float(np.quantile(arr, 1.0 - level))
    return _round(max(0.0, -q))


def historical_cvar(returns: Sequence[float], *, level: float = 0.95) -> float:
    """Historical Conditional VaR (expected shortfall) as a positive loss."""
    arr = as_array(returns)
    if arr.size == 0:
        return 0.0
    threshold = float(np.quantile(arr, 1.0 - level))
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return _round(max(0.0, -threshold))
    return _round(max(0.0, -float(np.mean(tail))))


def cornish_fisher_var(returns: Sequence[float], *, level: float = 0.95) -> float:
    """Modified (Cornish-Fisher) VaR adjusting the normal quantile for skew/kurt."""
    arr = as_array(returns)
    if arr.size < 4:
        return historical_var(returns, level=level)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return 0.0
    s = float(stats.skew(arr, bias=False))
    k = float(stats.kurtosis(arr, fisher=True, bias=False))  # excess kurtosis
    z = float(stats.norm.ppf(1.0 - level))
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    )
    return _round(max(0.0, -(mean + z_cf * sd)))


def rolling_sharpe(
    returns: Sequence[float], *, window: int = 126
) -> list[float | None]:
    """Trailing annualised Sharpe over a rolling ``window`` (None until filled)."""
    arr = as_array(returns)
    out: list[float | None] = []
    for i in range(arr.size):
        if i + 1 < window:
            out.append(None)
            continue
        win = arr[i + 1 - window : i + 1]
        out.append(sharpe_ratio(win))
    return out


def return_histogram(
    returns: Sequence[float], *, bins: int = 41
) -> tuple[list[float], list[int]]:
    """Histogram of returns: (bin centres, counts)."""
    arr = as_array(returns)
    if arr.size == 0:
        return [], []
    counts, edges = np.histogram(arr, bins=bins)
    centres = [(float(edges[i]) + float(edges[i + 1])) / 2.0 for i in range(bins)]
    return [_round(c) for c in centres], [int(x) for x in counts]


def monthly_returns(
    dates: Sequence[date], returns: Sequence[float]
) -> list[dict[str, float | int]]:
    """Compound daily returns into calendar-month returns."""
    arr = as_array(returns)
    if len(dates) != arr.size:
        raise ValueError("dates and returns must be the same length")
    buckets: dict[tuple[int, int], float] = {}
    for d, r in zip(dates, arr, strict=True):
        key = (d.year, d.month)
        buckets[key] = (1.0 + buckets.get(key, 0.0)) * (1.0 + float(r)) - 1.0
    return [
        {"year": year, "month": month, "ret": _round(value)}
        for (year, month), value in sorted(buckets.items())
    ]


def compute_performance(
    returns: Sequence[float], *, risk_free: float = 0.0
) -> PerformanceStats:
    """Assemble the scalar performance/risk summary for a return series."""
    arr = as_array(returns)
    n = arr.size
    if n == 0:
        raise ValueError("returns must not be empty")

    mdd = max_drawdown(returns)
    ann_ret = annualized_return(returns)
    calmar = _round(ann_ret / abs(mdd)) if mdd < 0.0 else 0.0
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False)) if n > 3 else 0.0

    return PerformanceStats(
        n_sessions=n,
        annualized_return=ann_ret,
        annualized_volatility=annualized_volatility(returns),
        sharpe=sharpe_ratio(returns, risk_free=risk_free),
        sortino=sortino_ratio(returns, risk_free=risk_free),
        calmar=calmar,
        max_drawdown=mdd,
        max_drawdown_duration=max_drawdown_duration(returns),
        time_to_recovery=time_to_recovery(returns),
        skewness=_round(skew),
        excess_kurtosis=_round(kurt),
        best_day=_round(float(np.max(arr))),
        worst_day=_round(float(np.min(arr))),
        positive_fraction=_round(float(np.mean(arr > 0.0))),
        var_95=historical_var(returns, level=0.95),
        cvar_95=historical_cvar(returns, level=0.95),
        var_99=historical_var(returns, level=0.99),
        cvar_99=historical_cvar(returns, level=0.99),
        cornish_fisher_var_95=cornish_fisher_var(returns, level=0.95),
    )
