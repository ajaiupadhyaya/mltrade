"""Institutional analytics layer for the MLTrade backtest.

Pure, deterministic statistics computed from per-session return series:

- :mod:`returns` — return-series primitives (annualised return/vol, Sharpe).
- :mod:`performance` — Sortino, Calmar, drawdown depth/duration/recovery,
  distribution moments, historical and Cornish-Fisher tail risk, and the chart
  series (drawdown, rolling Sharpe, histogram, monthly returns).
- :mod:`benchmark` — beta, Jensen's alpha (with t-stat), tracking error,
  information ratio, capture ratios.
- :mod:`overfitting` — Deflated Sharpe Ratio and Probability of Backtest
  Overfitting (PBO via CSCV).
- :mod:`attribution` — returns-based macro factor exposures.
"""

from __future__ import annotations

from mltrade.analytics.attribution import (
    ASSET_CLASS,
    MACRO_FACTORS,
    AttributionStats,
    FactorExposure,
    compute_attribution,
)
from mltrade.analytics.benchmark import BenchmarkStats, compute_benchmark_stats
from mltrade.analytics.overfitting import (
    OverfittingStats,
    compute_overfitting,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from mltrade.analytics.performance import (
    PerformanceStats,
    compute_performance,
    drawdown_series,
    monthly_returns,
    return_histogram,
    rolling_sharpe,
)
from mltrade.analytics.returns import (
    annualized_return,
    annualized_volatility,
    equity_to_returns,
    per_period_sharpe,
    sharpe_ratio,
)

__all__ = [
    "ASSET_CLASS",
    "MACRO_FACTORS",
    "AttributionStats",
    "BenchmarkStats",
    "FactorExposure",
    "OverfittingStats",
    "PerformanceStats",
    "annualized_return",
    "annualized_volatility",
    "compute_attribution",
    "compute_benchmark_stats",
    "compute_overfitting",
    "compute_performance",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "equity_to_returns",
    "monthly_returns",
    "per_period_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "return_histogram",
    "rolling_sharpe",
    "sharpe_ratio",
]
