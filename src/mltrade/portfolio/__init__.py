"""Portfolio construction package.

Public API
----------
build_target : callable
    Solve for a constrained long-only portfolio given symbol forecasts and
    trailing volatility estimates.  Returns an :class:`OptimizationResult`.

PortfolioLimits : frozen Pydantic model
    Hard constraints forwarded to the CVXPY solver.

OptimizationResult : frozen Pydantic model
    Weights, cash weight, and a ``blocked`` flag produced by
    :func:`build_target`.
"""

from mltrade.portfolio.optimizer import (
    OptimizationResult,
    PortfolioLimits,
    build_target,
)

__all__ = [
    "OptimizationResult",
    "PortfolioLimits",
    "build_target",
]
