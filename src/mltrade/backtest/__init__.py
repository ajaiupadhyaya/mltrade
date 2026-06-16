"""Walk-forward backtest engine for MLTrade.

Public API::

    from mltrade.backtest import (
        BacktestResult,
        Fill,
        PortfolioState,
        apply_fill,
        mark_to_market,
        run_backtest,
    )
"""

from mltrade.backtest.accounting import Fill, PortfolioState, apply_fill, mark_to_market
from mltrade.backtest.engine import BacktestConfig, run_backtest
from mltrade.backtest.reporting import BacktestResult, CostSummary, EvaluationWindow

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostSummary",
    "EvaluationWindow",
    "Fill",
    "PortfolioState",
    "apply_fill",
    "mark_to_market",
    "run_backtest",
]
