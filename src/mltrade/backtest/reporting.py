"""Backtest result data models and metric computation.

``BacktestResult`` is the top-level frozen Pydantic model returned by
``run_backtest``.  All float fields are rounded to 10 decimal places for
determinism.

Metric definitions
------------------
- annualized_return: (final_equity / initial_equity) ^ (252 / n_sessions) - 1
- annualized_volatility: std(daily_returns) * sqrt(252)
- sharpe: annualized_return / annualized_volatility  (zero risk-free rate)
- max_drawdown: maximum peak-to-trough drawdown (always <= 0)
- hit_rate: fraction of sessions with positive daily return
- turnover: average per-session (traded notional / equity)
- equal_weight_return: baseline rebalancing to 1/N each session
- cash_return: always 0.0 (no interest on cash)
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

# Rounding precision for all float outputs (determinism guarantee).
_ROUND_PLACES: int = 10


def _round(x: float) -> float:
    """Round a float to _ROUND_PLACES decimal places for determinism."""
    return round(x, _ROUND_PLACES)


class SessionMetrics(BaseModel):
    """Per-session backtest metrics."""

    model_config = ConfigDict(frozen=True)

    session_date: object  # date — typed as object to avoid import cycle
    equity: float
    cash: float
    turnover: float  # traded notional / equity
    costs: float


class CostSummary(BaseModel):
    """Backtest performance summary at a specific cost level."""

    model_config = ConfigDict(frozen=True)

    cost_bps: Decimal
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    total_costs: float
    turnover: float
    hit_rate: float


class EvaluationWindow(BaseModel):
    """Metrics for one non-overlapping execution-session window."""

    model_config = ConfigDict(frozen=True)

    start_session: date
    end_session: date
    sessions: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float


class BacktestResult(BaseModel):
    """Immutable backtest result with headline metrics and cost sensitivity.

    All float fields are rounded to 10 decimal places for cross-run
    determinism.  ``first == second`` holds for two identical ``run_backtest``
    calls with the same inputs.
    """

    model_config = ConfigDict(frozen=True)

    sessions: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    total_costs: float
    hit_rate: float
    cost_sensitivity: dict[Decimal, CostSummary]
    per_symbol_contribution: dict[str, float]
    equal_weight_return: float
    cash_return: float
    evaluation_windows: tuple[EvaluationWindow, ...] = ()


def compute_metrics(
    equity_curve: list[float],
    total_costs_val: float,
    turnover_vals: list[float],
    hit_flags: list[bool],
    cost_bps: Decimal,
) -> CostSummary:
    """Compute backtest performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        List of equity values, one per session, starting from initial equity.
    total_costs_val:
        Total accumulated transaction costs over the backtest.
    turnover_vals:
        Per-session traded notional / equity values.
    hit_flags:
        Per-session booleans: True if session return > 0.
    cost_bps:
        The transaction cost level used (for labelling).

    Returns
    -------
    CostSummary
        Performance metrics rounded for determinism.
    """
    n = len(equity_curve)
    if n < 2:
        return CostSummary(
            cost_bps=cost_bps,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            total_costs=_round(total_costs_val),
            turnover=0.0,
            hit_rate=0.0,
        )

    initial = equity_curve[0]
    final = equity_curve[-1]
    n_sessions = n - 1  # number of return periods

    # Annualised return
    if initial <= 0.0 or final <= 0.0:
        ann_return = 0.0
    else:
        ann_return = (final / initial) ** (252.0 / n_sessions) - 1.0

    # Daily returns
    daily_returns = [
        equity_curve[i + 1] / equity_curve[i] - 1.0
        for i in range(n_sessions)
    ]

    # Annualised volatility
    if len(daily_returns) < 2:
        ann_vol = 0.0
    else:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / (
            len(daily_returns) - 1
        )
        ann_vol = math.sqrt(variance) * math.sqrt(252.0)

    # Sharpe ratio (zero risk-free rate)
    sharpe = ann_return / ann_vol if ann_vol > 0.0 else 0.0

    # Max drawdown (peak-to-trough)
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0.0 else 0.0
        if dd < max_dd:
            max_dd = dd

    # Hit rate
    hit_rate = sum(1 for h in hit_flags if h) / len(hit_flags) if hit_flags else 0.0

    # Average turnover
    avg_turnover = sum(turnover_vals) / len(turnover_vals) if turnover_vals else 0.0

    return CostSummary(
        cost_bps=cost_bps,
        annualized_return=_round(ann_return),
        annualized_volatility=_round(ann_vol),
        sharpe=_round(sharpe),
        max_drawdown=_round(max_dd),
        total_costs=_round(total_costs_val),
        turnover=_round(avg_turnover),
        hit_rate=_round(hit_rate),
    )


def compute_evaluation_windows(
    *,
    equity_curve: list[float],
    cost_vals: list[float],
    turnover_vals: list[float],
    hit_flags: list[bool],
    execution_sessions: list[date],
    cost_bps: Decimal,
    window_sessions: int,
) -> tuple[EvaluationWindow, ...]:
    """Compute consecutive, non-overlapping headline evaluation windows."""
    period_count = len(equity_curve) - 1
    if not (
        len(cost_vals)
        == len(turnover_vals)
        == len(hit_flags)
        == len(execution_sessions)
        == period_count
    ):
        raise ValueError("evaluation-window inputs must be period-aligned")

    windows: list[EvaluationWindow] = []
    start = 0
    while start < period_count:
        end = min(start + window_sessions, period_count)
        sessions = end - start
        if sessions < 63:
            break

        summary = compute_metrics(
            equity_curve[start : end + 1],
            sum(cost_vals[start:end]),
            turnover_vals[start:end],
            hit_flags[start:end],
            cost_bps,
        )
        windows.append(
            EvaluationWindow(
                start_session=execution_sessions[start],
                end_session=execution_sessions[end - 1],
                sessions=sessions,
                annualized_return=summary.annualized_return,
                annualized_volatility=summary.annualized_volatility,
                sharpe=summary.sharpe,
                max_drawdown=summary.max_drawdown,
            )
        )
        start = end

    return tuple(windows)
