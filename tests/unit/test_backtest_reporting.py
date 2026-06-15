from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from mltrade.backtest.reporting import (
    compute_evaluation_windows,
    compute_metrics,
)

_COST_BPS = Decimal("5")


def _sessions(count: int) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_evaluation_windows_use_exact_consecutive_period_slices() -> None:
    equity_curve = [100.0 + float(index) for index in range(127)]
    cost_vals = [float(index) / 10.0 for index in range(126)]
    turnover_vals = [float(index) / 100.0 for index in range(126)]
    hit_flags = [index % 3 == 0 for index in range(126)]
    execution_sessions = _sessions(126)

    windows = compute_evaluation_windows(
        equity_curve=equity_curve,
        cost_vals=cost_vals,
        turnover_vals=turnover_vals,
        hit_flags=hit_flags,
        execution_sessions=execution_sessions,
        cost_bps=_COST_BPS,
        window_sessions=63,
    )

    assert len(windows) == 2
    first_expected = compute_metrics(
        equity_curve[0:64],
        sum(cost_vals[0:63]),
        turnover_vals[0:63],
        hit_flags[0:63],
        _COST_BPS,
    )
    second_expected = compute_metrics(
        equity_curve[63:127],
        sum(cost_vals[63:126]),
        turnover_vals[63:126],
        hit_flags[63:126],
        _COST_BPS,
    )

    first, second = windows
    assert first.start_session == execution_sessions[0]
    assert first.end_session == execution_sessions[62]
    assert first.sessions == 63
    assert second.start_session == execution_sessions[63]
    assert second.end_session == execution_sessions[125]
    assert second.sessions == 63
    assert first.model_dump(exclude={"start_session", "end_session", "sessions"}) == (
        first_expected.model_dump(exclude={"cost_bps"})
    )
    assert second.model_dump(exclude={"start_session", "end_session", "sessions"}) == (
        second_expected.model_dump(exclude={"cost_bps"})
    )


def test_evaluation_windows_omit_62_period_partial_and_include_63() -> None:
    def build(periods: int) -> int:
        windows = compute_evaluation_windows(
            equity_curve=[100.0 + float(index) for index in range(periods + 1)],
            cost_vals=[1.0] * periods,
            turnover_vals=[0.1] * periods,
            hit_flags=[True] * periods,
            execution_sessions=_sessions(periods),
            cost_bps=_COST_BPS,
            window_sessions=63,
        )
        return len(windows)

    assert build(125) == 1
    assert build(126) == 2


def test_compute_metrics_one_return_period_is_rounded_deterministically() -> None:
    summary = compute_metrics(
        [100.0, 101.23456789],
        0.123456789012,
        [0.333333333333],
        [True],
        _COST_BPS,
    )

    assert summary.annualized_return == round(
        (101.23456789 / 100.0) ** 252.0 - 1.0,
        10,
    )
    assert summary.annualized_volatility == 0.0
    assert summary.sharpe == 0.0
    assert summary.max_drawdown == 0.0
    assert summary.total_costs == 0.123456789
    assert summary.turnover == 0.3333333333
    assert summary.hit_rate == 1.0


def test_compute_metrics_two_return_periods_are_rounded_deterministically() -> None:
    equity_curve = [100.0, 110.0, 99.0]
    summary = compute_metrics(
        equity_curve,
        1.234567890123,
        [0.1, 0.2],
        [True, False],
        _COST_BPS,
    )
    repeated = compute_metrics(
        equity_curve,
        1.234567890123,
        [0.1, 0.2],
        [True, False],
        _COST_BPS,
    )

    assert summary == repeated
    assert summary.annualized_return == round((99.0 / 100.0) ** 126.0 - 1.0, 10)
    # Daily returns are [0.10, -0.10]; sample stdev (ddof=1) is sqrt(0.02).
    assert summary.annualized_volatility == round((0.02**0.5) * (252.0**0.5), 10)
    assert summary.sharpe == round(
        summary.annualized_return / summary.annualized_volatility,
        10,
    )
    assert summary.max_drawdown == -0.1
    assert summary.total_costs == 1.2345678901
    assert summary.turnover == 0.15
    assert summary.hit_rate == 0.5
