"""Integration tests for walk-forward backtester (Task 10).

Tests cover:
1. test_walk_forward_backtest_is_deterministic — first == second
2. test_higher_cost_lowers_return — 10 bps return ≤ 2 bps return
3. test_baselines_present — equal_weight and cash baselines exist
4. test_sessions_count — sessions > 250
5. test_metrics_finite — sharpe, max_drawdown, annualized_return, annualized_volatility

Fixture range: date(2019,1,2)..date(2026,6,12)
  ~1900 XNYS sessions total
  504 for warmup warmup + ~1400 backtest sessions
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from mltrade.backtest import BacktestResult, run_backtest
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.universe import MVP_UNIVERSE

# ---------------------------------------------------------------------------
# Module-scoped fixture — bars built once for all tests in this module.
# ---------------------------------------------------------------------------

_INGESTED_AT = datetime(2026, 6, 12, 20, 0, tzinfo=UTC)
_FIXTURE_START = date(2019, 1, 2)
_FIXTURE_END = date(2026, 6, 12)

_bars = DeterministicBarSource(seed=42).fetch(
    MVP_UNIVERSE,
    _FIXTURE_START,
    _FIXTURE_END,
    _INGESTED_AT,
)


@pytest.fixture(scope="module")
def backtest_result() -> BacktestResult:
    """Run the backtest once and share across tests."""
    return run_backtest(_bars, cost_bps=Decimal("5"))


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_walk_forward_backtest_is_deterministic() -> None:
    """Two identical run_backtest calls produce identical results."""
    first = run_backtest(_bars, cost_bps=Decimal("5"))
    second = run_backtest(_bars, cost_bps=Decimal("5"))
    assert first == second, "Backtest is not deterministic: first != second"
    # cost_sensitivity keys must be exactly {2, 5, 10}
    assert set(first.cost_sensitivity) == {
        Decimal("2"),
        Decimal("5"),
        Decimal("10"),
    }
    # sessions must be > 250
    assert first.sessions > 250


# ---------------------------------------------------------------------------
# 2. Higher cost lowers return
# ---------------------------------------------------------------------------


def test_higher_cost_lowers_return(backtest_result: BacktestResult) -> None:
    """Return at 10 bps <= return at 2 bps (costs reduce net return)."""
    ret_2 = backtest_result.cost_sensitivity[Decimal("2")].annualized_return
    ret_10 = backtest_result.cost_sensitivity[Decimal("10")].annualized_return
    assert ret_10 <= ret_2, (
        f"Expected 10 bps return ({ret_10:.6f}) <= 2 bps return ({ret_2:.6f})"
    )


# ---------------------------------------------------------------------------
# 3. Baselines present
# ---------------------------------------------------------------------------


def test_baselines_present(backtest_result: BacktestResult) -> None:
    """equal_weight_return and cash_return must exist and be finite."""
    assert math.isfinite(backtest_result.equal_weight_return)
    assert math.isfinite(backtest_result.cash_return)
    # Cash return should always be 0.0
    assert backtest_result.cash_return == 0.0


# ---------------------------------------------------------------------------
# 4. Sessions count
# ---------------------------------------------------------------------------


def test_sessions_count(backtest_result: BacktestResult) -> None:
    """sessions > 250 after warmup."""
    assert backtest_result.sessions > 250, (
        f"Expected > 250 backtest sessions, got {backtest_result.sessions}"
    )


# ---------------------------------------------------------------------------
# 5. Metrics finite
# ---------------------------------------------------------------------------


def test_metrics_finite(backtest_result: BacktestResult) -> None:
    """Core metrics are all finite numbers (not NaN or inf)."""
    assert math.isfinite(backtest_result.sharpe), "sharpe is not finite"
    assert math.isfinite(backtest_result.max_drawdown), "max_drawdown is not finite"
    assert math.isfinite(backtest_result.annualized_return), (
        "annualized_return is not finite"
    )
    assert math.isfinite(backtest_result.annualized_volatility), (
        "annualized_volatility is not finite"
    )


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------


def test_cost_sensitivity_keys(backtest_result: BacktestResult) -> None:
    """cost_sensitivity has exactly {2, 5, 10} bps keys."""
    assert set(backtest_result.cost_sensitivity) == {
        Decimal("2"),
        Decimal("5"),
        Decimal("10"),
    }


def test_cost_sensitivity_metrics_finite(backtest_result: BacktestResult) -> None:
    """All cost_sensitivity entries have finite metrics."""
    for bps, summary in backtest_result.cost_sensitivity.items():
        assert math.isfinite(summary.annualized_return), (
            f"cost {bps} bps: annualized_return not finite"
        )
        assert math.isfinite(summary.annualized_volatility), (
            f"cost {bps} bps: annualized_volatility not finite"
        )
        assert math.isfinite(summary.sharpe), f"cost {bps} bps: sharpe not finite"
        assert math.isfinite(summary.max_drawdown), (
            f"cost {bps} bps: max_drawdown not finite"
        )


def test_max_drawdown_is_nonpositive(backtest_result: BacktestResult) -> None:
    """Max drawdown must be <= 0 (it's a loss metric)."""
    assert backtest_result.max_drawdown <= 0.0


def test_hit_rate_in_range(backtest_result: BacktestResult) -> None:
    """Hit rate must be in [0, 1]."""
    assert 0.0 <= backtest_result.hit_rate <= 1.0


def test_per_symbol_contribution_has_universe_symbols(
    backtest_result: BacktestResult,
) -> None:
    """per_symbol_contribution has at least some universe symbols."""
    assert len(backtest_result.per_symbol_contribution) > 0
    for contrib in backtest_result.per_symbol_contribution.values():
        assert math.isfinite(contrib)
