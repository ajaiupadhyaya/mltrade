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
from pydantic import ValidationError

from mltrade.backtest import BacktestConfig, BacktestResult, run_backtest
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.models import ForecastBlocked, RidgeForecastConfig
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


def test_backtest_config_defaults_preserve_existing_result() -> None:
    original = run_backtest(_bars)
    explicit = run_backtest(_bars, config=BacktestConfig())
    assert explicit == original


def test_backtest_emits_deterministic_evaluation_windows() -> None:
    first = run_backtest(_bars, config=BacktestConfig())
    second = run_backtest(_bars, config=BacktestConfig())

    assert first.evaluation_windows
    assert first.evaluation_windows == second.evaluation_windows


def test_evaluation_window_length_is_configurable() -> None:
    result = run_backtest(
        _bars,
        config=BacktestConfig(evaluation_window_sessions=126),
    )

    assert result.evaluation_windows
    assert all(window.sessions <= 126 for window in result.evaluation_windows)
    assert all(window.sessions == 126 for window in result.evaluation_windows[:-1])
    assert all(window.sessions >= 63 for window in result.evaluation_windows)
    execution_sessions = sorted({bar.session for bar in _bars})[505:]
    covered_sessions = sum(
        window.sessions for window in result.evaluation_windows
    )
    remainder = result.sessions % 126
    expected_covered = (
        result.sessions
        if remainder == 0 or remainder >= 63
        else result.sessions - remainder
    )
    assert covered_sessions == expected_covered
    offset = 0
    for window in result.evaluation_windows:
        assert window.start_session == execution_sessions[offset]
        assert window.end_session == execution_sessions[
            offset + window.sessions - 1
        ]
        offset += window.sessions
    for previous, current in zip(
        result.evaluation_windows,
        result.evaluation_windows[1:],
        strict=False,
    ):
        assert previous.end_session < current.start_session


def test_backtest_uses_configurable_cadence_and_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mltrade.backtest.engine as engine

    forecast_calls = 0
    simulated_costs: list[Decimal] = []

    def block_forecast(*args: object, **kwargs: object) -> None:
        nonlocal forecast_calls
        forecast_calls += 1
        raise ForecastBlocked("test block")

    def fake_run_sim(
        decisions: list[tuple[int, dict[str, Decimal]]],
        sessions: list[object],
        cost_bps: Decimal,
        all_symbols: list[str],
        initial_equity: Decimal,
    ) -> tuple[list[float], float, list[float], list[bool], list[float]]:
        del decisions, all_symbols
        simulated_costs.append(cost_bps)
        return (
            [float(initial_equity)] * (len(sessions) + 1),
            0.0,
            [0.0] * len(sessions),
            [False] * len(sessions),
            [0.0] * len(sessions),
        )

    monkeypatch.setattr(engine, "generate_forecast_batch", block_forecast)
    monkeypatch.setattr(engine, "_run_sim", fake_run_sim)

    config = BacktestConfig(
        retrain_every_sessions=7,
        cost_bps=Decimal("3"),
        cost_sensitivity_bps=(Decimal("1"), Decimal("4")),
    )
    result = run_backtest(_bars, config=config)

    expected_calls = (result.sessions + 6) // 7
    assert forecast_calls == expected_calls
    assert simulated_costs == [
        Decimal("3"),
        Decimal("1"),
        Decimal("4"),
    ]
    assert set(result.cost_sensitivity) == {Decimal("1"), Decimal("4")}


def test_backtest_rejects_conflicting_legacy_cost() -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        run_backtest(
            _bars,
            cost_bps=Decimal("7"),
            config=BacktestConfig(cost_bps=Decimal("3")),
        )


def test_explicit_default_legacy_cost_conflicts_with_different_config() -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        run_backtest(
            _bars,
            cost_bps=Decimal("5"),
            config=BacktestConfig(cost_bps=Decimal("10")),
        )


def test_equal_legacy_and_config_costs_are_accepted() -> None:
    config = BacktestConfig(cost_bps=Decimal("7"))

    configured = run_backtest(_bars, config=config)
    explicit_equal = run_backtest(
        _bars,
        cost_bps=Decimal("7"),
        config=config,
    )

    assert explicit_equal == configured


def test_legacy_cost_without_config_is_used() -> None:
    legacy = run_backtest(_bars, cost_bps=Decimal("7"))
    configured = run_backtest(
        _bars,
        config=BacktestConfig(cost_bps=Decimal("7")),
    )

    assert legacy == configured


@pytest.mark.parametrize("value", (5, 5.0, "5"))
def test_backtest_config_requires_strict_decimal_headline_cost(
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="cost_bps"):
        BacktestConfig(cost_bps=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (5, 5.0, "5"))
def test_backtest_config_requires_strict_decimal_sensitivity_costs(
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="cost_sensitivity_bps"):
        BacktestConfig(
            cost_sensitivity_bps=(value,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", (5, 5.0, "5"))
def test_legacy_cost_requires_strict_decimal(value: object) -> None:
    with pytest.raises(ValidationError, match="cost_bps"):
        run_backtest(_bars, cost_bps=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (Decimal("-0.01"), Decimal("100.01")),
)
def test_legacy_cost_rejects_out_of_range_decimal(value: Decimal) -> None:
    with pytest.raises(ValidationError, match="cost_bps"):
        run_backtest(_bars, cost_bps=value)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("retrain_every_sessions", 0),
        ("cost_bps", Decimal("-1")),
        ("cost_bps", Decimal("101")),
        ("cost_sensitivity_bps", ()),
        ("evaluation_window_sessions", 62),
        ("retrain_every_sessions", True),
        ("evaluation_window_sessions", 63.0),
    ),
)
def test_backtest_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match=field):
        BacktestConfig.model_validate({field: value})


def test_backtest_config_copy_revalidates_nested_constructed_models() -> None:
    config = BacktestConfig()
    unsafe = RidgeForecastConfig.model_construct(alpha=0.0)

    updated = config.model_copy(update={"cost_bps": Decimal("7")})
    assert updated.cost_bps == Decimal("7")
    with pytest.raises(ValidationError, match="alpha"):
        config.model_copy(update={"forecast": unsafe})
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValidationError, match="evaluation_window_sessions"):
            config.copy(update={"evaluation_window_sessions": 1})


def test_backtest_config_rejects_direct_constructed_nested_model() -> None:
    unsafe = RidgeForecastConfig.model_construct(alpha=0.0)

    with pytest.raises(ValidationError, match="alpha"):
        BacktestConfig(forecast=unsafe)


def test_backtest_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BacktestConfig.model_validate({"slippage_model": "linear"})


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
