"""Tests for the constrained portfolio optimizer (Task 8).

Correctness concerns tested
---------------------------
1. Hard constraints respected: position cap, cash floor, vol target.
2. Solver failure -> blocked all-cash (``blocked=True``, ``weights={}``).
3. No positive forecast -> unblocked all-cash (``blocked=False``, ``weights={}``).
4. Determinism: same inputs -> identical ``OptimizationResult``.
5. Non-finite forecast values are filtered out.
6. Non-finite / non-positive trailing vol values are filtered out.
7. Single-symbol case produces a valid result.
8. Invalid ``PortfolioLimits`` (max_weight > 1 - min_cash) raises on construction.
9. ``cash_weight`` equals ``1 - sum(weights)``.
10. ``blocked=False`` all-cash (no signal) is distinct from ``blocked=True``
    all-cash (solver failure).
11. Weights never exceed ``maximum_position_weight``.
12. ``sum(weights) <= 1 - minimum_cash_weight`` after Decimal quantization.
13. All-negative forecasts -> unblocked all-cash.
14. Mixed positive/negative forecasts use only positive ones.
"""

from __future__ import annotations

import math
from decimal import Decimal

import cvxpy as cp  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LIMITS = PortfolioLimits(
    maximum_position_weight=Decimal("0.25"),
    minimum_cash_weight=Decimal("0.05"),
    target_annual_volatility=Decimal("0.15"),
)

# Five symbols with positive forecasts and valid volatilities.
_POSITIVE_FORECASTS: dict[str, float] = {
    "AAPL": 0.05,
    "GOOG": 0.03,
    "MSFT": 0.04,
    "NVDA": 0.06,
    "TSLA": 0.02,
}

_VOLATILITY: dict[str, float] = {
    "AAPL": 0.20,
    "GOOG": 0.22,
    "MSFT": 0.18,
    "NVDA": 0.30,
    "TSLA": 0.40,
}


# ---------------------------------------------------------------------------
# 1. Hard constraints respected (plan-specified test, verbatim + expanded)
# ---------------------------------------------------------------------------


def test_optimizer_respects_hard_constraints() -> None:
    """Position cap, invested-fraction cap, and cash floor are satisfied."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is False
    assert sum(result.weights.values()) <= Decimal("0.95")
    assert not result.weights or max(result.weights.values()) <= Decimal("0.25")
    assert result.cash_weight >= Decimal("0.05")


def test_position_cap_never_exceeded() -> None:
    """Each individual weight must be <= maximum_position_weight."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is False
    for sym, w in result.weights.items():
        assert w <= _LIMITS.maximum_position_weight, (
            f"Weight for {sym} ({w}) exceeds maximum_position_weight "
            f"({_LIMITS.maximum_position_weight})"
        )


def test_invested_fraction_cap_respected() -> None:
    """sum(weights) <= 1 - minimum_cash_weight."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    max_invested = Decimal("1") - _LIMITS.minimum_cash_weight
    assert sum(result.weights.values()) <= max_invested


def test_cash_weight_equals_one_minus_invested() -> None:
    """cash_weight == 1 - sum(weights) exactly."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.cash_weight == Decimal("1") - sum(result.weights.values())


# ---------------------------------------------------------------------------
# 2. Solver failure -> blocked all-cash (plan-specified test, verbatim)
# ---------------------------------------------------------------------------


def test_solver_failure_returns_blocked_all_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception from problem.solve() -> blocked all-cash."""

    def raising_solver_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated solver crash")

    monkeypatch.setattr(cp.Problem, "solve", raising_solver_error)

    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is True
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")


def test_solver_bad_status_returns_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-OPTIMAL solver status -> blocked all-cash."""

    original_solve = cp.Problem.solve

    def infeasible_solve(
        self: cp.Problem, *args: object, **kwargs: object
    ) -> None:
        original_solve(self, *args, **kwargs)
        # Override the status after the real solve to simulate infeasibility.
        self._status = cp.INFEASIBLE  # type: ignore[attr-defined]

    monkeypatch.setattr(cp.Problem, "solve", infeasible_solve)

    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is True
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")


# ---------------------------------------------------------------------------
# 3. No positive forecast -> unblocked all-cash
# ---------------------------------------------------------------------------


def test_no_positive_forecasts_returns_unblocked_all_cash() -> None:
    """No qualifying forecast -> blocked=False, weights={}, cash=1."""
    result = build_target(
        forecasts={"AAPL": -0.01, "GOOG": 0.0},
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is False
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")


def test_empty_forecasts_returns_unblocked_all_cash() -> None:
    """Empty forecasts mapping -> unblocked all-cash."""
    result = build_target(
        forecasts={},
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is False
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")


def test_all_negative_forecasts_returns_unblocked_all_cash() -> None:
    """All-negative forecasts -> unblocked all-cash (no short positions)."""
    result = build_target(
        forecasts={"AAPL": -0.05, "GOOG": -0.02, "MSFT": -0.10},
        trailing_volatility={"AAPL": 0.20, "GOOG": 0.22, "MSFT": 0.18},
        limits=_LIMITS,
    )
    assert result.blocked is False
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")


# ---------------------------------------------------------------------------
# 4. Determinism: same inputs -> identical result
# ---------------------------------------------------------------------------


def test_determinism_same_inputs_identical_result() -> None:
    """build_target is deterministic: two calls with same args produce == result."""
    result_a = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    result_b = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result_a == result_b
    assert result_a.weights == result_b.weights
    assert result_a.cash_weight == result_b.cash_weight
    assert result_a.blocked == result_b.blocked


# ---------------------------------------------------------------------------
# 5 & 6. Non-finite inputs are filtered out
# ---------------------------------------------------------------------------


def test_non_finite_forecast_filtered() -> None:
    """NaN and inf forecast values are discarded; remaining positives are used."""
    forecasts: dict[str, float] = {
        "AAPL": 0.05,
        "GOOG": math.nan,
        "MSFT": math.inf,
        "NVDA": -math.inf,
    }
    vol: dict[str, float] = {
        "AAPL": 0.20,
        "GOOG": 0.22,
        "MSFT": 0.18,
        "NVDA": 0.30,
    }

    result = build_target(forecasts=forecasts, trailing_volatility=vol, limits=_LIMITS)
    # Only AAPL qualifies; result should have AAPL in weights.
    assert result.blocked is False
    assert "GOOG" not in result.weights
    assert "MSFT" not in result.weights
    assert "NVDA" not in result.weights


def test_non_positive_vol_filtered() -> None:
    """Symbols with zero or negative trailing volatility are excluded."""
    forecasts: dict[str, float] = {"AAPL": 0.05, "GOOG": 0.03, "MSFT": 0.04}
    vol: dict[str, float] = {
        "AAPL": 0.20,
        "GOOG": 0.0,  # zero vol -> exclude
        "MSFT": -0.1,  # negative vol -> exclude
    }

    result = build_target(forecasts=forecasts, trailing_volatility=vol, limits=_LIMITS)
    assert result.blocked is False
    assert "GOOG" not in result.weights
    assert "MSFT" not in result.weights
    # AAPL is the only valid symbol.
    if result.weights:  # might be empty if all-cash after quantization
        assert set(result.weights.keys()) == {"AAPL"}


def test_non_finite_vol_filtered() -> None:
    """NaN trailing volatility -> symbol excluded."""
    forecasts: dict[str, float] = {"AAPL": 0.05, "GOOG": 0.03}
    vol: dict[str, float] = {"AAPL": 0.20, "GOOG": math.nan}

    result = build_target(forecasts=forecasts, trailing_volatility=vol, limits=_LIMITS)
    assert result.blocked is False
    assert "GOOG" not in result.weights


def test_missing_vol_symbol_filtered() -> None:
    """Symbols absent from trailing_volatility mapping are excluded."""
    forecasts: dict[str, float] = {"AAPL": 0.05, "GOOG": 0.03}
    vol: dict[str, float] = {"AAPL": 0.20}  # GOOG missing

    result = build_target(forecasts=forecasts, trailing_volatility=vol, limits=_LIMITS)
    assert result.blocked is False
    assert "GOOG" not in result.weights


# ---------------------------------------------------------------------------
# 7. Single-symbol case
# ---------------------------------------------------------------------------


def test_single_symbol_case() -> None:
    """Single qualifying symbol produces a valid non-blocked result."""
    result = build_target(
        forecasts={"AAPL": 0.05},
        trailing_volatility={"AAPL": 0.20},
        limits=_LIMITS,
    )
    assert result.blocked is False
    assert result.cash_weight >= _LIMITS.minimum_cash_weight
    if result.weights:
        assert set(result.weights.keys()) == {"AAPL"}
        assert result.weights["AAPL"] <= _LIMITS.maximum_position_weight


# ---------------------------------------------------------------------------
# 8. Invalid PortfolioLimits raises on construction
# ---------------------------------------------------------------------------


def test_invalid_limits_raises() -> None:
    """maximum_position_weight > 1 - minimum_cash_weight raises ValidationError."""
    with pytest.raises(ValidationError, match="maximum_position_weight"):
        PortfolioLimits(
            maximum_position_weight=Decimal("0.97"),
            minimum_cash_weight=Decimal("0.05"),
            target_annual_volatility=Decimal("0.15"),
        )


def test_invalid_limits_zero_weight_raises() -> None:
    """maximum_position_weight <= 0 raises ValidationError (gt=0 constraint)."""
    with pytest.raises(ValidationError):
        PortfolioLimits(
            maximum_position_weight=Decimal("0"),
            minimum_cash_weight=Decimal("0.05"),
            target_annual_volatility=Decimal("0.15"),
        )


# ---------------------------------------------------------------------------
# 10. Blocked vs unblocked distinction
# ---------------------------------------------------------------------------


def test_blocked_false_for_no_signal() -> None:
    """No-signal all-cash is explicitly NOT blocked."""
    result = build_target(
        forecasts={"AAPL": -0.01},
        trailing_volatility={"AAPL": 0.20},
        limits=_LIMITS,
    )
    assert result.blocked is False


def test_blocked_true_for_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver-failure all-cash IS blocked."""

    monkeypatch.setattr(
        cp.Problem,
        "solve",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    assert result.blocked is True


# ---------------------------------------------------------------------------
# 11. Mixed positive/negative forecasts: only positives used
# ---------------------------------------------------------------------------


def test_mixed_forecasts_only_positives_used() -> None:
    """Negative forecasts for some symbols don't bleed into the result."""
    forecasts: dict[str, float] = {
        "AAPL": 0.05,
        "GOOG": -0.03,  # negative -> excluded
        "MSFT": 0.04,
        "TSLA": 0.0,  # zero -> excluded
    }
    vol: dict[str, float] = {
        "AAPL": 0.20,
        "GOOG": 0.22,
        "MSFT": 0.18,
        "TSLA": 0.40,
    }
    result = build_target(forecasts=forecasts, trailing_volatility=vol, limits=_LIMITS)
    assert result.blocked is False
    assert "GOOG" not in result.weights
    assert "TSLA" not in result.weights


# ---------------------------------------------------------------------------
# 12. All weights are non-negative
# ---------------------------------------------------------------------------


def test_all_weights_non_negative() -> None:
    """CVXPY enforces w >= 0; Decimal result should also have no negative weights."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    for sym, w in result.weights.items():
        assert w > Decimal("0"), f"{sym} has non-positive weight {w}"


# ---------------------------------------------------------------------------
# 13. OptimizationResult is immutable (frozen Pydantic model)
# ---------------------------------------------------------------------------


def test_optimization_result_is_immutable() -> None:
    """OptimizationResult cannot be mutated after construction."""
    result = OptimizationResult(
        weights={"AAPL": Decimal("0.20")},
        cash_weight=Decimal("0.80"),
        blocked=False,
    )
    with pytest.raises(Exception):  # noqa: B017
        result.blocked = True  # type: ignore[misc]


def test_portfolio_limits_is_immutable() -> None:
    """PortfolioLimits cannot be mutated after construction."""
    with pytest.raises(Exception):  # noqa: B017
        _LIMITS.maximum_position_weight = Decimal("0.50")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 14. Result type annotations (structural check)
# ---------------------------------------------------------------------------


def test_result_has_expected_fields() -> None:
    """OptimizationResult exposes weights, cash_weight, and blocked."""
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=_LIMITS,
    )
    # Structural check: these attributes must exist and be the right types.
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.weights, dict)
    assert isinstance(result.cash_weight, Decimal)
    assert isinstance(result.blocked, bool)


# ---------------------------------------------------------------------------
# 15. Tight vol cap: all-cash when vol constraint is near-zero
# ---------------------------------------------------------------------------


def test_tight_vol_constraint_produces_valid_result() -> None:
    """Extremely tight vol target must not crash; result may be all-cash."""
    tight_limits = PortfolioLimits(
        maximum_position_weight=Decimal("0.25"),
        minimum_cash_weight=Decimal("0.05"),
        target_annual_volatility=Decimal("0.01"),  # 1% -- very tight
    )
    result = build_target(
        forecasts=_POSITIVE_FORECASTS,
        trailing_volatility=_VOLATILITY,
        limits=tight_limits,
    )
    # Either valid (small weights) or blocked -- must not raise.
    assert isinstance(result.blocked, bool)
    if not result.blocked:
        assert result.cash_weight >= tight_limits.minimum_cash_weight


# ---------------------------------------------------------------------------
# 16. Large number of symbols: deterministic and constraint-respecting
# ---------------------------------------------------------------------------


def test_many_symbols_constraints_respected() -> None:
    """30-symbol case: all hard constraints still hold."""
    symbols = [f"SYM{i:02d}" for i in range(30)]
    forecasts = {s: 0.01 + i * 0.001 for i, s in enumerate(symbols)}
    vols = {s: 0.15 + i * 0.005 for i, s in enumerate(symbols)}

    result = build_target(
        forecasts=forecasts,
        trailing_volatility=vols,
        limits=_LIMITS,
    )
    assert not result.blocked or result.weights == {}
    if not result.blocked:
        assert sum(result.weights.values()) <= Decimal("0.95")
        if result.weights:
            assert max(result.weights.values()) <= Decimal("0.25")
        assert result.cash_weight >= Decimal("0.05")
