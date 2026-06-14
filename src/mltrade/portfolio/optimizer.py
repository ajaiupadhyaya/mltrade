"""Constrained portfolio target optimizer.

Algorithm overview
------------------
1. Accept ``forecasts: Mapping[str, float]`` (symbol to predicted return) and
   ``trailing_volatility: Mapping[str, float]`` (symbol to annualised vol > 0).
2. Keep only symbols with a *positive*, finite forecast AND a positive, finite
   trailing volatility.
3. If no symbol qualifies, return an all-cash ``OptimizationResult`` with
   ``blocked=False``.  This is a valid no-signal flat position, not a failure.
4. Build a "desired" weight vector:
   a. Conviction = predicted_return / trailing_vol  (inverse-vol scaled)
   b. Normalize desired weights so they sum to ``1 - minimum_cash_weight``
      (our target invested fraction).
5. Solve with CVXPY using a FIXED symbol order (alphabetically sorted) and
   a DETERMINISTIC solver (CLARABEL, ships with cvxpy >= 1.6):

       minimize   sum_squares(w - desired)
       subject to w >= 0
                  w <= maximum_position_weight
                  sum(w) <= 1 - minimum_cash_weight
                  w' S w <= target_variance

   where S = diag(vol_i^2) (zero cross-correlation; diagonal covariance is a
   deliberate MVP simplification documented here).

6. Accept only ``OPTIMAL`` or ``OPTIMAL_INACCURATE`` status.  Any other
   status, exception from ``solve()``, ``None`` weight values, or non-finite
   weight values return a *blocked* all-cash result (``blocked=True``).

Decimal-quantization strategy (critical for hard-constraint compliance)
-----------------------------------------------------------------------
The solver returns floating-point weights satisfying constraints to ~1e-8.
To avoid rounding pushes that violate ``maximum_position_weight`` or the
invested-fraction cap:

- Each float weight is quantized DOWNWARD (``ROUND_FLOOR``) to 6 decimal
  places before conversion.
- Weights that quantize to 0 are dropped.
- ``cash_weight = Decimal("1") - sum(weights)`` is computed last, so the
  cash floor ``>= minimum_cash_weight`` is automatically satisfied (because
  sum(w) <= 1 - minimum_cash_weight was enforced by the solver).
- A final Python-level check verifies all three Decimal inequalities; if
  somehow violated the function returns blocked all-cash (defensive).

Determinism guarantee
---------------------
- Symbol order: always ``sorted(symbols)`` before building the numpy arrays.
- Solver: ``cp.CLARABEL`` -- deterministic convex solver.  Never falls back
  to the CVXPY default solver selection (which is heuristic and can vary).
- No randomness is introduced anywhere in this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal

import cvxpy as cp
import numpy as np

from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits

__all__ = [
    "OptimizationResult",
    "PortfolioLimits",
    "build_target",
]

# Precision for floor-quantization of solver output weights.
_WEIGHT_QUANT = Decimal("0.000001")  # 6 d.p.

# Solver statuses we treat as success.
_ACCEPTABLE_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _all_cash_flat() -> OptimizationResult:
    """Return a valid all-cash target (no signal, not blocked)."""
    return OptimizationResult(
        weights={},
        cash_weight=Decimal("1"),
        blocked=False,
    )


def _all_cash_blocked() -> OptimizationResult:
    """Return a blocked all-cash target (solver failure or infeasibility)."""
    return OptimizationResult(
        weights={},
        cash_weight=Decimal("1"),
        blocked=True,
    )


def build_target(
    *,
    forecasts: Mapping[str, float],
    trailing_volatility: Mapping[str, float],
    limits: PortfolioLimits,
) -> OptimizationResult:
    """Solve for a constrained long-only portfolio target.

    Parameters
    ----------
    forecasts:
        Symbol → predicted forward return.  Only *positive*, finite values
        are considered; negative or zero forecasts are discarded (we never
        short in the MVP).
    trailing_volatility:
        Symbol → annualised trailing volatility.  Must be positive and finite
        for a symbol to be included.
    limits:
        Hard portfolio constraints.  See :class:`PortfolioLimits`.

    Returns
    -------
    OptimizationResult
        Immutable result with ``weights``, ``cash_weight``, and ``blocked``.
        See :class:`OptimizationResult` for the blocked/unblocked distinction.
    """
    # ------------------------------------------------------------------
    # Step 1: Filter to symbols with positive finite forecast AND positive
    #         finite trailing volatility.
    # ------------------------------------------------------------------
    qualified: list[str] = []
    for symbol in sorted(forecasts):  # sorted for determinism
        fc = forecasts[symbol]
        if not math.isfinite(fc) or fc <= 0.0:
            continue
        vol = trailing_volatility.get(symbol)
        if vol is None or not math.isfinite(vol) or vol <= 0.0:
            continue
        qualified.append(symbol)

    # ------------------------------------------------------------------
    # Step 2: No qualifying symbol → unblocked all-cash (valid flat target).
    # ------------------------------------------------------------------
    if not qualified:
        return _all_cash_flat()

    n = len(qualified)

    # ------------------------------------------------------------------
    # Step 3: Build desired weight vector (inverse-vol scaled, normalized).
    # ------------------------------------------------------------------
    fc_arr = np.array([forecasts[s] for s in qualified], dtype=np.float64)
    vol_arr = np.array([trailing_volatility[s] for s in qualified], dtype=np.float64)

    conviction = fc_arr / vol_arr  # inverse-vol scaling
    total_conviction = float(np.sum(conviction))
    target_invested = float(Decimal("1") - limits.minimum_cash_weight)
    desired = (conviction / total_conviction) * target_invested  # shape (n,)

    # ------------------------------------------------------------------
    # Step 4: Build diagonal covariance matrix S = diag(vol_i^2).
    #
    # Assumption: zero cross-correlations (diagonal covariance).  This is a
    # deliberate MVP simplification -- correlations are non-trivial to
    # estimate reliably on short lookback windows and introduce estimation
    # error that could destabilize the optimizer.  The diagonal approximation
    # conservatively treats each position's risk as fully independent.
    # ------------------------------------------------------------------
    cov_matrix = np.diag(vol_arr**2)  # shape (n, n)
    target_variance = float(limits.target_annual_volatility) ** 2
    max_weight = float(limits.maximum_position_weight)

    # ------------------------------------------------------------------
    # Step 5: CVXPY formulation.
    # ------------------------------------------------------------------
    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(w - desired))  # type: ignore[attr-defined]
    constraints = [
        w >= 0,
        w <= max_weight,
        cp.sum(w) <= target_invested,  # type: ignore[attr-defined]
        cp.quad_form(w, cov_matrix) <= target_variance,  # type: ignore[attr-defined]
    ]
    problem = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------
    # Step 6: Solve with a DETERMINISTIC solver.
    #
    # CLARABEL is a deterministic interior-point solver that ships with
    # cvxpy >= 1.6.  We pin the solver explicitly so the solution is
    # reproducible and never depends on CVXPY's heuristic solver selection.
    # ------------------------------------------------------------------
    try:
        problem.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
    except Exception:  # broad catch: monkeypatched in tests
        return _all_cash_blocked()

    # ------------------------------------------------------------------
    # Step 7: Validate solver output.
    # ------------------------------------------------------------------
    if problem.status not in _ACCEPTABLE_STATUSES:
        return _all_cash_blocked()

    w_values = w.value  # numpy array or None
    if w_values is None:
        return _all_cash_blocked()

    if not np.all(np.isfinite(w_values)):
        return _all_cash_blocked()

    # ------------------------------------------------------------------
    # Step 8: Convert to Decimal with ROUND_FLOOR to prevent rounding from
    #         violating position-cap or invested-fraction constraints.
    # ------------------------------------------------------------------
    max_weight_dec = limits.maximum_position_weight
    max_invested_dec = Decimal("1") - limits.minimum_cash_weight

    raw_weights: dict[str, Decimal] = {}
    for symbol, w_float in zip(qualified, w_values, strict=True):
        w_dec = Decimal(str(w_float)).quantize(_WEIGHT_QUANT, rounding=ROUND_FLOOR)
        if w_dec <= Decimal("0"):
            continue
        raw_weights[symbol] = w_dec

    # ------------------------------------------------------------------
    # Step 9: Defensive post-conversion validation.
    # ------------------------------------------------------------------
    if raw_weights:
        if max(raw_weights.values()) > max_weight_dec:
            return _all_cash_blocked()
        total_invested = sum(raw_weights.values())
        if total_invested > max_invested_dec:
            return _all_cash_blocked()
        cash_weight = Decimal("1") - total_invested
        if cash_weight < limits.minimum_cash_weight:
            return _all_cash_blocked()
    else:
        # All weights quantized to zero → valid flat (no position worth holding).
        return _all_cash_flat()

    return OptimizationResult(
        weights=raw_weights,
        cash_weight=Decimal("1") - sum(raw_weights.values()),
        blocked=False,
    )
