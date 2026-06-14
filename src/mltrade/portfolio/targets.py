"""Value objects for portfolio target representation.

These are separated from optimizer.py so that downstream consumers (e.g. the
pre-trade risk policy in Task 9) can import the data types without pulling in
the cvxpy solver dependency.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PortfolioLimits(BaseModel):
    """Hard constraints passed to the portfolio optimizer.

    Attributes
    ----------
    maximum_position_weight:
        Maximum fraction of NAV that any single symbol may hold.
        Must be positive and satisfy
        ``maximum_position_weight <= 1 - minimum_cash_weight``.
    minimum_cash_weight:
        Minimum fraction of NAV that must remain as cash.  Must be positive.
    target_annual_volatility:
        Target annualised portfolio volatility (e.g. ``Decimal("0.15")`` for
        15 %).  Must be positive.
    """

    model_config = ConfigDict(frozen=True)

    maximum_position_weight: Decimal = Field(gt=0)
    minimum_cash_weight: Decimal = Field(gt=0)
    target_annual_volatility: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def _validate_weight_consistency(self) -> PortfolioLimits:
        if self.maximum_position_weight > Decimal("1") - self.minimum_cash_weight:
            raise ValueError(
                "maximum_position_weight cannot exceed 1 - minimum_cash_weight"
            )
        return self


class OptimizationResult(BaseModel):
    """Immutable portfolio target produced by
    :func:`~mltrade.portfolio.optimizer.build_target`.

    Attributes
    ----------
    weights:
        Mapping from symbol to target weight (0 < w <= maximum_position_weight).
        Empty when the result is all-cash (either blocked or no-signal flat).
    cash_weight:
        ``1 - sum(weights.values())``.  Always ``>= minimum_cash_weight``.
    blocked:
        ``True`` when the solver failed or produced an infeasible solution.
        ``False`` when weights represent a valid (possibly all-cash) target.

    Distinguishing no-signal from solver failure
    --------------------------------------------
    - No qualifying forecasts → ``blocked=False``, ``weights={}``,
      ``cash_weight=Decimal("1")``.  This is a valid flat position.
    - Solver failure / infeasibility → ``blocked=True``, ``weights={}``,
      ``cash_weight=Decimal("1")``.  Callers must treat this as a fail-closed
      signal and skip the rebalance.
    """

    model_config = ConfigDict(frozen=True)

    weights: dict[str, Decimal]
    cash_weight: Decimal
    blocked: bool
