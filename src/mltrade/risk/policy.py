"""Deterministic, fail-closed pre-trade risk policy.

``evaluate_pre_trade`` ALWAYS emits a :class:`~mltrade.risk.checks.RiskCheck`
for EVERY rule, in a fixed order, regardless of earlier results.  This means:

- ``RiskReport.by_code(code)`` always succeeds for a known code.
- The report is complete evidence: every gate is accounted for.
- ``RiskReport.blocked`` is authoritative: any BLOCK status halts execution.

Check codes (stable, in emission order)
----------------------------------------
1.  ``snapshot_health``            — snapshot not blocked / identity matches.
2.  ``snapshot_freshness``         — snapshot_last_session == expected_last_session.
3.  ``decision_session_freshness`` — decision_session == expected_decision_session.
4.  ``model_version``              — model_version == expected_model_version.
5.  ``feature_version``            — feature_version == expected_feature_version.
6.  ``finite_values``              — all numeric inputs finite.
7.  ``position_limit``             — max(|weight|) <= maximum_position_weight.
8.  ``gross_exposure``             — sum(|weight|) <= max_gross (1.0).
9.  ``net_exposure``               — 0 <= net_weight <= 1 (long-only).
10. ``cash_reserve``               — cash_weight >= minimum_cash_weight.
11. ``per_order_notional``         — each |order notional| <= max_order_weight * equity.
12. ``total_rebalance_notional``   — sum(|notional|) <= max_rebalance_weight * equity.
13. ``minimum_order_notional``     — no order has notional below min_order_notional.
14. ``duplicate_intent``           — no duplicate execution intent client IDs.
15. ``broker_account_status``      — paper account active, not trading-blocked.
16. ``reconciliation``             — broker vs internal state agrees.
17. ``live_trading_disabled``      — live_trading_enabled is False.

``PreTradeContext`` vs. persisted frozen artifacts
---------------------------------------------------
Persisted artifacts (``Forecast``, ``OptimizationResult``, ``DailyBar``, etc.)
override ``model_copy`` to reject any ``update`` argument, enforcing that
persisted records are never silently mutated.

``PreTradeContext`` is a *transient input context* assembled immediately before
evaluation.  It is NOT stored in the database or snapshot store, and its
lifecycle ends as soon as the policy returns.  Allowing ``model_copy(update=…)``
(standard Pydantic frozen behaviour) is:

- Required by the test suite (``valid_context.model_copy(update={...})``).
- Safe: the copy is ephemeral and never replaces an authoritative artifact.
- Appropriate: the context is a parameter bag, not a persisted fact.

Therefore ``PreTradeContext`` is frozen (immutable in-place) but does NOT add
the update-rejecting ``model_copy`` override present on persisted types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from mltrade.risk.checks import CheckStatus, RiskCheck, RiskReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_GROSS_EXPOSURE = Decimal("1")
_MIN_NET_EXPOSURE = Decimal("0")
_MAX_NET_EXPOSURE = Decimal("1")


# ---------------------------------------------------------------------------
# Input context
# ---------------------------------------------------------------------------


class PreTradeContext(BaseModel):
    """Transient input context for :func:`evaluate_pre_trade`.

    This model is frozen (fields are immutable once constructed) but does NOT
    override ``model_copy`` — callers may use ``model_copy(update={...})`` to
    produce modified copies for testing or scenario analysis.

    All weights are fractions of NAV (e.g. ``Decimal("0.20")`` = 20 %).
    All notionals are in dollars as :class:`~decimal.Decimal`.

    Snapshot / session provenance
    ------------------------------
    snapshot_blocked:
        True when the upstream data pipeline marked the snapshot as blocked
        (quality gate failed).  Triggers BLOCK on ``snapshot_health``.
    snapshot_last_session:
        The latest XNYS session present in the bar snapshot that produced the
        portfolio weights.  Must equal ``expected_last_session``.
    expected_last_session:
        The calendar session we expect the snapshot to cover.
    decision_session:
        The XNYS session on which the forecasts and weights were generated.
    expected_decision_session:
        The session we expect decisions to have been made for.

    Model / feature versioning
    --------------------------
    model_version:
        Actual model version string embedded in the forecast batch.
    expected_model_version:
        Required value (e.g. ``"ridge-trend-v1"``).
    feature_version:
        Actual feature version string.
    expected_feature_version:
        Required value (e.g. ``"trend-momentum-v1"``).

    Portfolio weights
    ------------------
    weights:
        Symbol → target weight (Decimal).  Must not include cash.
    cash_weight:
        Cash allocation as fraction of NAV.  Must be >= minimum_cash_weight.

    Gross and net exposure are computed internally from ``weights``:
    - gross = sum(|w| for w in weights.values())
    - net   = sum(w for w in weights.values())   (long-only: == gross)

    Order intents
    -------------
    order_notionals:
        Mapping from a per-order client-ID to the signed notional ($) of that
        order.  Absolute value is used for limit checks.
    equity:
        Total portfolio equity in dollars (NAV).
    intent_client_ids:
        Tuple of all execution intent client IDs for the current batch.
        Duplicates trigger BLOCK on ``duplicate_intent``.

    Risk limits
    -----------
    maximum_position_weight:
        Hard cap on any single symbol weight.
    minimum_cash_weight:
        Minimum required cash fraction.
    maximum_order_weight:
        Maximum |notional| for a single order as fraction of equity.
    maximum_rebalance_weight:
        Maximum sum(|notional|) for the entire rebalance batch as fraction of equity.
    minimum_order_notional:
        Orders with |notional| < this value must NOT appear in the batch
        (they should have been filtered upstream).

    Broker / account state
    ----------------------
    broker_account_active:
        True when the paper account is in active / open status.
    broker_account_blocked:
        True when Alpaca has flagged the account as trading-blocked.
    reconciliation_ok:
        True when internal state and broker state agree within tolerance.

    Safety
    ------
    live_trading_enabled:
        Must always be False.  True → BLOCK on ``live_trading_disabled``.
    """

    model_config = ConfigDict(frozen=True)

    # Snapshot / session provenance
    snapshot_blocked: bool
    snapshot_last_session: date
    expected_last_session: date
    decision_session: date
    expected_decision_session: date

    # Model / feature versioning
    model_version: str
    expected_model_version: str
    feature_version: str
    expected_feature_version: str

    # Portfolio weights (symbol → weight, excluding cash)
    weights: Mapping[str, Decimal]
    cash_weight: Decimal

    # Order intents
    order_notionals: Mapping[str, Decimal]  # client_id → signed notional
    equity: Decimal
    intent_client_ids: tuple[str, ...]

    # Risk limits
    maximum_position_weight: Decimal
    minimum_cash_weight: Decimal
    maximum_order_weight: Decimal
    maximum_rebalance_weight: Decimal
    minimum_order_notional: Decimal

    # Broker / account state
    broker_account_active: bool
    broker_account_blocked: bool
    reconciliation_ok: bool

    # Safety
    live_trading_enabled: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pass(code: str, message: str) -> RiskCheck:
    return RiskCheck(code=code, status=CheckStatus.PASS, message=message)


def _warn(code: str, message: str, detail: str = "") -> RiskCheck:
    return RiskCheck(
        code=code, status=CheckStatus.WARN, message=message, detail=detail
    )


def _block(code: str, message: str, detail: str = "") -> RiskCheck:
    return RiskCheck(
        code=code, status=CheckStatus.BLOCK, message=message, detail=detail
    )


def _is_finite_decimal(value: Decimal) -> bool:
    try:
        fv = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(fv)


# ---------------------------------------------------------------------------
# Individual rule evaluators (one function per check code)
# Each returns exactly one RiskCheck.
# ---------------------------------------------------------------------------


def _check_snapshot_health(ctx: PreTradeContext) -> RiskCheck:
    code = "snapshot_health"
    if ctx.snapshot_blocked:
        return _block(code, "Upstream snapshot is marked blocked by the data pipeline.")
    return _pass(code, "Snapshot is healthy (not blocked).")


def _check_snapshot_freshness(ctx: PreTradeContext) -> RiskCheck:
    code = "snapshot_freshness"
    if ctx.snapshot_last_session != ctx.expected_last_session:
        return _block(
            code,
            "Snapshot last session does not match expected last session.",
            detail=(
                f"snapshot_last_session={ctx.snapshot_last_session}, "
                f"expected={ctx.expected_last_session}"
            ),
        )
    return _pass(
        code,
        f"Snapshot covers expected last session {ctx.expected_last_session}.",
    )


def _check_decision_session_freshness(ctx: PreTradeContext) -> RiskCheck:
    code = "decision_session_freshness"
    if ctx.decision_session != ctx.expected_decision_session:
        return _block(
            code,
            "Decision session does not match expected decision session.",
            detail=(
                f"decision_session={ctx.decision_session}, "
                f"expected={ctx.expected_decision_session}"
            ),
        )
    return _pass(
        code,
        f"Decision session matches expected {ctx.expected_decision_session}.",
    )


def _check_model_version(ctx: PreTradeContext) -> RiskCheck:
    code = "model_version"
    if ctx.model_version != ctx.expected_model_version:
        return _block(
            code,
            "Model version mismatch.",
            detail=(
                f"actual={ctx.model_version!r}, "
                f"expected={ctx.expected_model_version!r}"
            ),
        )
    return _pass(code, f"Model version matches {ctx.expected_model_version!r}.")


def _check_feature_version(ctx: PreTradeContext) -> RiskCheck:
    code = "feature_version"
    if ctx.feature_version != ctx.expected_feature_version:
        return _block(
            code,
            "Feature version mismatch.",
            detail=(
                f"actual={ctx.feature_version!r}, "
                f"expected={ctx.expected_feature_version!r}"
            ),
        )
    return _pass(code, f"Feature version matches {ctx.expected_feature_version!r}.")


def _check_finite_values(ctx: PreTradeContext) -> RiskCheck:
    """Verify all prices, weights, notionals, and equity are finite."""
    code = "finite_values"
    bad: list[str] = []

    if not _is_finite_decimal(ctx.equity):
        bad.append(f"equity={ctx.equity!r}")
    if not _is_finite_decimal(ctx.cash_weight):
        bad.append(f"cash_weight={ctx.cash_weight!r}")

    for sym, w in ctx.weights.items():
        if not _is_finite_decimal(w):
            bad.append(f"weights[{sym!r}]={w!r}")

    for cid, notional in ctx.order_notionals.items():
        if not _is_finite_decimal(notional):
            bad.append(f"order_notionals[{cid!r}]={notional!r}")

    if bad:
        return _block(
            code,
            "Non-finite numeric values detected.",
            detail="; ".join(bad),
        )
    return _pass(code, "All numeric values are finite.")


def _check_position_limit(ctx: PreTradeContext) -> RiskCheck:
    code = "position_limit"
    # Non-finite weights are covered by finite_values, but since all checks
    # always run we must guard against NaN Decimal raising on comparison.
    if not all(_is_finite_decimal(w) for w in ctx.weights.values()):
        return _block(
            code,
            "Cannot evaluate position limit: one or more weights are non-finite.",
        )
    violations: list[str] = []
    for sym, w in ctx.weights.items():
        if abs(w) > ctx.maximum_position_weight:
            violations.append(
                f"{sym}: |{w}| > {ctx.maximum_position_weight}"
            )
    if violations:
        return _block(
            code,
            "One or more position weights exceed the per-symbol limit.",
            detail="; ".join(violations),
        )
    return _pass(
        code,
        f"All positions within limit ({ctx.maximum_position_weight}).",
    )


def _check_gross_exposure(ctx: PreTradeContext) -> RiskCheck:
    code = "gross_exposure"
    if not all(_is_finite_decimal(w) for w in ctx.weights.values()):
        return _block(
            code,
            "Cannot evaluate gross exposure: one or more weights are non-finite.",
        )
    gross = sum(abs(w) for w in ctx.weights.values())
    if gross > _MAX_GROSS_EXPOSURE:
        return _block(
            code,
            "Gross exposure exceeds 1.0.",
            detail=f"gross={gross}",
        )
    return _pass(code, f"Gross exposure {gross} <= {_MAX_GROSS_EXPOSURE}.")


def _check_net_exposure(ctx: PreTradeContext) -> RiskCheck:
    code = "net_exposure"
    if not all(_is_finite_decimal(w) for w in ctx.weights.values()):
        return _block(
            code,
            "Cannot evaluate net exposure: one or more weights are non-finite.",
        )
    net = sum(ctx.weights.values())
    if net < _MIN_NET_EXPOSURE or net > _MAX_NET_EXPOSURE:
        return _block(
            code,
            f"Net exposure {net} is outside long-only range "
            f"[{_MIN_NET_EXPOSURE}, {_MAX_NET_EXPOSURE}].",
            detail=f"net={net}",
        )
    return _pass(code, f"Net exposure {net} within long-only bounds.")


def _check_cash_reserve(ctx: PreTradeContext) -> RiskCheck:
    code = "cash_reserve"
    # Guard against non-finite cash_weight (caught by finite_values, but all
    # checks run regardless — comparison with a NaN Decimal raises).
    if not _is_finite_decimal(ctx.cash_weight):
        return _block(
            code,
            "Cash weight is non-finite; cannot evaluate reserve requirement.",
            detail=f"cash_weight={ctx.cash_weight!r}",
        )
    if ctx.cash_weight < ctx.minimum_cash_weight:
        return _block(
            code,
            f"Cash weight {ctx.cash_weight} is below minimum "
            f"{ctx.minimum_cash_weight}.",
            detail=f"cash_weight={ctx.cash_weight}",
        )
    return _pass(
        code,
        f"Cash reserve {ctx.cash_weight} >= minimum {ctx.minimum_cash_weight}.",
    )


def _check_per_order_notional(ctx: PreTradeContext) -> RiskCheck:
    code = "per_order_notional"
    # Guard: non-finite equity or notionals are caught by finite_values, but
    # since all checks always run we must avoid NaN Decimal raising on
    # comparison.
    if not _is_finite_decimal(ctx.equity):
        return _block(
            code,
            "Cannot evaluate per-order notional limit: equity is non-finite.",
        )
    if not all(_is_finite_decimal(n) for n in ctx.order_notionals.values()):
        return _block(
            code,
            "Cannot evaluate per-order notional limit: "
            "one or more order notionals are non-finite.",
        )
    max_notional = ctx.maximum_order_weight * ctx.equity
    violations: list[str] = []
    for cid, notional in ctx.order_notionals.items():
        if abs(notional) > max_notional:
            violations.append(f"{cid}: |{notional}| > {max_notional}")
    if violations:
        return _block(
            code,
            "One or more orders exceed the per-order notional limit.",
            detail="; ".join(violations),
        )
    return _pass(
        code,
        f"All order notionals <= {max_notional} "
        f"({ctx.maximum_order_weight} x {ctx.equity}).",
    )


def _check_total_rebalance_notional(ctx: PreTradeContext) -> RiskCheck:
    code = "total_rebalance_notional"
    if not _is_finite_decimal(ctx.equity):
        return _block(
            code,
            "Cannot evaluate total rebalance notional: equity is non-finite.",
        )
    if not all(_is_finite_decimal(n) for n in ctx.order_notionals.values()):
        return _block(
            code,
            "Cannot evaluate total rebalance notional: "
            "one or more order notionals are non-finite.",
        )
    max_total = ctx.maximum_rebalance_weight * ctx.equity
    total = sum(abs(n) for n in ctx.order_notionals.values())
    if total > max_total:
        return _block(
            code,
            f"Total rebalance notional {total} exceeds limit {max_total}.",
            detail=f"total={total}, limit={max_total}",
        )
    return _pass(
        code,
        f"Total rebalance notional {total} <= limit {max_total}.",
    )


def _check_minimum_order_notional(ctx: PreTradeContext) -> RiskCheck:
    """Verify no sub-minimum order appears in the batch.

    Sub-minimum orders should have been filtered upstream.  If one reaches
    the gate, it is an upstream defect and must BLOCK.
    """
    code = "minimum_order_notional"
    if not all(_is_finite_decimal(n) for n in ctx.order_notionals.values()):
        return _block(
            code,
            "Cannot evaluate minimum order notional: "
            "one or more order notionals are non-finite.",
        )
    violations: list[str] = []
    for cid, notional in ctx.order_notionals.items():
        if abs(notional) < ctx.minimum_order_notional:
            violations.append(
                f"{cid}: |{notional}| < {ctx.minimum_order_notional}"
            )
    if violations:
        return _block(
            code,
            "Sub-minimum notional order(s) reached the pre-trade gate "
            "(should have been filtered upstream).",
            detail="; ".join(violations),
        )
    return _pass(
        code,
        f"All orders at or above minimum notional {ctx.minimum_order_notional}.",
    )


def _check_duplicate_intent(ctx: PreTradeContext) -> RiskCheck:
    code = "duplicate_intent"
    seen: set[str] = set()
    duplicates: list[str] = []
    for cid in ctx.intent_client_ids:
        if cid in seen:
            duplicates.append(cid)
        else:
            seen.add(cid)
    if duplicates:
        return _block(
            code,
            "Duplicate execution intent client IDs detected.",
            detail=f"duplicates={duplicates}",
        )
    return _pass(code, "No duplicate execution intent client IDs.")


def _check_broker_account_status(ctx: PreTradeContext) -> RiskCheck:
    code = "broker_account_status"
    if not ctx.broker_account_active:
        return _block(
            code,
            "Broker paper account is not active.",
        )
    if ctx.broker_account_blocked:
        return _block(
            code,
            "Broker paper account is trading-blocked.",
        )
    return _pass(code, "Broker paper account is active and not blocked.")


def _check_reconciliation(ctx: PreTradeContext) -> RiskCheck:
    code = "reconciliation"
    if not ctx.reconciliation_ok:
        return _block(
            code,
            "Broker vs internal state reconciliation failed.",
        )
    return _pass(code, "Broker and internal state reconcile successfully.")


def _check_live_trading_disabled(ctx: PreTradeContext) -> RiskCheck:
    code = "live_trading_disabled"
    if ctx.live_trading_enabled:
        return _block(
            code,
            "live_trading_enabled is True — live trading is not permitted in "
            "this release.  This gate must never be reached in production.",
        )
    return _pass(code, "Live trading is disabled (paper-only mode).")


# ---------------------------------------------------------------------------
# Ordered rule registry
# ---------------------------------------------------------------------------

# Each entry is a callable that accepts PreTradeContext and returns RiskCheck.
# The order here defines the deterministic emission order of the report.
_RULES = (
    _check_snapshot_health,
    _check_snapshot_freshness,
    _check_decision_session_freshness,
    _check_model_version,
    _check_feature_version,
    _check_finite_values,
    _check_position_limit,
    _check_gross_exposure,
    _check_net_exposure,
    _check_cash_reserve,
    _check_per_order_notional,
    _check_total_rebalance_notional,
    _check_minimum_order_notional,
    _check_duplicate_intent,
    _check_broker_account_status,
    _check_reconciliation,
    _check_live_trading_disabled,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_pre_trade(context: PreTradeContext) -> RiskReport:
    """Evaluate all pre-trade risk rules and return an authoritative report.

    Every rule is evaluated regardless of earlier results — the report is
    always complete.  ``RiskReport.blocked`` is ``True`` iff any check has
    status :attr:`~mltrade.risk.checks.CheckStatus.BLOCK`.

    Parameters
    ----------
    context:
        Fully populated :class:`PreTradeContext` for the rebalance attempt.

    Returns
    -------
    RiskReport
        Immutable, deterministically ordered collection of risk checks.
    """
    checks = tuple(rule(context) for rule in _RULES)
    return RiskReport(checks=checks)
