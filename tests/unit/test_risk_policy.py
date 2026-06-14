"""Tests for the pre-trade risk policy (Task 9).

Coverage
--------
1.  ``valid_context`` passes all checks (no BLOCK, ``report.blocked`` is False).
2.  ``by_code`` finds every check code for the valid context.
3.  WARN does not cause ``report.blocked``.
4.  Each of the 17 check codes independently blocks when its precondition
    is violated:
    - snapshot_health
    - snapshot_freshness
    - decision_session_freshness
    - model_version
    - feature_version
    - finite_values (weight, notional, cash_weight, equity)
    - position_limit
    - gross_exposure
    - net_exposure
    - cash_reserve
    - per_order_notional
    - total_rebalance_notional
    - minimum_order_notional
    - duplicate_intent
    - broker_account_status (not active / blocked)
    - reconciliation
    - live_trading_disabled
5.  ``by_code`` raises ``KeyError`` for an unknown code.
6.  ``model_copy(update=...)`` works on ``PreTradeContext`` (frozen but
    not update-rejecting — required by design and these tests).
7.  All checks are always emitted (exactly 17 entries in ``report.checks``).
8.  Deterministic ordering: same context → same check order.
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import pytest

from mltrade.risk import CheckStatus, evaluate_pre_trade
from mltrade.risk.policy import PreTradeContext

# ---------------------------------------------------------------------------
# Shared valid context fixture
# ---------------------------------------------------------------------------

_TODAY = date(2026, 6, 12)
_YESTERDAY = date(2026, 6, 11)

# A fully-healthy context that must pass ALL checks with status PASS or WARN
# (never BLOCK).
valid_context = PreTradeContext(
    # Snapshot / session provenance
    snapshot_blocked=False,
    snapshot_last_session=_TODAY,
    expected_last_session=_TODAY,
    decision_session=_TODAY,
    expected_decision_session=_TODAY,
    # Model / feature versioning
    model_version="ridge-trend-v1",
    expected_model_version="ridge-trend-v1",
    feature_version="trend-momentum-v1",
    expected_feature_version="trend-momentum-v1",
    # Portfolio weights: two symbols totalling 0.30; cash = 0.70
    weights={
        "SPY": Decimal("0.15"),
        "QQQ": Decimal("0.15"),
    },
    cash_weight=Decimal("0.70"),
    # Order intents: two orders, each $50 000 (0.05 * $1 000 000)
    order_notionals={
        "intent-spy-001": Decimal("50000"),
        "intent-qqq-001": Decimal("50000"),
    },
    equity=Decimal("1000000"),
    intent_client_ids=("intent-spy-001", "intent-qqq-001"),
    # Risk limits (matching config defaults)
    maximum_position_weight=Decimal("0.25"),
    minimum_cash_weight=Decimal("0.05"),
    maximum_order_weight=Decimal("0.10"),
    maximum_rebalance_weight=Decimal("0.50"),
    minimum_order_notional=Decimal("500"),
    # Broker / account state
    broker_account_active=True,
    broker_account_blocked=False,
    reconciliation_ok=True,
    # Safety
    live_trading_enabled=False,
)

# Total number of rules/checks we expect in every report.
_EXPECTED_CHECK_COUNT = 17


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _flip(field: str, value: object) -> PreTradeContext:
    """Return a copy of valid_context with one field changed."""
    return valid_context.model_copy(update={field: value})


# ---------------------------------------------------------------------------
# 1. Valid context — all checks pass
# ---------------------------------------------------------------------------


def test_all_checks_pass_for_valid_preview() -> None:
    report = evaluate_pre_trade(context=valid_context)
    assert report.blocked is False
    assert all(check.status is not CheckStatus.BLOCK for check in report.checks)


def test_valid_context_emits_all_checks() -> None:
    report = evaluate_pre_trade(context=valid_context)
    assert len(report.checks) == _EXPECTED_CHECK_COUNT


def test_valid_context_all_checks_are_pass() -> None:
    """Every check on a healthy context must be PASS (no WARN expected here)."""
    report = evaluate_pre_trade(context=valid_context)
    for check in report.checks:
        assert check.status is CheckStatus.PASS, (
            f"Expected PASS for {check.code!r}, got {check.status}: {check.message}"
        )


def test_report_is_deterministic() -> None:
    """Same context → identical check codes in identical order."""
    r1 = evaluate_pre_trade(context=valid_context)
    r2 = evaluate_pre_trade(context=valid_context)
    assert [c.code for c in r1.checks] == [c.code for c in r2.checks]


# ---------------------------------------------------------------------------
# 2. by_code finds every expected check
# ---------------------------------------------------------------------------

_EXPECTED_CODES = [
    "snapshot_health",
    "snapshot_freshness",
    "decision_session_freshness",
    "model_version",
    "feature_version",
    "finite_values",
    "position_limit",
    "gross_exposure",
    "net_exposure",
    "cash_reserve",
    "per_order_notional",
    "total_rebalance_notional",
    "minimum_order_notional",
    "duplicate_intent",
    "broker_account_status",
    "reconciliation",
    "live_trading_disabled",
]


@pytest.mark.parametrize("code", _EXPECTED_CODES)
def test_by_code_finds_every_check(code: str) -> None:
    report = evaluate_pre_trade(context=valid_context)
    check = report.by_code(code)
    assert check.code == code


def test_by_code_raises_for_unknown_code() -> None:
    report = evaluate_pre_trade(context=valid_context)
    with pytest.raises(KeyError):
        report.by_code("no_such_check_code_xyz")


# ---------------------------------------------------------------------------
# 3. WARN does not block
# ---------------------------------------------------------------------------


def test_warn_does_not_block() -> None:
    """Synthesise a WARN check to verify it doesn't set blocked=True."""
    from mltrade.risk.checks import RiskCheck, RiskReport

    report = RiskReport(
        checks=(
            RiskCheck(code="c1", status=CheckStatus.PASS, message="ok"),
            RiskCheck(code="c2", status=CheckStatus.WARN, message="be careful"),
        )
    )
    assert report.blocked is False


# ---------------------------------------------------------------------------
# 4.  model_copy(update=...) works on PreTradeContext
# ---------------------------------------------------------------------------


def test_pre_trade_context_model_copy_update_works() -> None:
    """PreTradeContext is frozen but allows model_copy(update=...).

    This is the key design difference from persisted artifacts: PreTradeContext
    is a transient parameter bag, not a stored fact.
    """
    ctx2 = valid_context.model_copy(update={"snapshot_last_session": _YESTERDAY})
    assert ctx2.snapshot_last_session == _YESTERDAY
    # Original is unchanged.
    assert valid_context.snapshot_last_session == _TODAY


# ---------------------------------------------------------------------------
# 5.  Individual BLOCK checks (plan tests + one per remaining check)
# ---------------------------------------------------------------------------


# --- snapshot_freshness (plan-mandated test) --------------------------------


def test_stale_snapshot_blocks_submission() -> None:
    """Plan-mandated test: stale snapshot_last_session → BLOCK."""
    report = evaluate_pre_trade(
        context=valid_context.model_copy(
            update={"snapshot_last_session": date(2026, 6, 11)}
        )
    )
    assert report.blocked is True
    assert report.by_code("snapshot_freshness").status is CheckStatus.BLOCK


# --- snapshot_health --------------------------------------------------------


def test_snapshot_health_blocks_when_upstream_blocked() -> None:
    report = evaluate_pre_trade(context=_flip("snapshot_blocked", True))
    assert report.blocked is True
    assert report.by_code("snapshot_health").status is CheckStatus.BLOCK


# --- decision_session_freshness ---------------------------------------------


def test_stale_decision_session_blocks() -> None:
    report = evaluate_pre_trade(context=_flip("decision_session", _YESTERDAY))
    assert report.blocked is True
    assert report.by_code("decision_session_freshness").status is CheckStatus.BLOCK


# --- model_version ----------------------------------------------------------


def test_wrong_model_version_blocks() -> None:
    report = evaluate_pre_trade(context=_flip("model_version", "ridge-trend-v0"))
    assert report.blocked is True
    assert report.by_code("model_version").status is CheckStatus.BLOCK


# --- feature_version --------------------------------------------------------


def test_wrong_feature_version_blocks() -> None:
    report = evaluate_pre_trade(
        context=_flip("feature_version", "trend-momentum-v0")
    )
    assert report.blocked is True
    assert report.by_code("feature_version").status is CheckStatus.BLOCK


# --- finite_values ----------------------------------------------------------


def test_non_finite_weight_blocks() -> None:
    report = evaluate_pre_trade(
        context=_flip(
            "weights",
            {"SPY": Decimal("0.15"), "QQQ": Decimal(math.inf)},
        )
    )
    assert report.blocked is True
    assert report.by_code("finite_values").status is CheckStatus.BLOCK


def test_non_finite_notional_blocks() -> None:
    report = evaluate_pre_trade(
        context=_flip(
            "order_notionals",
            {
                "intent-spy-001": Decimal("50000"),
                "intent-qqq-001": Decimal(-math.inf),
            },
        )
    )
    assert report.blocked is True
    assert report.by_code("finite_values").status is CheckStatus.BLOCK


def test_non_finite_cash_weight_blocks() -> None:
    report = evaluate_pre_trade(context=_flip("cash_weight", Decimal(math.nan)))
    assert report.blocked is True
    assert report.by_code("finite_values").status is CheckStatus.BLOCK


def test_non_finite_equity_blocks() -> None:
    report = evaluate_pre_trade(context=_flip("equity", Decimal(math.inf)))
    assert report.blocked is True
    assert report.by_code("finite_values").status is CheckStatus.BLOCK


# --- position_limit ---------------------------------------------------------


def test_position_limit_blocks_when_weight_too_large() -> None:
    # SPY at 0.30 > maximum_position_weight 0.25
    ctx = valid_context.model_copy(
        update={
            "weights": {"SPY": Decimal("0.30"), "QQQ": Decimal("0.10")},
            "cash_weight": Decimal("0.60"),
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("position_limit").status is CheckStatus.BLOCK


# --- gross_exposure ---------------------------------------------------------


def test_gross_exposure_blocks_when_exceeds_1() -> None:
    # weights sum to > 1; cash_weight left at 0.70 (internal gross check uses weights)
    ctx = valid_context.model_copy(
        update={
            "weights": {
                "SPY": Decimal("0.25"),
                "QQQ": Decimal("0.25"),
                "IWM": Decimal("0.25"),
                "DIA": Decimal("0.25"),
                "GLD": Decimal("0.10"),  # total = 1.10
            },
            "cash_weight": Decimal("0.05"),
            "order_notionals": {
                "intent-spy-001": Decimal("50000"),
                "intent-qqq-001": Decimal("50000"),
            },
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("gross_exposure").status is CheckStatus.BLOCK


# --- net_exposure -----------------------------------------------------------


def test_net_exposure_blocks_when_negative() -> None:
    # Negative weight violates long-only constraint → net < 0
    ctx = valid_context.model_copy(
        update={
            "weights": {"SPY": Decimal("-0.10"), "QQQ": Decimal("0.05")},
            "cash_weight": Decimal("0.70"),
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("net_exposure").status is CheckStatus.BLOCK


# --- cash_reserve -----------------------------------------------------------


def test_cash_reserve_blocks_when_below_minimum() -> None:
    # cash_weight 0.02 < minimum_cash_weight 0.05
    report = evaluate_pre_trade(context=_flip("cash_weight", Decimal("0.02")))
    assert report.blocked is True
    assert report.by_code("cash_reserve").status is CheckStatus.BLOCK


# --- per_order_notional -----------------------------------------------------


def test_per_order_notional_blocks_when_order_too_large() -> None:
    # max_order_weight=0.10, equity=1_000_000 → max notional $100k
    # one order at $150k → BLOCK
    ctx = valid_context.model_copy(
        update={
            "order_notionals": {
                "intent-spy-001": Decimal("150000"),
                "intent-qqq-001": Decimal("50000"),
            },
            "intent_client_ids": ("intent-spy-001", "intent-qqq-001"),
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("per_order_notional").status is CheckStatus.BLOCK


# --- total_rebalance_notional -----------------------------------------------


def test_total_rebalance_notional_blocks_when_sum_too_large() -> None:
    # max_rebalance_weight=0.50, equity=1_000_000 → limit $500k
    # two orders at $300k each = $600k → BLOCK
    ctx = valid_context.model_copy(
        update={
            "order_notionals": {
                "intent-spy-001": Decimal("300000"),
                "intent-qqq-001": Decimal("300000"),
            },
            "intent_client_ids": ("intent-spy-001", "intent-qqq-001"),
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("total_rebalance_notional").status is CheckStatus.BLOCK


# --- minimum_order_notional -------------------------------------------------


def test_minimum_order_notional_blocks_when_sub_minimum_order_present() -> None:
    # An order at $200 < minimum_order_notional $500 → BLOCK
    ctx = valid_context.model_copy(
        update={
            "order_notionals": {
                "intent-spy-001": Decimal("50000"),
                "intent-qqq-001": Decimal("200"),  # sub-minimum
            },
            "intent_client_ids": ("intent-spy-001", "intent-qqq-001"),
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("minimum_order_notional").status is CheckStatus.BLOCK


# --- duplicate_intent -------------------------------------------------------


def test_duplicate_intent_blocks_when_client_ids_repeat() -> None:
    ctx = valid_context.model_copy(
        update={
            "intent_client_ids": (
                "intent-spy-001",
                "intent-qqq-001",
                "intent-spy-001",  # duplicate
            )
        }
    )
    report = evaluate_pre_trade(context=ctx)
    assert report.blocked is True
    assert report.by_code("duplicate_intent").status is CheckStatus.BLOCK


# --- broker_account_status (two failure modes) ------------------------------


def test_broker_account_status_blocks_when_not_active() -> None:
    report = evaluate_pre_trade(context=_flip("broker_account_active", False))
    assert report.blocked is True
    assert report.by_code("broker_account_status").status is CheckStatus.BLOCK


def test_broker_account_status_blocks_when_trading_blocked() -> None:
    report = evaluate_pre_trade(context=_flip("broker_account_blocked", True))
    assert report.blocked is True
    assert report.by_code("broker_account_status").status is CheckStatus.BLOCK


# --- reconciliation ---------------------------------------------------------


def test_reconciliation_blocks_when_mismatch() -> None:
    report = evaluate_pre_trade(context=_flip("reconciliation_ok", False))
    assert report.blocked is True
    assert report.by_code("reconciliation").status is CheckStatus.BLOCK


# --- live_trading_disabled --------------------------------------------------


def test_live_trading_disabled_blocks_when_live_enabled() -> None:
    report = evaluate_pre_trade(context=_flip("live_trading_enabled", True))
    assert report.blocked is True
    assert report.by_code("live_trading_disabled").status is CheckStatus.BLOCK


# ---------------------------------------------------------------------------
# 6.  Complete check coverage — every rule ALWAYS emits exactly one check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", _EXPECTED_CODES)
def test_every_check_always_emitted_even_when_blocked(code: str) -> None:
    """All 17 checks are emitted even when earlier checks block."""
    # Use a maximally-broken context: snapshot blocked + stale + wrong versions
    ctx = PreTradeContext(
        snapshot_blocked=True,
        snapshot_last_session=_YESTERDAY,
        expected_last_session=_TODAY,
        decision_session=_YESTERDAY,
        expected_decision_session=_TODAY,
        model_version="bad-v0",
        expected_model_version="ridge-trend-v1",
        feature_version="bad-v0",
        expected_feature_version="trend-momentum-v1",
        weights={},
        cash_weight=Decimal("0"),
        order_notionals={},
        equity=Decimal("1000000"),
        intent_client_ids=(),
        maximum_position_weight=Decimal("0.25"),
        minimum_cash_weight=Decimal("0.05"),
        maximum_order_weight=Decimal("0.10"),
        maximum_rebalance_weight=Decimal("0.50"),
        minimum_order_notional=Decimal("500"),
        broker_account_active=False,
        broker_account_blocked=True,
        reconciliation_ok=False,
        live_trading_enabled=True,
    )
    report = evaluate_pre_trade(context=ctx)
    assert len(report.checks) == _EXPECTED_CHECK_COUNT
    # All named codes are present.
    check = report.by_code(code)
    assert check.code == code


# ---------------------------------------------------------------------------
# 7.  Structural / type checks
# ---------------------------------------------------------------------------


def test_report_checks_is_tuple() -> None:
    report = evaluate_pre_trade(context=valid_context)
    assert isinstance(report.checks, tuple)


def test_risk_check_has_code_status_message() -> None:
    report = evaluate_pre_trade(context=valid_context)
    for check in report.checks:
        assert isinstance(check.code, str)
        assert isinstance(check.status, CheckStatus)
        assert isinstance(check.message, str)


def test_blocked_false_when_no_blocks() -> None:
    report = evaluate_pre_trade(context=valid_context)
    assert report.blocked is False


def test_blocked_true_when_any_block_present() -> None:
    # single BLOCK should flip the flag
    report = evaluate_pre_trade(context=_flip("reconciliation_ok", False))
    assert report.blocked is True
