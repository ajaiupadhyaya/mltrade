"""Tests for ExecutionIntent and build_intent.

Core invariants
---------------
1. Same inputs → same ``client_order_id`` (idempotency / dedup).
2. Any input difference → different ``client_order_id``.
3. ID format: ``mlt-YYYYMMDD-<24-hex>``, total length <= 48.
4. Cross-process stability: sha256, not builtin ``hash()``.
5. ``ExecutionIntent`` is frozen (no mutation).
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from datetime import date
from decimal import Decimal

import pytest

from mltrade.execution import ExecutionIntent, OrderSide, build_intent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS: dict[str, object] = dict(
    environment="paper",
    strategy_version="ridge-trend-v1",
    decision_session=date(2026, 6, 12),
    symbol="SPY",
    side=OrderSide.BUY,
    target_quantity=Decimal("10"),
)


def _make(**overrides: object) -> ExecutionIntent:
    kw = dict(_DEFAULT_KWARGS)
    kw.update(overrides)
    return build_intent(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identity / idempotency
# ---------------------------------------------------------------------------


def test_execution_identity_is_stable() -> None:
    """Same arguments always produce the same client_order_id."""
    a = _make()
    b = _make()
    assert a.client_order_id == b.client_order_id


def test_client_order_id_length_le_48() -> None:
    """client_order_id length must fit within 48 characters."""
    intent = _make()
    assert len(intent.client_order_id) <= 48


def test_client_order_id_format() -> None:
    """Format: mlt-YYYYMMDD-<24 lowercase hex chars>."""
    intent = _make()
    pattern = re.compile(r"^mlt-\d{8}-[0-9a-f]{24}$")
    assert pattern.fullmatch(intent.client_order_id), (
        f"client_order_id {intent.client_order_id!r} does not match expected format"
    )


# ---------------------------------------------------------------------------
# Collision tests — each dimension must produce a different ID
# ---------------------------------------------------------------------------


def test_different_symbol_gives_different_id() -> None:
    """Different symbols must produce different client_order_ids."""
    spy = _make(symbol="SPY")
    qqq = _make(symbol="QQQ")
    assert spy.client_order_id != qqq.client_order_id


def test_different_side_gives_different_id() -> None:
    """BUY vs SELL must produce different client_order_ids."""
    buy = _make(side=OrderSide.BUY)
    sell = _make(side=OrderSide.SELL)
    assert buy.client_order_id != sell.client_order_id


def test_different_date_gives_different_id() -> None:
    """Different decision_session dates must produce different client_order_ids."""
    d1 = _make(decision_session=date(2026, 6, 12))
    d2 = _make(decision_session=date(2026, 6, 13))
    assert d1.client_order_id != d2.client_order_id


def test_different_quantity_gives_different_id() -> None:
    """Different target_quantity values must produce different client_order_ids."""
    q10 = _make(target_quantity=Decimal("10"))
    q11 = _make(target_quantity=Decimal("11"))
    assert q10.client_order_id != q11.client_order_id


def test_different_environment_gives_different_id() -> None:
    """Different environment strings must produce different client_order_ids."""
    paper = _make(environment="paper")
    live = _make(environment="live")
    assert paper.client_order_id != live.client_order_id


def test_different_strategy_version_gives_different_id() -> None:
    """Different strategy_version strings must produce different client_order_ids."""
    v1 = _make(strategy_version="ridge-trend-v1")
    v2 = _make(strategy_version="ridge-trend-v2")
    assert v1.client_order_id != v2.client_order_id


# ---------------------------------------------------------------------------
# Cross-process stability — proves sha256, not hash()
# ---------------------------------------------------------------------------


def test_cross_process_stability() -> None:
    """client_order_id must be identical in a separate Python process.

    If builtin ``hash()`` were used, ``PYTHONHASHSEED`` randomisation would
    cause the child process to produce a different digest.  This test is the
    definitive proof that we use sha256 instead.
    """
    # Compute the expected ID in this process
    expected = _make().client_order_id

    script = textwrap.dedent(
        """
        import sys
        from datetime import date
        from decimal import Decimal
        from mltrade.execution import OrderSide, build_intent

        intent = build_intent(
            environment="paper",
            strategy_version="ridge-trend-v1",
            decision_session=date(2026, 6, 12),
            symbol="SPY",
            side=OrderSide.BUY,
            target_quantity=Decimal("10"),
        )
        print(intent.client_order_id)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    child_id = result.stdout.strip()

    assert child_id == expected, (
        f"client_order_id differs across processes: "
        f"parent={expected!r}, child={child_id!r}.  "
        f"This indicates builtin hash() was used instead of sha256."
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_intent_is_frozen() -> None:
    """ExecutionIntent must reject field mutation."""
    intent = _make()
    with pytest.raises((TypeError, Exception)):
        intent.symbol = "QQQ"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_intent_fields_preserved() -> None:
    """All input fields must be accessible on the returned ExecutionIntent."""
    session = date(2026, 6, 12)
    intent = _make(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=session,
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    assert intent.environment == "paper"
    assert intent.strategy_version == "ridge-trend-v1"
    assert intent.decision_session == session
    assert intent.symbol == "SPY"
    assert intent.side is OrderSide.BUY
    assert intent.target_quantity == Decimal("10")
    assert intent.client_order_id.startswith("mlt-20260612-")
