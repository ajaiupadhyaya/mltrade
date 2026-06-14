"""Unit tests for backtest accounting module (Task 10).

Tests cover:
1. test_trade_cost_reduces_cash_and_equity -- buy 100 SPY @ $500, 5 bps
2. test_sell_fill_increases_cash_minus_cost -- sell 50 SPY @ $400, 5 bps
3. test_initial_state -- PortfolioState.initial() fields
4. test_mark_to_market -- equity = cash + holdings * prices
5. test_apply_fill_buy_updates_holdings -- buy updates holdings correctly
6. test_apply_fill_sell_updates_holdings -- sell updates holdings correctly
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from mltrade.backtest import (
    Fill,
    PortfolioState,
    apply_fill,
    mark_to_market,
)

# ---------------------------------------------------------------------------
# 3. Initial state
# ---------------------------------------------------------------------------


def test_initial_state() -> None:
    """PortfolioState.initial(1_000_000) has correct fields."""
    state = PortfolioState.initial(Decimal("1000000"))
    assert state.cash == Decimal("1000000")
    assert state.holdings == {}
    assert state.total_costs == Decimal("0")


# ---------------------------------------------------------------------------
# 1. Buy reduces cash and accumulates cost
# ---------------------------------------------------------------------------


def test_trade_cost_reduces_cash_and_equity() -> None:
    """Buy 100 SPY @ $500, 5 bps: cash should be $949,975 and total_costs=$25."""
    state = PortfolioState.initial(Decimal("1000000"))
    fill = Fill(symbol="SPY", quantity=100, price=Decimal("500"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))

    # notional = 100 * 500 = 50,000
    # cost = 50,000 * (5 / 10,000) = 25
    # cash = 1,000,000 - 50,000 - 25 = 949,975
    assert new_state.cash == Decimal("949975")
    assert new_state.total_costs == Decimal("25")


# ---------------------------------------------------------------------------
# 2. Sell increases cash (minus cost)
# ---------------------------------------------------------------------------


def test_sell_fill_increases_cash_minus_cost() -> None:
    """Sell 50 SPY @ $400, 5 bps: cash increases by 50*400 - cost."""
    # Start with 100 shares already held and enough cash
    state = PortfolioState(
        cash=Decimal("500000"),
        holdings={"SPY": 100},
        total_costs=Decimal("0"),
    )
    fill = Fill(symbol="SPY", quantity=-50, price=Decimal("400"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))

    # notional = 50 * 400 = 20,000
    # cost = 20,000 * (5 / 10,000) = 10
    # cash = 500,000 + 20,000 - 10 = 519,990
    assert new_state.cash == Decimal("519990")
    assert new_state.total_costs == Decimal("10")
    assert new_state.holdings["SPY"] == 50


# ---------------------------------------------------------------------------
# 4. Mark-to-market
# ---------------------------------------------------------------------------


def test_mark_to_market() -> None:
    """equity = cash + holdings * prices."""
    state = PortfolioState(
        cash=Decimal("500000"),
        holdings={"SPY": 100, "QQQ": 50},
        total_costs=Decimal("0"),
    )
    prices: dict[str, Decimal] = {
        "SPY": Decimal("450"),
        "QQQ": Decimal("300"),
    }
    equity = mark_to_market(state, prices)
    # 500,000 + 100*450 + 50*300 = 500,000 + 45,000 + 15,000 = 560,000
    assert equity == Decimal("560000")


# ---------------------------------------------------------------------------
# 5. Buy updates holdings
# ---------------------------------------------------------------------------


def test_apply_fill_buy_updates_holdings() -> None:
    """After buying 100 SPY, holdings['SPY'] == 100."""
    state = PortfolioState.initial(Decimal("1000000"))
    fill = Fill(symbol="SPY", quantity=100, price=Decimal("500"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))
    assert new_state.holdings.get("SPY") == 100


# ---------------------------------------------------------------------------
# 6. Sell updates holdings
# ---------------------------------------------------------------------------


def test_apply_fill_sell_updates_holdings() -> None:
    """After selling 30 of 100 SPY, holdings['SPY'] == 70."""
    state = PortfolioState(
        cash=Decimal("1000000"),
        holdings={"SPY": 100},
        total_costs=Decimal("0"),
    )
    fill = Fill(symbol="SPY", quantity=-30, price=Decimal("500"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))
    assert new_state.holdings.get("SPY") == 70


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_apply_fill_sell_removes_zero_holdings() -> None:
    """Selling all shares removes the symbol from holdings."""
    state = PortfolioState(
        cash=Decimal("1000000"),
        holdings={"SPY": 50},
        total_costs=Decimal("0"),
    )
    fill = Fill(symbol="SPY", quantity=-50, price=Decimal("400"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))
    assert "SPY" not in new_state.holdings


def test_apply_fill_buy_existing_adds_to_holdings() -> None:
    """Buying more of an existing position adds to existing count."""
    state = PortfolioState(
        cash=Decimal("1000000"),
        holdings={"SPY": 50},
        total_costs=Decimal("0"),
    )
    fill = Fill(symbol="SPY", quantity=30, price=Decimal("500"))
    new_state = apply_fill(state, fill, transaction_cost_bps=Decimal("5"))
    assert new_state.holdings.get("SPY") == 80


def test_portfolio_state_is_frozen() -> None:
    """PortfolioState is an immutable frozen Pydantic model."""
    state = PortfolioState.initial(Decimal("1000000"))
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        state.cash = Decimal("999")  # type: ignore[misc]


def test_fill_is_frozen() -> None:
    """Fill is an immutable frozen Pydantic model."""
    fill = Fill(symbol="SPY", quantity=100, price=Decimal("500"))
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        fill.quantity = 200  # type: ignore[misc]
