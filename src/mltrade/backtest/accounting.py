"""Backtest portfolio accounting with Decimal precision.

Provides immutable portfolio state and fill application for backtesting.
All monetary values use Decimal to avoid floating-point accumulation errors.

Design
------
- ``PortfolioState`` is a frozen Pydantic model capturing cash, holdings, and
  total accumulated transaction costs at a point in time.
- ``Fill`` is a frozen Pydantic model representing a single trade execution.
- ``apply_fill`` returns a new ``PortfolioState`` after applying a fill with a
  given transaction cost in basis points.
- ``mark_to_market`` computes total portfolio equity from state + prices.

Transaction cost model
----------------------
    notional = abs(quantity) * price
    cost = notional * (bps / 10_000)
    buy:  cash -= notional + cost
    sell: cash += notional - cost

The cost is always subtracted regardless of trade direction (both buys and
sells pay the cost). This is a half-spread / market-impact model simplified
to a single bps rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class Fill(BaseModel):
    """Immutable single-trade execution record.

    Attributes
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    quantity:
        Shares traded: positive for a buy, negative for a sell.
        Zero-quantity fills are valid (no-op).
    price:
        Execution price per share (must be positive; validated by caller).
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: int  # positive=buy, negative=sell
    price: Decimal


class PortfolioState(BaseModel):
    """Immutable snapshot of portfolio cash, holdings, and cumulative costs.

    Attributes
    ----------
    cash:
        Cash balance in dollars.  May be negative (margin / error case).
    holdings:
        Mapping from symbol to share count.  Zero-share entries are not stored
        (they are removed by ``apply_fill`` when the position reaches zero).
    total_costs:
        Cumulative transaction costs paid since inception of this state chain.
    """

    model_config = ConfigDict(frozen=True)

    cash: Decimal
    holdings: dict[str, int]  # symbol -> shares
    total_costs: Decimal

    @classmethod
    def initial(cls, equity: Decimal) -> PortfolioState:
        """Return a fresh all-cash portfolio with the given starting equity."""
        return cls(cash=equity, holdings={}, total_costs=Decimal("0"))


def apply_fill(
    state: PortfolioState,
    fill: Fill,
    *,
    transaction_cost_bps: Decimal,
) -> PortfolioState:
    """Return the new portfolio state after applying a fill.

    Parameters
    ----------
    state:
        Current portfolio state.
    fill:
        The trade to apply.  ``fill.quantity > 0`` is a buy; ``< 0`` is a sell.
    transaction_cost_bps:
        Transaction cost in basis points (e.g. ``Decimal("5")`` for 5 bps).
        Applied symmetrically to buys and sells.

    Returns
    -------
    PortfolioState
        Updated state with new cash, holdings, and total_costs.
    """
    notional = Decimal(abs(fill.quantity)) * fill.price
    cost = notional * (transaction_cost_bps / Decimal("10000"))

    # Update cash: buys drain cash, sells replenish cash; cost always subtracted
    if fill.quantity >= 0:
        new_cash = state.cash - notional - cost
    else:
        new_cash = state.cash + notional - cost

    # Update holdings (copy and mutate)
    new_holdings = dict(state.holdings)
    current_shares = new_holdings.get(fill.symbol, 0)
    updated_shares = current_shares + fill.quantity
    if updated_shares == 0:
        new_holdings.pop(fill.symbol, None)
    else:
        new_holdings[fill.symbol] = updated_shares

    return PortfolioState(
        cash=new_cash,
        holdings=new_holdings,
        total_costs=state.total_costs + cost,
    )


def mark_to_market(
    state: PortfolioState,
    prices: Mapping[str, Decimal],
) -> Decimal:
    """Compute total portfolio equity as cash + sum(holdings * prices).

    Parameters
    ----------
    state:
        Current portfolio state.
    prices:
        Mapping from symbol to current market price.  Symbols not in
        ``state.holdings`` are ignored.

    Returns
    -------
    Decimal
        Total equity value.

    Raises
    ------
    KeyError
        If a symbol in ``state.holdings`` is missing from ``prices``.
    """
    equity = state.cash
    for symbol, shares in state.holdings.items():
        equity += Decimal(shares) * prices[symbol]
    return equity
