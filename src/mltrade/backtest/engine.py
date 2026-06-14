"""Walk-forward backtest engine.

``run_backtest`` is the top-level function.  It accepts a tuple of
``DailyBar`` objects and returns a ``BacktestResult``.

Algorithm
---------
1.  Build feature rows from all bars (``build_feature_rows``).
2.  Group bars by session date, sort sessions ascending.
3.  Apply a 504-session warmup: the first 504 sessions are training-only.
4.  For each *decision session* after the warmup (where a "next session"
    exists for trade execution):
    a.  Every 21 sessions (retrain checkpoint): call ``generate_forecast_batch``
        to get a new ``ForecastBatch``.  If ``ForecastBlocked``, hold the
        previous target weights (or stay flat).
    b.  Extract forecasts (symbol → predicted_return) from the batch.
    c.  Extract trailing realized_volatility_21 per symbol from feature rows.
    d.  Call ``build_target`` → ``OptimizationResult`` (target weights).
    e.  Store target weights (held between retrains).
5.  Decision:
    - Compute target share quantities at the *next session's OPEN* price.
    - Fills are executed at the next session's open.
6.  Accounting:
    - Decisions are computed ONCE (ceteris-paribus).
    - The accounting sim is run for cost levels {2, 5, 10} bps independently,
      plus once for the requested cost_bps (headline metrics).
7.  Mark positions to CLOSE at the end of each session.
8.  Baselines:
    - equal_weight: rebalance to 1/N each session (same cost model as headline).
    - cash: always 0% return.
9.  Build and return ``BacktestResult``.

Determinism
-----------
- Symbol order: always ``sorted(symbols)`` throughout.
- Feature rows and forecasts are deterministic (same inputs → same output).
- ``Decimal`` arithmetic for fills; float for equity curve.
- All float outputs rounded to 10 d.p. before storing.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import NamedTuple

from mltrade.backtest.accounting import (
    Fill,
    PortfolioState,
    apply_fill,
    mark_to_market,
)
from mltrade.backtest.reporting import BacktestResult, CostSummary, compute_metrics
from mltrade.data.bars import DailyBar
from mltrade.features.pipeline import build_feature_rows
from mltrade.models.forecasts import ForecastBlocked
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARMUP_SESSIONS: int = 504  # minimum training history required
_RETRAIN_EVERY: int = 21  # sessions between model retrains

_DEFAULT_LIMITS = PortfolioLimits(
    maximum_position_weight=Decimal("0.25"),
    minimum_cash_weight=Decimal("0.05"),
    target_annual_volatility=Decimal("0.15"),
)

_INITIAL_EQUITY = Decimal("1_000_000")
_COST_SENSITIVITY_LEVELS: tuple[Decimal, ...] = (
    Decimal("2"),
    Decimal("5"),
    Decimal("10"),
)

_ROUND_PLACES = 10


def _round(x: float) -> float:
    return round(x, _ROUND_PLACES)


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------


class _SessionData(NamedTuple):
    """Prices for one XNYS session."""

    session: date
    open_prices: dict[str, Decimal]
    close_prices: dict[str, Decimal]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_session_map(
    bars: tuple[DailyBar, ...],
) -> dict[date, _SessionData]:
    """Group bars by session, returning open and close prices per symbol."""
    raw: dict[date, tuple[dict[str, Decimal], dict[str, Decimal]]] = {}
    for bar in bars:
        sess = bar.session
        if sess not in raw:
            raw[sess] = ({}, {})
        raw[sess][0][bar.instrument.symbol] = bar.open
        raw[sess][1][bar.instrument.symbol] = bar.close
    return {
        sess: _SessionData(
            session=sess,
            open_prices=opens,
            close_prices=closes,
        )
        for sess, (opens, closes) in raw.items()
    }


def _compute_target_quantities(
    target_weights: dict[str, Decimal],
    equity: Decimal,
    open_prices: dict[str, Decimal],
) -> dict[str, int]:
    """Convert portfolio weight targets to integer share quantities.

    Parameters
    ----------
    target_weights:
        Symbol → weight (fraction of NAV).
    equity:
        Current portfolio NAV (used to size positions).
    open_prices:
        Symbol → open price at the execution session.

    Returns
    -------
    dict[str, int]
        Symbol → target share count (rounded down to whole shares).
    """
    result: dict[str, int] = {}
    for symbol in sorted(target_weights):  # sorted for determinism
        weight = target_weights[symbol]
        price = open_prices.get(symbol)
        if price is None or price <= Decimal("0"):
            continue
        # Floor to whole shares
        shares = int((equity * weight) / price)
        if shares > 0:
            result[symbol] = shares
    return result


def _compute_fills(
    current_holdings: dict[str, int],
    target_quantities: dict[str, int],
    all_symbols: list[str],
) -> list[Fill]:
    """Compute list of fills needed to move from current to target holdings.

    Parameters
    ----------
    current_holdings:
        Symbol → current share count.
    target_quantities:
        Symbol → target share count (0 means sell everything).
    all_symbols:
        All symbols in the universe (sorted for determinism).

    Returns
    -------
    list[Fill]
        Fills with non-zero quantities only, sorted by symbol for determinism.
    """
    fills: list[Fill] = []
    all_syms = sorted(
        set(current_holdings.keys()) | set(target_quantities.keys()) | set(all_symbols)
    )
    for symbol in all_syms:
        current = current_holdings.get(symbol, 0)
        target = target_quantities.get(symbol, 0)
        delta = target - current
        if delta != 0:
            fills.append(Fill(symbol=symbol, quantity=delta, price=Decimal("0")))
    return fills


def _apply_fills_at_open(
    state: PortfolioState,
    target_quantities: dict[str, int],
    open_prices: dict[str, Decimal],
    cost_bps: Decimal,
    all_symbols: list[str],
) -> tuple[PortfolioState, float]:
    """Apply rebalancing fills at open prices.

    Returns the new state and the total traded notional for turnover tracking.
    """
    fills = _compute_fills(state.holdings, target_quantities, all_symbols)
    total_notional = 0.0
    for fill in fills:
        price = open_prices.get(fill.symbol)
        if price is None or price <= Decimal("0"):
            continue
        priced_fill = Fill(
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=price,
        )
        state = apply_fill(state, priced_fill, transaction_cost_bps=cost_bps)
        total_notional += abs(fill.quantity) * float(price)
    return state, total_notional


def _run_sim(
    decisions: list[tuple[int, dict[str, Decimal]]],
    sessions: list[_SessionData],
    cost_bps: Decimal,
    all_symbols: list[str],
    initial_equity: Decimal,
) -> tuple[list[float], float, list[float], list[bool]]:
    """Run accounting simulation for one cost level.

    Parameters
    ----------
    decisions:
        List of (decision_session_index, target_weights) pairs.  The index
        refers to the position in ``sessions``.  Execution happens at the
        *next* session's open.
    sessions:
        All backtest sessions in order (after warmup), indexed to match
        ``decisions``.
    cost_bps:
        Transaction cost in basis points.
    all_symbols:
        Sorted list of all universe symbols.
    initial_equity:
        Starting NAV.

    Returns
    -------
    tuple of (equity_curve, total_costs_float, turnover_vals, hit_flags)
    """
    state = PortfolioState.initial(initial_equity)

    # Map decision session index → target weights
    decision_map: dict[int, dict[str, Decimal]] = {
        idx: weights for idx, weights in decisions
    }

    equity_curve: list[float] = [float(initial_equity)]
    turnover_vals: list[float] = []
    hit_flags: list[bool] = []

    n_sessions = len(sessions)

    for i in range(n_sessions):
        sess = sessions[i]

        # Check if there's a rebalance decision for this session index.
        # decisions[i] maps to backtest_sessions_data[i] (1:1).
        if i in decision_map:
            target_weights = decision_map[i]
            # Size positions using equity marked to THIS session's open prices
            current_equity_for_sizing = state.cash + sum(
                Decimal(shares) * sess.open_prices.get(sym, Decimal("1"))
                for sym, shares in state.holdings.items()
            )

            target_quantities = _compute_target_quantities(
                target_weights,
                current_equity_for_sizing,
                sess.open_prices,
            )
            state, total_notional = _apply_fills_at_open(
                state,
                target_quantities,
                sess.open_prices,
                cost_bps,
                all_symbols,
            )
        else:
            total_notional = 0.0

        # Mark to market at CLOSE
        equity = mark_to_market(
            state,
            {s: p for s, p in sess.close_prices.items()},
        )
        equity_float = float(equity)
        prev_equity = equity_curve[-1]

        # Turnover: traded notional / previous equity
        turnover = total_notional / prev_equity if prev_equity > 0 else 0.0
        turnover_vals.append(turnover)

        # Hit: session return positive?
        session_return = equity_float / prev_equity - 1.0 if prev_equity > 0 else 0.0
        hit_flags.append(session_return > 0.0)

        equity_curve.append(equity_float)

    total_costs_float = float(state.total_costs)
    return equity_curve, total_costs_float, turnover_vals, hit_flags


def _run_equal_weight_sim(
    sessions: list[_SessionData],
    all_symbols: list[str],
    cost_bps: Decimal,
    initial_equity: Decimal,
) -> float:
    """Run equal-weight rebalancing baseline, return annualized return."""
    n_syms = len(all_symbols)
    if n_syms == 0:
        return 0.0

    state = PortfolioState.initial(initial_equity)
    equity_curve: list[float] = [float(initial_equity)]
    n_sessions = len(sessions)

    for i in range(n_sessions):
        sess = sessions[i]

        # Equal weight: 1/N per symbol (5% minimum cash = hold 95% invested)
        weight = Decimal("0.95") / Decimal(n_syms)
        target_weights = {sym: weight for sym in all_symbols}

        # Mark portfolio at open for sizing
        current_equity_for_sizing = state.cash + sum(
            Decimal(shares) * sess.open_prices.get(sym, Decimal("1"))
            for sym, shares in state.holdings.items()
        )

        target_quantities = _compute_target_quantities(
            target_weights,
            current_equity_for_sizing,
            sess.open_prices,
        )
        state, _ = _apply_fills_at_open(
            state,
            target_quantities,
            sess.open_prices,
            cost_bps,
            all_symbols,
        )

        # Mark to market at CLOSE
        equity = mark_to_market(state, sess.close_prices)
        equity_curve.append(float(equity))

    initial = equity_curve[0]
    final = equity_curve[-1]
    n = len(equity_curve) - 1
    if initial <= 0.0 or final <= 0.0 or n == 0:
        return 0.0
    ann_return = (final / initial) ** (252.0 / n) - 1.0
    return _round(ann_return)


# ---------------------------------------------------------------------------
# Per-symbol contribution
# ---------------------------------------------------------------------------


def _compute_per_symbol_contribution(
    bars: tuple[DailyBar, ...],
    backtest_sessions: list[date],
) -> dict[str, float]:
    """Compute per-symbol buy-and-hold contribution over the backtest period.

    Uses the simple buy-and-hold return for each symbol over the backtest
    sessions as a proxy for contribution.  Rounded for determinism.
    """
    if not backtest_sessions:
        return {}

    start_date = backtest_sessions[0]
    end_date = backtest_sessions[-1]

    # Find first and last close for each symbol within backtest period
    first_close: dict[str, Decimal] = {}
    last_close: dict[str, Decimal] = {}

    for bar in bars:
        sym = bar.instrument.symbol
        if bar.session < start_date or bar.session > end_date:
            continue
        if sym not in first_close:
            first_close[sym] = bar.close
        last_close[sym] = bar.close

    result: dict[str, float] = {}
    for sym in sorted(first_close.keys()):
        if sym in last_close and first_close[sym] > Decimal("0"):
            contrib = float(last_close[sym] / first_close[sym] - Decimal("1"))
            result[sym] = _round(contrib)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    bars: tuple[DailyBar, ...],
    *,
    cost_bps: Decimal = Decimal("5"),
    limits: PortfolioLimits | None = None,
    initial_equity: Decimal = _INITIAL_EQUITY,
) -> BacktestResult:
    """Run a walk-forward backtest on the provided bars.

    Parameters
    ----------
    bars:
        All daily bars sorted by (session, symbol).  Must span enough history
        for at least 504 distinct decision sessions (warmup) plus 1 execution
        session.
    cost_bps:
        Headline transaction cost in basis points.
    limits:
        Portfolio hard constraints.  Defaults to the standard MVP limits.
    initial_equity:
        Starting NAV in dollars.

    Returns
    -------
    BacktestResult
        Immutable result with headline metrics, cost sensitivity analysis,
        per-symbol contributions, and equal-weight / cash baselines.
    """
    if limits is None:
        limits = _DEFAULT_LIMITS

    if not bars:
        raise ValueError("bars must not be empty")

    # -----------------------------------------------------------------------
    # Step 1: Build all feature rows
    # -----------------------------------------------------------------------
    feature_rows = build_feature_rows(bars, snapshot_id="backtest-engine-v1")

    # -----------------------------------------------------------------------
    # Step 2: Group bars by session, determine sorted session list
    # -----------------------------------------------------------------------
    session_map = _build_session_map(bars)
    all_sessions_sorted = sorted(session_map.keys())

    # -----------------------------------------------------------------------
    # Step 3: Identify all unique symbols (sorted for determinism)
    # -----------------------------------------------------------------------
    all_symbols: list[str] = sorted(
        {bar.instrument.symbol for bar in bars}
    )

    # -----------------------------------------------------------------------
    # Step 4: Apply 504-session warmup
    # -----------------------------------------------------------------------
    if len(all_sessions_sorted) <= _WARMUP_SESSIONS + 1:
        raise ValueError(
            f"Not enough sessions for backtest: need > {_WARMUP_SESSIONS + 1} "
            f"sessions, got {len(all_sessions_sorted)}"
        )

    # Decision sessions are sessions after warmup, where a next session exists
    # for execution (so we exclude the very last session).
    decision_sessions = all_sessions_sorted[_WARMUP_SESSIONS:-1]
    execution_sessions = all_sessions_sorted[_WARMUP_SESSIONS + 1 :]

    # Backtest simulation sessions (what we track equity over) are
    # execution_sessions (we mark equity at close of execution sessions)
    # Actually we want to track equity from the first execution session onward.
    # Each backtest "step":
    #   - decision_sessions[i] is when we compute the signal
    #   - execution_sessions[i] is when we execute (open) and hold to close
    backtest_sessions_data: list[_SessionData] = [
        session_map[sess] for sess in execution_sessions
    ]

    # -----------------------------------------------------------------------
    # Step 5: Walk-forward: compute decisions (forecasts → weights)
    #         ONCE, then replay for each cost level.
    # -----------------------------------------------------------------------
    # decisions: list of (execution_session_index, target_weights)
    # index refers to backtest_sessions_data index
    decisions: list[tuple[int, dict[str, Decimal]]] = []

    current_target: OptimizationResult | None = None
    last_decision_weights: dict[str, Decimal] = {}

    for i, decision_sess in enumerate(decision_sessions):
        # Retrain every 21 sessions (or first session)
        if i % _RETRAIN_EVERY == 0:
            try:
                batch = generate_forecast_batch(feature_rows, decision_sess)
                forecasts_map: dict[str, float] = {
                    fc.symbol: fc.predicted_forward_return
                    for fc in batch.forecasts
                }

                # Extract trailing vol for symbols in forecast batch
                # Use the decision session's feature rows
                vol_map: dict[str, float] = {}
                for row in feature_rows:
                    if row.decision_session == decision_sess and not row.missing:
                        vol_map[row.symbol] = row.realized_volatility_21

                current_target = build_target(
                    forecasts=forecasts_map,
                    trailing_volatility=vol_map,
                    limits=limits,
                )

                if not current_target.blocked:
                    last_decision_weights = dict(current_target.weights)
                # If blocked, keep previous weights (set below)

            except ForecastBlocked:
                # Hold previous weights (all-cash if no previous)
                pass

        # Store decision for this execution slot
        # execution_session_index = i (1:1 mapping: decision[i] → execution[i])
        decisions.append((i, last_decision_weights.copy()))

    # -----------------------------------------------------------------------
    # Step 6: Run accounting sim for headline cost_bps
    # -----------------------------------------------------------------------
    equity_curve, total_costs_float, turnover_vals, hit_flags = _run_sim(
        decisions,
        backtest_sessions_data,
        cost_bps,
        all_symbols,
        initial_equity,
    )

    n_backtest_sessions = len(backtest_sessions_data)

    headline_summary = compute_metrics(
        equity_curve,
        total_costs_float,
        turnover_vals,
        hit_flags,
        cost_bps,
    )

    # -----------------------------------------------------------------------
    # Step 7: Run cost sensitivity (same decisions, different cost levels)
    # -----------------------------------------------------------------------
    cost_sensitivity: dict[Decimal, CostSummary] = {}
    for sens_bps in _COST_SENSITIVITY_LEVELS:
        s_eq, s_costs, s_turn, s_hits = _run_sim(
            decisions,
            backtest_sessions_data,
            sens_bps,
            all_symbols,
            initial_equity,
        )
        cost_sensitivity[sens_bps] = compute_metrics(
            s_eq, s_costs, s_turn, s_hits, sens_bps
        )

    # -----------------------------------------------------------------------
    # Step 8: Equal-weight baseline
    # -----------------------------------------------------------------------
    ew_return = _run_equal_weight_sim(
        backtest_sessions_data,
        all_symbols,
        cost_bps,
        initial_equity,
    )

    # -----------------------------------------------------------------------
    # Step 9: Per-symbol contribution
    # -----------------------------------------------------------------------
    execution_session_dates = [s.session for s in backtest_sessions_data]
    per_symbol = _compute_per_symbol_contribution(bars, execution_session_dates)

    # -----------------------------------------------------------------------
    # Step 10: Build result
    # -----------------------------------------------------------------------
    return BacktestResult(
        sessions=n_backtest_sessions,
        annualized_return=headline_summary.annualized_return,
        annualized_volatility=headline_summary.annualized_volatility,
        sharpe=headline_summary.sharpe,
        max_drawdown=headline_summary.max_drawdown,
        turnover=headline_summary.turnover,
        total_costs=headline_summary.total_costs,
        hit_rate=headline_summary.hit_rate,
        cost_sensitivity=cost_sensitivity,
        per_symbol_contribution=per_symbol,
        equal_weight_return=ew_return,
        cash_return=0.0,
    )
