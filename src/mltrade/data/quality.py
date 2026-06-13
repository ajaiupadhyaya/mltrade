"""Fail-closed market data quality validation.

Validates a tuple of :class:`~mltrade.data.bars.DailyBar` objects against a
set of structural and completeness rules.  ``validate_daily_bars`` evaluates
**all** rules in a single pass and returns an immutable
:class:`DataQualityReport`.  If *any* issue has severity
:attr:`IssueSeverity.BLOCK`, the report's ``blocked`` flag is ``True`` so
callers can gate downstream operations.

Issue codes (all severity BLOCK):
  - ``empty_input``                 — no bars supplied at all
  - ``duplicate_bar``               — same (symbol, session) appears more than once
  - ``unsorted_bars``               — bars not sorted by (session, symbol)
  - ``symbol_outside_universe``     — bar carries a symbol not in the universe
  - ``missing_universe_symbol``     — a universe symbol has zero bars
  - ``latest_session_mismatch``     — max observed session != expected_last_session
  - ``incomplete_latest_session``   — not every universe symbol has a bar on the
                                      expected last session
  - ``missing_session``             — a symbol is missing one or more XNYS sessions
                                      inside its own first-to-last observed range
  - ``non_finite_value``            — a numeric field is not finite (defensive)
"""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from mltrade.calendar import XNYSCalendar
from mltrade.data.bars import DailyBar
from mltrade.universe import Universe

# Module-level singleton — cheap to share, avoids re-creating the exchange
# calendar object on every call to validate_daily_bars.
_CALENDAR = XNYSCalendar()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class IssueSeverity(StrEnum):
    BLOCK = "block"
    WARN = "warn"


class QualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: IssueSeverity
    message: str


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[QualityIssue, ...] = ()

    @property
    def blocked(self) -> bool:
        """True when at least one issue has severity BLOCK."""
        return any(issue.severity is IssueSeverity.BLOCK for issue in self.issues)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _block(code: str, message: str) -> QualityIssue:
    return QualityIssue(code=code, severity=IssueSeverity.BLOCK, message=message)


def _warn(code: str, message: str) -> QualityIssue:
    return QualityIssue(code=code, severity=IssueSeverity.WARN, message=message)


def _check_finite(bar: DailyBar) -> list[QualityIssue]:
    """Defensive finiteness check on all numeric fields of a bar."""
    issues: list[QualityIssue] = []
    fields: dict[str, Any] = {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "vwap": bar.vwap,
    }
    for name, value in fields.items():
        try:
            fv = float(value)
        except (TypeError, ValueError):
            issues.append(
                _block(
                    "non_finite_value",
                    f"{bar.instrument.symbol} {bar.session}: {name} could not "
                    "be converted to float",
                )
            )
            continue
        if not math.isfinite(fv):
            issues.append(
                _block(
                    "non_finite_value",
                    f"{bar.instrument.symbol} {bar.session}: {name}={value!r} "
                    "is not finite",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_daily_bars(
    bars: tuple[DailyBar, ...],
    *,
    universe: Universe,
    expected_last_session: date,
) -> DataQualityReport:
    """Evaluate all data quality rules over *bars* and return a report.

    Parameters
    ----------
    bars:
        The sequence of :class:`~mltrade.data.bars.DailyBar` objects to
        validate.  The tuple should be sorted by (session, symbol) as produced
        by :class:`~mltrade.data.fixtures.DeterministicBarSource`.
    universe:
        The :class:`~mltrade.universe.Universe` that every bar must belong to.
    expected_last_session:
        The most recent XNYS session that should be present in *bars*.

    Returns
    -------
    DataQualityReport
        Immutable report.  ``report.blocked`` is ``True`` when any issue has
        severity :attr:`IssueSeverity.BLOCK`.
    """
    issues: list[QualityIssue] = []

    # ------------------------------------------------------------------
    # Rule 1: empty input
    # ------------------------------------------------------------------
    if not bars:
        issues.append(_block("empty_input", "No bars were supplied."))
        return DataQualityReport(issues=tuple(issues))

    universe_symbols: frozenset[str] = frozenset(universe.symbols)

    # ------------------------------------------------------------------
    # Rule 2: symbol outside universe
    # ------------------------------------------------------------------
    outside: set[str] = set()
    for bar in bars:
        sym = bar.instrument.symbol
        if sym not in universe_symbols:
            outside.add(sym)
    if outside:
        issues.append(
            _block(
                "symbol_outside_universe",
                f"Bars contain symbols not in the universe: "
                f"{sorted(outside)}",
            )
        )

    # ------------------------------------------------------------------
    # Rule 3: missing universe symbol (zero bars for a universe member)
    # ------------------------------------------------------------------
    observed_symbols: set[str] = {bar.instrument.symbol for bar in bars}
    missing_syms: frozenset[str] = universe_symbols - observed_symbols
    if missing_syms:
        issues.append(
            _block(
                "missing_universe_symbol",
                f"Universe symbols have no bars: {sorted(missing_syms)}",
            )
        )

    # ------------------------------------------------------------------
    # Rule 4: duplicate (symbol, session) pairs
    # ------------------------------------------------------------------
    seen_pairs: set[tuple[str, date]] = set()
    duplicate_pairs: set[tuple[str, date]] = set()
    for bar in bars:
        key = (bar.instrument.symbol, bar.session)
        if key in seen_pairs:
            duplicate_pairs.add(key)
        else:
            seen_pairs.add(key)
    if duplicate_pairs:
        sample = sorted(duplicate_pairs)[:5]
        issues.append(
            _block(
                "duplicate_bar",
                f"Duplicate (symbol, session) pairs detected "
                f"(showing up to 5): {sample}",
            )
        )

    # ------------------------------------------------------------------
    # Rule 5: unsorted output — must be sorted by (session, symbol)
    # ------------------------------------------------------------------
    expected_order = sorted(bars, key=lambda b: (b.session, b.instrument.symbol))
    if list(bars) != expected_order:
        issues.append(
            _block(
                "unsorted_bars",
                "Bars are not sorted by (session, symbol).",
            )
        )

    # ------------------------------------------------------------------
    # Rule 6: non-finite numeric values (defensive)
    # ------------------------------------------------------------------
    for bar in bars:
        issues.extend(_check_finite(bar))

    # ------------------------------------------------------------------
    # Rule 7: latest session mismatch
    # ------------------------------------------------------------------
    max_session: date = max(bar.session for bar in bars)
    if max_session != expected_last_session:
        issues.append(
            _block(
                "latest_session_mismatch",
                f"Latest observed session {max_session} does not match "
                f"expected {expected_last_session}.",
            )
        )

    # ------------------------------------------------------------------
    # Rule 8: incomplete latest session
    # Evaluate only when the latest session matches (avoid double-reporting
    # the same missing data under two different codes).
    # ------------------------------------------------------------------
    if max_session == expected_last_session:
        present_on_last: set[str] = {
            bar.instrument.symbol
            for bar in bars
            if bar.session == expected_last_session
        }
        # Only check universe symbols (outside-universe ones are already flagged)
        missing_on_last = universe_symbols - present_on_last
        if missing_on_last:
            issues.append(
                _block(
                    "incomplete_latest_session",
                    f"Universe symbols missing on latest session "
                    f"{expected_last_session}: {sorted(missing_on_last)}",
                )
            )

    # ------------------------------------------------------------------
    # Rule 9: missing XNYS sessions per symbol within its observed range
    # ------------------------------------------------------------------
    # Build per-symbol session sets (only universe symbols to avoid noise from
    # outside-universe bars that are already flagged).
    symbol_sessions: dict[str, set[date]] = {}
    for bar in bars:
        sym = bar.instrument.symbol
        if sym in universe_symbols:
            symbol_sessions.setdefault(sym, set()).add(bar.session)

    missing_session_details: list[str] = []
    for sym, sessions in symbol_sessions.items():
        first_session = min(sessions)
        last_session = max(sessions)
        expected_sessions = set(
            _CALENDAR.sessions_in_range(first_session, last_session)
        )
        gap = expected_sessions - sessions
        if gap:
            missing_session_details.append(
                f"{sym}: {len(gap)} missing session(s), "
                f"e.g. {sorted(gap)[0]}"
            )
    if missing_session_details:
        issues.append(
            _block(
                "missing_session",
                "Some symbols are missing XNYS sessions within their observed "
                f"range: {missing_session_details[:5]}",
            )
        )

    return DataQualityReport(issues=tuple(issues))
