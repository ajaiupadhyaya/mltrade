"""Value objects for the pre-trade risk policy.

Mirrors the ``IssueSeverity`` / ``QualityIssue`` / ``DataQualityReport``
pattern from :mod:`mltrade.data.quality` but uses domain names appropriate
for a pre-trade gate.

Design notes
------------
- ``CheckStatus`` uses the same StrEnum convention as ``IssueSeverity``.
- ``RiskCheck`` is frozen (code + status + message, optional detail).
- ``RiskReport`` is frozen; ``blocked`` is a property; ``by_code`` returns
  the single check with that code or raises ``KeyError``.
- Deterministic ordering is enforced by the policy — the report stores checks
  in the order they were emitted.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class CheckStatus(StrEnum):
    """Result of a single pre-trade risk check."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Individual check result
# ---------------------------------------------------------------------------


class RiskCheck(BaseModel):
    """Immutable result of one named pre-trade risk rule.

    Attributes
    ----------
    code:
        Stable machine-readable identifier (e.g. ``"snapshot_freshness"``).
    status:
        ``PASS``, ``WARN``, or ``BLOCK``.
    message:
        Human-readable description of the outcome.
    detail:
        Optional structured detail (e.g. actual vs. expected values).
    """

    model_config = ConfigDict(frozen=True)

    code: str
    status: CheckStatus
    message: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class RiskReport(BaseModel):
    """Immutable, ordered collection of :class:`RiskCheck` results.

    Attributes
    ----------
    checks:
        All checks emitted by :func:`~mltrade.risk.policy.evaluate_pre_trade`,
        in deterministic order.

    Properties
    ----------
    blocked:
        ``True`` when at least one check has status :attr:`CheckStatus.BLOCK`.

    Methods
    -------
    by_code(code):
        Return the single :class:`RiskCheck` whose ``code`` matches.
        Raises :exc:`KeyError` if absent.
    """

    model_config = ConfigDict(frozen=True)

    checks: tuple[RiskCheck, ...] = ()

    @property
    def blocked(self) -> bool:
        """True when at least one check has status BLOCK."""
        return any(c.status is CheckStatus.BLOCK for c in self.checks)

    def by_code(self, code: str) -> RiskCheck:
        """Return the check with the given code.

        Raises
        ------
        KeyError
            If no check with that code is present in the report.
        """
        for check in self.checks:
            if check.code == code:
                return check
        raise KeyError(f"No risk check with code {code!r} in this report.")
