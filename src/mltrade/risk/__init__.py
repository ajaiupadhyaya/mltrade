"""Fail-closed pre-trade risk policy for the MLTrade paper-trading MVP.

Public API
----------
- ``CheckStatus``   — StrEnum: PASS, WARN, BLOCK
- ``RiskCheck``     — frozen value object: (code, status, message)
- ``RiskReport``    — frozen collection of checks; ``blocked`` property;
                      ``by_code`` lookup
- ``PreTradeContext`` — frozen input context (allows model_copy(update=...))
- ``evaluate_pre_trade`` — deterministic, fail-closed evaluator
"""

from mltrade.risk.checks import CheckStatus, RiskCheck, RiskReport
from mltrade.risk.policy import PreTradeContext, evaluate_pre_trade

__all__ = [
    "CheckStatus",
    "PreTradeContext",
    "RiskCheck",
    "RiskReport",
    "evaluate_pre_trade",
]
