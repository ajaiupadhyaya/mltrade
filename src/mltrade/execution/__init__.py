"""Broker execution contracts and simulated broker.

Public API
----------
Protocol / exceptions
~~~~~~~~~~~~~~~~~~~~~
- ``Broker``            — structural Protocol for broker adapters
- ``BrokerError``       — base exception
- ``BrokerTimeout``     — raised on timeout (before or after submission)
- ``BrokerSafetyError`` — raised when a safety invariant is violated

Value objects
~~~~~~~~~~~~~
- ``BrokerAccount``  — account cash/equity/flags
- ``BrokerPosition`` — symbol + quantity + avg_price
- ``BrokerOrder``    — order identity + fill state
- ``BrokerFill``     — single confirmed fill (UTC timestamp)

Enums
~~~~~
- ``OrderSide``    — buy / sell
- ``OrderStatus``  — new / partially_filled / filled / rejected

Intents
~~~~~~~
- ``ExecutionIntent`` — idempotent, frozen execution intent
- ``build_intent``    — factory with sha256-derived ``client_order_id``

Simulation
~~~~~~~~~~
- ``SimulatedBroker`` — in-memory broker for testing / offline workflows
- ``SubmitOutcome``   — outcome control surface for :class:`SimulatedBroker`

Reconciliation
~~~~~~~~~~~~~~
- ``InternalState``              — system's internal beliefs (cash/positions/orders)
- ``ReconciliationDifference``   — one discrepancy between internal and broker state
- ``ReconciliationResult``       — collection of differences; ``blocked`` property
- ``reconcile``                  — compare InternalState vs Broker; returns result

Service
~~~~~~~
- ``ExecutionService`` — orchestrates preview + safe idempotent submit
- ``Preview``          — pre-submit snapshot (intents + reconciliation + risk)
- ``SubmitResult``     — aggregate outcome of a submit call
- ``IntentOutcome``    — per-intent submit result
"""

from mltrade.execution.broker import (
    Broker,
    BrokerAccount,
    BrokerError,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerSafetyError,
    BrokerTimeout,
    OrderSide,
    OrderStatus,
)
from mltrade.execution.intents import ExecutionIntent, build_intent
from mltrade.execution.reconciliation import (
    InternalState,
    ReconciliationDifference,
    ReconciliationResult,
    reconcile,
)
from mltrade.execution.service import (
    ExecutionService,
    IntentOutcome,
    Preview,
    SubmitResult,
)
from mltrade.execution.simulated import SimulatedBroker, SubmitOutcome

__all__ = [
    "Broker",
    "BrokerAccount",
    "BrokerError",
    "BrokerFill",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerSafetyError",
    "BrokerTimeout",
    "ExecutionIntent",
    "ExecutionService",
    "IntentOutcome",
    "InternalState",
    "OrderSide",
    "OrderStatus",
    "Preview",
    "ReconciliationDifference",
    "ReconciliationResult",
    "SimulatedBroker",
    "SubmitOutcome",
    "SubmitResult",
    "build_intent",
    "reconcile",
]
