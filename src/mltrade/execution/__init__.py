"""Broker execution contracts and simulated broker.

Public API
----------
Protocol / exceptions
~~~~~~~~~~~~~~~~~~~~~
- ``Broker``        — structural Protocol for broker adapters
- ``BrokerError``   — base exception
- ``BrokerTimeout`` — raised on timeout (before or after submission)

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
"""

from mltrade.execution.broker import (
    Broker,
    BrokerAccount,
    BrokerError,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerTimeout,
    OrderSide,
    OrderStatus,
)
from mltrade.execution.intents import ExecutionIntent, build_intent
from mltrade.execution.simulated import SimulatedBroker, SubmitOutcome

__all__ = [
    "Broker",
    "BrokerAccount",
    "BrokerError",
    "BrokerFill",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerTimeout",
    "ExecutionIntent",
    "OrderSide",
    "OrderStatus",
    "SimulatedBroker",
    "SubmitOutcome",
    "build_intent",
]
