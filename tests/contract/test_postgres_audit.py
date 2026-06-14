import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from mltrade.operations.audit import AuditService
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import AuditEvent, Base

pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.environ.get("MLTRADE_TEST_DATABASE_URL"),
        reason="requires MLTRADE_TEST_DATABASE_URL and a running Postgres",
    ),
]


def test_postgres_persists_audit_event() -> None:
    database_url = os.environ["MLTRADE_TEST_DATABASE_URL"]
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)

    with session_scope(engine) as session:
        event_id = AuditService(session).record(
            event_type="system.contract_test",
            occurred_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
            actor="pytest",
            correlation_id="contract-1",
            payload={"database": "postgresql"},
        )

    with session_scope(engine) as session:
        stored = session.scalar(
            select(AuditEvent).where(AuditEvent.id == event_id)
        )
        assert stored is not None
        assert stored.payload == {"database": "postgresql"}
        assert stored.occurred_at == datetime(
            2026,
            6,
            12,
            21,
            0,
            tzinfo=UTC,
        )
        assert stored.occurred_at.tzinfo is UTC
