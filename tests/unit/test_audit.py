from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mltrade.operations.audit import AuditService
from mltrade.operations.models import AuditEvent, Base


def test_audit_service_appends_structured_event() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        service = AuditService(session)
        event_id = service.record(
            event_type="risk.trading_blocked",
            occurred_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
            actor="system",
            correlation_id="run-123",
            payload={"reason": "stale_data"},
        )

        stored = session.scalar(
            select(AuditEvent).where(AuditEvent.id == event_id)
        )

    assert stored is not None
    assert stored.event_type == "risk.trading_blocked"
    assert stored.payload == {"reason": "stale_data"}
    assert stored.occurred_at == datetime(2026, 6, 12, 21, 0, tzinfo=UTC)
    assert stored.occurred_at.tzinfo is UTC
    engine.dispose()
