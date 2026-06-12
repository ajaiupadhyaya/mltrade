from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import AuditEvent, Base


def make_event() -> AuditEvent:
    return AuditEvent(
        event_type="system.test",
        occurred_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
        actor="pytest",
        correlation_id="database-test",
        payload={"result": "ok"},
    )


def test_session_scope_commits_on_success() -> None:
    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with session_scope(engine) as session:
        session.add(make_event())

    with session_scope(engine) as session:
        count = session.scalar(select(func.count()).select_from(AuditEvent))

    assert count == 1
    engine.dispose()


def test_session_scope_rolls_back_on_error() -> None:
    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with pytest.raises(RuntimeError, match="stop"):
        with session_scope(engine) as session:
            session.add(make_event())
            session.flush()
            raise RuntimeError("stop")

    with session_scope(engine) as session:
        count = session.scalar(select(func.count()).select_from(AuditEvent))

    assert count == 0
    engine.dispose()
