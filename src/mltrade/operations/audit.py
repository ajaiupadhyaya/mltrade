from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from mltrade.domain.time import require_utc
from mltrade.operations.models import AuditEvent


class AuditService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        *,
        event_type: str,
        occurred_at: datetime,
        actor: str,
        correlation_id: str,
        payload: dict[str, object],
    ) -> UUID:
        event = AuditEvent(
            event_type=event_type,
            occurred_at=require_utc(occurred_at),
            actor=actor,
            correlation_id=correlation_id,
            payload=payload,
        )
        self._session.add(event)
        self._session.flush()
        return event.id
