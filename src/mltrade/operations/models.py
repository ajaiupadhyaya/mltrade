from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from mltrade.domain.time import require_utc


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        normalized = require_utc(value)
        if dialect.name == "sqlite":
            return normalized.replace(tzinfo=None)
        return normalized

    def process_result_value(
        self,
        value: datetime | None,
        _dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    actor: Mapped[str] = mapped_column(String(120))
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
