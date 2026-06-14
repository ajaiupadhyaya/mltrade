from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
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


# ---------------------------------------------------------------------------
# Operational evidence tables
# ---------------------------------------------------------------------------


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class QualityReport(Base):
    __tablename__ = "quality_reports"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class ForecastBatch(Base):
    __tablename__ = "forecast_batches"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class PortfolioTarget(Base):
    __tablename__ = "portfolio_targets"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class RiskReportRow(Base):
    __tablename__ = "risk_report_rows"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class ExecutionPreview(Base):
    __tablename__ = "execution_previews"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class ExecutionIntentRow(Base):
    __tablename__ = "execution_intents"
    __table_args__ = (
        UniqueConstraint(
            "client_order_id",
            name="uq_execution_intents_client_order_id",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_order_id: Mapped[str] = mapped_column(String(120), index=True)
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(10))
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class BrokerOrder(Base):
    __tablename__ = "broker_orders"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class ReconciliationRun(Base):
    __tablename__ = "reconciliation_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    decision_session: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
