"""Operational evidence repository.

Persists all pipeline artifacts — previews, intents, snapshots, reports, etc.
— to the configured SQL database.

Design notes
------------
- All write methods flush (never commit): callers own the transaction boundary,
  exactly as :class:`~mltrade.operations.audit.AuditService` does.
- ``save_intent`` is idempotent on ``client_order_id``: if a row already exists
  for that ID the existing UUID is returned and no second row is inserted.
- ``load_preview`` reconstructs a :class:`~mltrade.execution.service.Preview`
  from the stored JSON blob so that
  ``load_preview(save_preview(p)) == p`` holds exactly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from mltrade.execution.intents import ExecutionIntent
from mltrade.execution.service import Preview
from mltrade.operations.models import (
    BacktestRun,
    BrokerOrder,
    DatasetSnapshot,
    ExecutionIntentRow,
    ExecutionPreview,
    ForecastBatch,
    PortfolioTarget,
    QualityReport,
    ReconciliationRun,
    RiskReportRow,
)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class OperationsRepository:
    """Persist operational evidence to a SQL database.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.  The repository flushes after every write
        but never commits — the caller owns the transaction boundary.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # ExecutionPreview
    # ------------------------------------------------------------------

    def save_preview(
        self,
        preview: Preview,
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist *preview* and return its storage UUID.

        Serialises the :class:`~mltrade.execution.service.Preview` pydantic
        model to JSON so that :meth:`load_preview` can reconstruct it exactly.
        """
        blob: dict[str, object] = json.loads(preview.model_dump_json())
        row = ExecutionPreview(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=blob,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    def load_preview(self, preview_id: UUID) -> Preview:
        """Load and reconstruct a :class:`~mltrade.execution.service.Preview`.

        Raises
        ------
        KeyError
            When no row with *preview_id* exists.
        """
        row = self._session.scalar(
            select(ExecutionPreview).where(ExecutionPreview.id == preview_id)
        )
        if row is None:
            raise KeyError(f"No ExecutionPreview with id={preview_id!r}")
        return Preview.model_validate_json(json.dumps(row.payload))

    # ------------------------------------------------------------------
    # ExecutionIntent  (idempotent on client_order_id)
    # ------------------------------------------------------------------

    def save_intent(
        self,
        intent: ExecutionIntent,
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist *intent* idempotently and return its storage UUID.

        If a row with the same ``client_order_id`` already exists, the
        existing row's UUID is returned and no new row is inserted.
        """
        existing = self._session.scalar(
            select(ExecutionIntentRow).where(
                ExecutionIntentRow.client_order_id == intent.client_order_id
            )
        )
        if existing is not None:
            return existing.id

        blob: dict[str, object] = json.loads(intent.model_dump_json())
        row = ExecutionIntentRow(
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=str(intent.side.value),
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=blob,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    def count_intents(self, client_order_id: str) -> int:
        """Return the number of rows stored for *client_order_id* (0 or 1)."""
        rows = self._session.scalars(
            select(ExecutionIntentRow).where(
                ExecutionIntentRow.client_order_id == client_order_id
            )
        ).all()
        return len(rows)

    # ------------------------------------------------------------------
    # DatasetSnapshot
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a dataset-snapshot record."""
        row = DatasetSnapshot(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # QualityReport
    # ------------------------------------------------------------------

    def save_quality_report(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a data-quality report."""
        row = QualityReport(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # ForecastBatch
    # ------------------------------------------------------------------

    def save_forecast_batch(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a forecast-batch record."""
        row = ForecastBatch(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # BacktestRun
    # ------------------------------------------------------------------

    def save_backtest_run(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a backtest-run record."""
        row = BacktestRun(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # PortfolioTarget
    # ------------------------------------------------------------------

    def save_portfolio_target(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a portfolio-target record."""
        row = PortfolioTarget(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # RiskReportRow
    # ------------------------------------------------------------------

    def save_risk_report(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a risk-report record."""
        row = RiskReportRow(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # BrokerOrder
    # ------------------------------------------------------------------

    def save_broker_order(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a broker-order record."""
        row = BrokerOrder(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id

    # ------------------------------------------------------------------
    # ReconciliationRun
    # ------------------------------------------------------------------

    def save_reconciliation_run(
        self,
        payload: dict[str, object],
        *,
        correlation_id: str = "",
        decision_session: str = "",
    ) -> UUID:
        """Persist a reconciliation-run record."""
        row = ReconciliationRun(
            correlation_id=correlation_id,
            decision_session=decision_session,
            created_at=_now_utc(),
            payload=payload,
        )
        self._session.add(row)
        self._session.flush()
        return row.id
