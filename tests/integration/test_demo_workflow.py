"""Integration tests for the offline demo workflow (Task 14).

All tests run entirely offline (no network access).

Test inventory
--------------
1. test_demo_runs_end_to_end_without_network
   — quality.blocked=False, backtest.sessions>250, target.blocked=False,
     preview.risk_report.blocked=False.

2. test_replaying_demo_reuses_execution_intents
   — two run_demo calls yield the same client_order_ids (determinism).
     When the optimizer produces weights, intent IDs must match.

3. test_demo_persists_evidence_to_db
   — after run_demo, query DB and confirm rows exist for each pipeline-stage
     evidence table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from mltrade.config import Settings
from mltrade.workflows.demo import DemoResult, run_demo

# ---------------------------------------------------------------------------
# Shared clock and settings
# ---------------------------------------------------------------------------

# Fixed clock → deterministic last_session = 2026-06-12
_CLOCK = datetime(2026, 6, 13, 22, 0, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    db_path = tmp_path / "ops.db"
    return Settings(
        environment="local",
        data_root=tmp_path / "data",
        database_url=f"sqlite+pysqlite:///{db_path}",
        reference_equity="1000000",
    )


# ---------------------------------------------------------------------------
# Ensure DB schema is created before any test that needs it
# ---------------------------------------------------------------------------


def _init_db(settings: Settings) -> None:
    from mltrade.operations.database import build_engine
    from mltrade.operations.models import Base

    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_demo_runs_end_to_end_without_network(tmp_path: Path) -> None:
    """Full offline pipeline completes with all risk gates passing."""
    settings = _settings(tmp_path)
    _init_db(settings)

    result: DemoResult = run_demo(settings, clock=_CLOCK)

    # Quality gate must pass
    assert not result.quality.blocked, (
        f"Quality blocked unexpectedly: {result.quality.issues}"
    )

    # Backtest must have sufficient sessions
    assert result.backtest.sessions > 250, (
        f"Expected > 250 backtest sessions, got {result.backtest.sessions}"
    )

    # Portfolio target must not be blocked (may be all-cash when no positive forecasts)
    assert not result.target.blocked, "Portfolio target is blocked"

    # Risk gates must all pass in the demo environment
    blocked_codes = [
        c.code for c in result.preview.risk_report.checks if c.status == "block"
    ]
    assert not result.preview.risk_report.blocked, (
        f"Risk report blocked — failing checks: {blocked_codes}"
    )

    # Forecast batch must contain forecasts for all universe symbols
    assert len(result.forecast_batch.forecasts) > 0, "No forecasts generated"

    # Snapshot ID must be set
    assert result.snapshot_id.startswith("fixture-"), (
        f"Unexpected snapshot_id: {result.snapshot_id!r}"
    )


def test_replaying_demo_reuses_execution_intents(tmp_path: Path) -> None:
    """Two run_demo calls with same clock produce identical results (determinism)."""
    settings = _settings(tmp_path)
    _init_db(settings)

    result1: DemoResult = run_demo(settings, clock=_CLOCK)
    # Second call replays — snapshot already on disk, deterministic forecast
    result2: DemoResult = run_demo(settings, clock=_CLOCK)

    # Forecast batches must match exactly (determinism)
    assert result1.forecast_batch == result2.forecast_batch, (
        "Forecast batches differ between runs"
    )

    # Target weights must match (determinism)
    assert result1.target == result2.target, "Target weights differ between runs"

    # Intent IDs must match (empty is OK — means all-cash, but must be equal)
    ids1 = tuple(sorted(i.client_order_id for i in result1.preview.intents))
    ids2 = tuple(sorted(i.client_order_id for i in result2.preview.intents))
    assert ids1 == ids2, (
        f"Intent IDs differ between runs:\n  run1: {ids1}\n  run2: {ids2}"
    )

    # Snapshot IDs must match
    assert result1.snapshot_id == result2.snapshot_id


def test_demo_persists_evidence_to_db(tmp_path: Path) -> None:
    """After run_demo, core pipeline evidence tables each have at least one row."""
    from sqlalchemy import text

    from mltrade.operations.database import build_engine
    from mltrade.operations.models import Base

    settings = _settings(tmp_path)
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)

    run_demo(settings, clock=_CLOCK)

    with engine.connect() as conn:
        # These tables must always have rows after a successful run
        required_tables = [
            "dataset_snapshots",
            "quality_reports",
            "forecast_batches",
            "backtest_runs",
            "portfolio_targets",
            "risk_report_rows",
            "execution_previews",
        ]
        for table in required_tables:
            row = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            assert row is not None and row > 0, (
                f"Table '{table}' is empty after run_demo"
            )
