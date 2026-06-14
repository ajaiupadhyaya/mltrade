"""Integration tests for the paper-trading workflow (Task 14).

All tests run entirely offline (no network access) by using a
:class:`~mltrade.execution.simulated.SimulatedBroker`.

Test inventory
--------------
1. test_stale_snapshot_blocks_paper_submission
   — snapshot_last_session != expected_last_session (freshness mismatch) →
     preview.risk_report.blocked=True; submit is refused.

2. test_paper_refuses_submit_without_submit_true
   — submit=False → submit_result is None, no orders sent to broker.

3. test_paper_refuses_blocked_preview
   — blocked risk report → no submission even with submit=True.

4. test_paper_requires_paper_environment
   — environment != PAPER → RuntimeError raised.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from mltrade.config import Settings
from mltrade.execution.broker import BrokerAccount
from mltrade.execution.simulated import SimulatedBroker, SubmitOutcome
from mltrade.workflows.paper import PaperResult, run_paper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOCK = datetime(2026, 6, 13, 22, 0, tzinfo=UTC)


def _demo_settings(tmp_path: Path) -> Settings:
    db_path = tmp_path / "ops.db"
    return Settings(
        environment="local",
        data_root=tmp_path / "data",
        database_url=f"sqlite+pysqlite:///{db_path}",
        reference_equity="1000000",
    )


def _paper_settings(tmp_path: Path) -> Settings:
    db_path = tmp_path / "ops.db"
    return Settings(
        environment="paper",
        data_root=tmp_path / "data",
        database_url=f"sqlite+pysqlite:///{db_path}",
        reference_equity="1000000",
    )


def _init_db(settings: Settings) -> None:
    from mltrade.operations.database import build_engine
    from mltrade.operations.models import Base

    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)


def _simulated_broker(equity: Decimal = Decimal("1000000")) -> SimulatedBroker:
    account = BrokerAccount(
        id="test-paper-001",
        status="ACTIVE",
        cash=equity,
        equity=equity,
        account_blocked=False,
        trading_blocked=False,
        pattern_day_trader=False,
    )
    return SimulatedBroker(account, default_outcome=SubmitOutcome.COMPLETE_FILL)


def _build_manifest_via_demo(tmp_path: Path) -> object:
    """Run the demo workflow to generate a real snapshot on disk, return manifest."""
    from mltrade.calendar import XNYSCalendar
    from mltrade.data.fixtures import DeterministicBarSource
    from mltrade.data.publication import DailyBarPublisher
    from mltrade.data.quality import validate_daily_bars
    from mltrade.storage.snapshots import SnapshotStore
    from mltrade.universe import MVP_UNIVERSE

    cal = XNYSCalendar()
    last_session = cal.last_completed_session(_CLOCK)
    snapshot_id = f"fixture-{last_session.isoformat()}"

    data_root = tmp_path / "data"
    source = DeterministicBarSource(seed=42)
    bars = source.fetch(MVP_UNIVERSE, date(2019, 1, 2), last_session, _CLOCK)
    quality = validate_daily_bars(
        bars, universe=MVP_UNIVERSE, expected_last_session=last_session
    )
    assert not quality.blocked

    store = SnapshotStore(data_root)
    publisher = DailyBarPublisher(store)
    try:
        published = publisher.publish(
            bars=bars,
            quality=quality,
            snapshot_id=snapshot_id,
            created_at=_CLOCK,
        )
        return published.manifest
    except FileExistsError:
        return store.load_manifest("daily_bars", snapshot_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stale_snapshot_blocks_paper_submission(tmp_path: Path) -> None:
    """Stale snapshot (last_session != expected) → risk report blocked.

    We build a real manifest but pass a *different* expected_last_session
    via a patched clock that is one week in the future, so the snapshot looks
    stale relative to what the calendar expects.
    """
    settings = _paper_settings(tmp_path)
    _init_db(settings)

    manifest = _build_manifest_via_demo(tmp_path)

    # Use a clock one year in the future so expected_last_session ≠ snapshot
    future_clock = datetime(2027, 6, 13, 22, 0, tzinfo=UTC)
    broker = _simulated_broker()

    result: PaperResult = run_paper(
        settings,
        manifest,  # type: ignore[arg-type]
        broker=broker,
        submit=True,
        clock=future_clock,
    )

    # The freshness check must fire: snapshot_last_session != expected_last_session
    assert result.preview.risk_report.blocked, (
        "Expected risk report to be blocked due to stale snapshot"
    )
    freshness = result.preview.risk_report.by_code("snapshot_freshness")
    assert freshness.status == "block", (
        f"Expected snapshot_freshness=BLOCK, got {freshness.status}"
    )

    # Even though submit=True, no orders must have been placed
    assert result.submit_result is None, (
        "submit_result should be None when risk report is blocked"
    )
    assert len(broker.list_orders()) == 0, "Broker should have no orders"


def test_paper_refuses_submit_without_submit_true(tmp_path: Path) -> None:
    """submit=False → dry run, submit_result is None, broker unchanged."""
    settings = _paper_settings(tmp_path)
    _init_db(settings)

    manifest = _build_manifest_via_demo(tmp_path)
    broker = _simulated_broker()

    result: PaperResult = run_paper(
        settings,
        manifest,  # type: ignore[arg-type]
        broker=broker,
        submit=False,
        clock=_CLOCK,
    )

    # Dry run: no submission attempted
    assert result.submit_result is None, (
        "Expected submit_result=None in dry-run (submit=False)"
    )
    assert len(broker.list_orders()) == 0, "Broker should have no orders on dry run"


def test_paper_refuses_blocked_preview(tmp_path: Path) -> None:
    """Blocked risk report → no submission even with submit=True."""
    settings = _paper_settings(tmp_path)
    _init_db(settings)

    manifest = _build_manifest_via_demo(tmp_path)
    broker = _simulated_broker()

    # Force a stale snapshot so preview is blocked
    future_clock = datetime(2027, 6, 13, 22, 0, tzinfo=UTC)

    result: PaperResult = run_paper(
        settings,
        manifest,  # type: ignore[arg-type]
        broker=broker,
        submit=True,
        clock=future_clock,
    )

    assert result.preview.risk_report.blocked, "Expected blocked preview"
    assert result.submit_result is None, (
        "No orders should be submitted when preview is blocked"
    )
    assert len(broker.list_orders()) == 0


def test_paper_requires_paper_environment(tmp_path: Path) -> None:
    """Environment != PAPER → RuntimeError raised before any work."""
    settings = _demo_settings(tmp_path)  # environment="local"
    _init_db(settings)

    manifest = _build_manifest_via_demo(tmp_path)
    broker = _simulated_broker()

    with pytest.raises(RuntimeError, match="run_paper requires environment=PAPER"):
        run_paper(settings, manifest, broker=broker, submit=False, clock=_CLOCK)  # type: ignore[arg-type]
