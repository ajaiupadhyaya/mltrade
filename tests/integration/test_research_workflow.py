"""Integration test for the research workflow (run_research).

run_research operates on an already-published snapshot: no fixture generation,
no broker, no DB writes. We publish a snapshot via run_demo, load its manifest,
then exercise the research pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from mltrade.config import Environment, Settings
from mltrade.storage.snapshots import SnapshotStore
from mltrade.workflows.demo import run_demo
from mltrade.workflows.research import run_research

_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.TEST,
        data_root=tmp_path,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
    )


def test_run_research_on_published_snapshot(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    # Publish a snapshot via the demo, then load its manifest.
    demo = run_demo(settings, clock=_CLOCK)
    store = SnapshotStore(settings.data_root)
    manifest = store.load_manifest("daily_bars", demo.snapshot_id)

    result = run_research(settings, manifest)

    assert result.snapshot_id == demo.snapshot_id
    assert result.quality.blocked is False
    assert result.backtest.sessions > 250
    assert result.forecast_batch.forecasts
    # Research shares the production target code; result is a valid target.
    assert result.target.blocked is False


def test_run_research_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    demo = run_demo(settings, clock=_CLOCK)
    manifest = SnapshotStore(settings.data_root).load_manifest(
        "daily_bars", demo.snapshot_id
    )

    first = run_research(settings, manifest)
    second = run_research(settings, manifest)

    assert first.backtest == second.backtest
    assert first.target == second.target
