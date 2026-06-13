"""Tests for immutable Parquet snapshot publication.

TDD: tests written first, drive the implementation in
mltrade.data.publication.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mltrade.data.bars import DailyBar
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.data.publication import DailyBarPublisher, PublishedSnapshot
from mltrade.data.quality import (
    DataQualityReport,
    IssueSeverity,
    QualityIssue,
    validate_daily_bars,
)
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.universe import MVP_UNIVERSE

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INGESTED_AT = datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
_LAST_SESSION = date(2026, 6, 12)
_RANGE_START = date(2026, 6, 2)
_CREATED_AT = datetime(2026, 6, 13, tzinfo=UTC)
_SNAPSHOT_ID = "fixture-20260612"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_bars() -> tuple[DailyBar, ...]:
    """Complete, sorted, duplicate-free grid ending 2026-06-12."""
    return DeterministicBarSource(seed=42).fetch(
        MVP_UNIVERSE,
        _RANGE_START,
        _LAST_SESSION,
        _INGESTED_AT,
    )


def _passing_report(bars: tuple[DailyBar, ...]) -> DataQualityReport:
    """Return a non-blocked DataQualityReport for bars."""
    report = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert not report.blocked, f"Expected passing report; got: {report.issues}"
    return report


def _blocked_report() -> DataQualityReport:
    """Return a blocked DataQualityReport."""
    return DataQualityReport(
        issues=(
            QualityIssue(
                code="empty_input",
                severity=IssueSeverity.BLOCK,
                message="forced block",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_bars() -> tuple[DailyBar, ...]:
    return _make_valid_bars()


@pytest.fixture()
def publisher(tmp_path: Path) -> DailyBarPublisher:
    return DailyBarPublisher(SnapshotStore(tmp_path))


# ---------------------------------------------------------------------------
# Happy-path: round-trip verifies content
# ---------------------------------------------------------------------------


def test_publish_round_trip_verifies_content(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    loaded = publisher.load_verified(published.manifest)
    assert loaded == valid_bars
    assert published.manifest.metadata["universe_version"] == "mvp-etf-v1"


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampered_parquet_is_rejected(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    parquet_path = published.data_files[0]
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content hash"):
        publisher.load_verified(published.manifest)


# ---------------------------------------------------------------------------
# Blocked quality report is rejected
# ---------------------------------------------------------------------------


def test_blocked_quality_report_raises_before_writing(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    with pytest.raises(ValueError, match="blocked"):
        publisher.publish(
            bars=valid_bars,
            quality=_blocked_report(),
            snapshot_id=_SNAPSHOT_ID,
            created_at=_CREATED_AT,
        )
    # Nothing should have been written
    snapshot_dir = SnapshotStore(tmp_path).snapshot_dir("daily_bars", _SNAPSHOT_ID)
    assert not snapshot_dir.exists()


# ---------------------------------------------------------------------------
# PublishedSnapshot shape
# ---------------------------------------------------------------------------


def test_published_snapshot_exposes_manifest_and_data_files(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    assert isinstance(published, PublishedSnapshot)
    assert isinstance(published.manifest, DatasetManifest)
    assert isinstance(published.data_files, tuple)
    assert len(published.data_files) == 1
    assert all(isinstance(p, Path) for p in published.data_files)
    assert published.data_files[0].exists()
    assert published.data_files[0].name == "daily-bars.parquet"


# ---------------------------------------------------------------------------
# Manifest fields
# ---------------------------------------------------------------------------


def test_manifest_has_correct_fields(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    manifest = published.manifest
    assert manifest.dataset == "daily_bars"
    assert manifest.snapshot_id == _SNAPSHOT_ID
    assert manifest.row_count == len(valid_bars)
    assert len(manifest.content_sha256) == 64
    assert manifest.data_files == ("daily-bars.parquet",)
    assert manifest.schema_version == 1


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_manifest_metadata_contains_expected_keys(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    meta = published.manifest.metadata
    assert "universe_version" in meta
    assert "schema_version" in meta
    assert "quality_version" in meta
    assert meta["universe_version"] == "mvp-etf-v1"


# ---------------------------------------------------------------------------
# Row count matches
# ---------------------------------------------------------------------------


def test_row_count_matches_bars_length(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    assert published.manifest.row_count == len(valid_bars)


# ---------------------------------------------------------------------------
# Exact Decimal round-trip
# ---------------------------------------------------------------------------


def test_exact_decimal_round_trip(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    """Decimal precision must survive the Parquet round-trip exactly."""
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    loaded = publisher.load_verified(published.manifest)
    for orig, got in zip(valid_bars, loaded, strict=True):
        assert orig.open == got.open, f"open mismatch: {orig.open!r} != {got.open!r}"
        assert orig.high == got.high, f"high mismatch: {orig.high!r} != {got.high!r}"
        assert orig.low == got.low, f"low mismatch: {orig.low!r} != {got.low!r}"
        assert orig.close == got.close, (
            f"close mismatch: {orig.close!r} != {got.close!r}"
        )
        assert orig.vwap == got.vwap, f"vwap mismatch: {orig.vwap!r} != {got.vwap!r}"


# ---------------------------------------------------------------------------
# Duplicate snapshot is rejected
# ---------------------------------------------------------------------------


def test_duplicate_publish_raises(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    with pytest.raises(FileExistsError):
        publisher.publish(
            bars=valid_bars,
            quality=_passing_report(valid_bars),
            snapshot_id=_SNAPSHOT_ID,
            created_at=_CREATED_AT,
        )


# ---------------------------------------------------------------------------
# PublishedSnapshot is immutable (frozen dataclass / frozen model)
# ---------------------------------------------------------------------------


def test_published_snapshot_is_immutable(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=_passing_report(valid_bars),
        snapshot_id=_SNAPSHOT_ID,
        created_at=_CREATED_AT,
    )
    with pytest.raises((TypeError, AttributeError)):
        published.manifest = published.manifest  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetManifest metadata field is immutable-friendly
# ---------------------------------------------------------------------------


def test_manifest_metadata_default_is_empty_dict() -> None:
    manifest = DatasetManifest(
        dataset="daily_bars",
        snapshot_id="test-snap",
        created_at=datetime(2026, 6, 13, tzinfo=UTC),
        source="test",
        schema_version=1,
        row_count=0,
        content_sha256="a" * 64,
    )
    assert manifest.metadata == {}


def test_manifest_metadata_round_trips_through_json() -> None:
    manifest = DatasetManifest(
        dataset="daily_bars",
        snapshot_id="test-snap",
        created_at=datetime(2026, 6, 13, tzinfo=UTC),
        source="test",
        schema_version=1,
        row_count=0,
        content_sha256="a" * 64,
        metadata={"key": "value"},
    )
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored.metadata == {"key": "value"}
