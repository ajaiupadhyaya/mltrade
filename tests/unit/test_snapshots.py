from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import PydanticDeprecatedSince20, ValidationError

from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore


def make_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="daily_prices",
        snapshot_id="2026-06-12T210000Z",
        created_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
        source="test-fixture",
        schema_version=1,
        row_count=2,
        content_sha256="a" * 64,
        data_files=("part-000.parquet",),
    )


def test_manifest_round_trips(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    manifest = make_manifest()

    saved_path = store.save_manifest(manifest)
    loaded = store.load_manifest(manifest.dataset, manifest.snapshot_id)

    assert saved_path.name == "manifest.json"
    assert loaded == manifest


def test_existing_snapshot_cannot_be_overwritten(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    manifest = make_manifest()
    store.save_manifest(manifest)

    with pytest.raises(FileExistsError):
        store.save_manifest(manifest)


def test_snapshot_id_rejects_parent_directory_segment() -> None:
    values = make_manifest().model_dump()
    values["snapshot_id"] = ".."

    with pytest.raises(ValidationError, match="safe path segment"):
        DatasetManifest.model_validate(values)


def test_store_rejects_unsafe_lookup_segments(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    with pytest.raises(ValueError, match="safe path segment"):
        store.load_manifest("daily_prices", "..")


def test_manifest_rejects_unchecked_updates() -> None:
    manifest = make_manifest()

    with pytest.raises(TypeError, match="DatasetManifest cannot be updated"):
        manifest.model_copy(update={"row_count": -1})


def test_manifest_rejects_legacy_unchecked_updates() -> None:
    manifest = make_manifest()

    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(
            TypeError,
            match="DatasetManifest cannot be updated",
        ):
            manifest.copy(update={"row_count": -1})


def test_store_rejects_symlinked_dataset_directory(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "daily_prices").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside snapshot root"):
        SnapshotStore(root).save_manifest(make_manifest())

    assert list(outside.iterdir()) == []
