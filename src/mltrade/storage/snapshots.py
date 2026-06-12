import os
from pathlib import Path
from uuid import uuid4

from mltrade.storage.manifests import (
    DatasetManifest,
    require_safe_path_segment,
)


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def snapshot_dir(self, dataset: str, snapshot_id: str) -> Path:
        safe_dataset = require_safe_path_segment(dataset)
        safe_snapshot_id = require_safe_path_segment(snapshot_id)
        return self._root / safe_dataset / safe_snapshot_id

    def save_manifest(self, manifest: DatasetManifest) -> Path:
        directory = self.snapshot_dir(manifest.dataset, manifest.snapshot_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "manifest.json"
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {target}")

        temporary = directory / f".manifest-{uuid4().hex}.tmp"
        payload = manifest.model_dump_json(indent=2)
        temporary.write_text(payload + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())

        try:
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target

    def load_manifest(self, dataset: str, snapshot_id: str) -> DatasetManifest:
        target = self.snapshot_dir(dataset, snapshot_id) / "manifest.json"
        return DatasetManifest.model_validate_json(
            target.read_text(encoding="utf-8")
        )
