import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from mltrade.storage.manifests import (
    DatasetManifest,
    require_safe_path_segment,
)


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def snapshot_dir(self, dataset: str, snapshot_id: str) -> Path:
        safe_dataset = require_safe_path_segment(dataset)
        safe_snapshot_id = require_safe_path_segment(snapshot_id)
        candidate = (self._root / safe_dataset / safe_snapshot_id).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError("snapshot path resolves outside snapshot root")
        return candidate

    @contextmanager
    def _open_snapshot_dir(
        self,
        dataset: str,
        snapshot_id: str,
    ) -> Iterator[int]:
        safe_dataset = require_safe_path_segment(dataset)
        safe_snapshot_id = require_safe_path_segment(snapshot_id)
        self._root.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current_fd = os.open(self._root, flags)
        try:
            try:
                for segment in (safe_dataset, safe_snapshot_id):
                    try:
                        os.mkdir(segment, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                    next_fd = os.open(segment, flags, dir_fd=current_fd)
                    os.close(current_fd)
                    current_fd = next_fd
            except OSError as error:
                raise ValueError(
                    "snapshot path resolves outside snapshot root"
                ) from error
            yield current_fd
        finally:
            os.close(current_fd)

    def save_manifest(self, manifest: DatasetManifest) -> Path:
        manifest = DatasetManifest.model_validate(manifest.model_dump())
        directory = self._root / manifest.dataset / manifest.snapshot_id
        target = directory / "manifest.json"
        temporary_name = f".manifest-{uuid4().hex}.tmp"
        payload = manifest.model_dump_json(indent=2)
        with self._open_snapshot_dir(
            manifest.dataset,
            manifest.snapshot_id,
        ) as directory_fd:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(
                    temporary_name,
                    "manifest.json",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise FileExistsError(
                    f"snapshot already exists: {target}"
                ) from error
            finally:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.fsync(directory_fd)
        return target

    def load_manifest(self, dataset: str, snapshot_id: str) -> DatasetManifest:
        target = self.snapshot_dir(dataset, snapshot_id) / "manifest.json"
        return DatasetManifest.model_validate_json(
            target.read_text(encoding="utf-8")
        )
