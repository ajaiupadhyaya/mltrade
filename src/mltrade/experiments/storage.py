"""Atomic, content-checked persistence of canonical experiment run records.

Each run is published as ``<root>/runs/<run_id>/run.json`` (plus any artifact
files) via a temp-dir + fsync + atomic-rename protocol.  Re-saving an identical
record is idempotent; saving different content under an existing run id is
rejected, as are symlinked run directories.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from mltrade.experiments.records import ExperimentRunRecord
from mltrade.storage.manifests import require_safe_path_segment


class RunStorageError(RuntimeError):
    pass


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class RunStore:
    """Persist and load canonical :class:`ExperimentRunRecord` objects."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    @property
    def runs_root(self) -> Path:
        return self._root / "runs"

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_root / require_safe_path_segment(run_id)

    def run_directory(self, run_id: str) -> Path:
        """Return the canonical directory for *run_id* (validated segment)."""
        return self._run_dir(run_id)

    def save(
        self,
        record: ExperimentRunRecord,
        *,
        artifacts: Mapping[str, bytes] | None = None,
    ) -> Path:
        run_dir = self._run_dir(record.run_id)
        run_json = run_dir / "run.json"

        if run_dir.is_symlink():
            raise RunStorageError(
                f"refusing to write to symlinked run directory: {run_dir}"
            )
        if run_dir.exists():
            existing = self.load(record.run_id)
            if existing == record:
                return run_json
            raise RunStorageError(
                f"run {record.run_id} already exists with different content"
            )

        self.runs_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.runs_root / f".{record.run_id}.tmp-{uuid4().hex}"
        tmp_dir.mkdir()
        try:
            _write_bytes(
                tmp_dir / "run.json",
                (record.model_dump_json(indent=2) + "\n").encode("utf-8"),
            )
            for name, data in (artifacts or {}).items():
                _write_bytes(tmp_dir / require_safe_path_segment(name), data)
            _fsync_dir(tmp_dir)
            tmp_dir.replace(run_dir)
            _fsync_dir(self.runs_root)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return run_json

    def load(self, run_id: str) -> ExperimentRunRecord:
        run_dir = self._run_dir(run_id)
        if run_dir.is_symlink():
            raise RunStorageError(
                f"refusing to read symlinked run directory: {run_dir}"
            )
        run_json = run_dir / "run.json"
        if run_json.is_symlink():
            raise RunStorageError(f"refusing to read symlinked run.json: {run_json}")
        return ExperimentRunRecord.model_validate_json(
            run_json.read_text(encoding="utf-8")
        )

    def replace_tracking_state(self, record: ExperimentRunRecord) -> Path:
        """Rewrite an existing run's ``run.json`` (tracking-state update only).

        Used after a tracking degradation to record ``tracking_status`` /
        ``failure`` without disturbing the rest of the canonical evidence.
        """
        run_dir = self._run_dir(record.run_id)
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise RunStorageError(f"run directory not found: {run_dir}")
        run_json = run_dir / "run.json"
        tmp = run_dir / f".run-{uuid4().hex}.tmp"
        _write_bytes(
            tmp, (record.model_dump_json(indent=2) + "\n").encode("utf-8")
        )
        tmp.replace(run_json)
        _fsync_dir(run_dir)
        return run_json

    def exists(self, run_id: str) -> bool:
        run_dir = self._run_dir(run_id)
        return run_dir.is_dir() and not run_dir.is_symlink()

    def list_records(self) -> list[ExperimentRunRecord]:
        if not self.runs_root.is_dir():
            return []
        records: list[ExperimentRunRecord] = []
        for child in sorted(self.runs_root.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            try:
                records.append(self.load(child.name))
            except (RunStorageError, OSError, ValueError):
                continue
        return records
