import hashlib

import pytest

from mltrade.experiments import provenance
from mltrade.experiments.provenance import capture_provenance


def test_capture_provenance_records_dirty_diff_hash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git_commit", lambda _: "a" * 40)
    monkeypatch.setattr(provenance, "_git_diff", lambda _: "changed\n")

    result = capture_provenance(tmp_path, command=("mltrade", "experiment", "run"))

    assert result.git_dirty is True
    assert result.git_diff_sha256 == hashlib.sha256(b"changed\n").hexdigest()
    assert result.git_commit == "a" * 40
    assert result.command == ("mltrade", "experiment", "run")
    assert "python_version" in result.model_dump()


def test_clean_worktree_has_no_diff_hash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git_commit", lambda _: "b" * 40)
    monkeypatch.setattr(provenance, "_git_diff", lambda _: "")

    result = capture_provenance(tmp_path, command=("mltrade",))

    assert result.git_dirty is False
    assert result.git_diff_sha256 is None


def test_dependencies_are_name_version_strings_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git_commit", lambda _: "c" * 40)
    monkeypatch.setattr(provenance, "_git_diff", lambda _: "")

    result = capture_provenance(tmp_path, command=("mltrade",))

    assert result.dependencies
    assert all(
        isinstance(name, str) and isinstance(version, str)
        for name, version in result.dependencies.items()
    )
    assert "pydantic" in result.dependencies
