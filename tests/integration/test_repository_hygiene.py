"""Repository hygiene checks.

These tests verify that runtime artefacts are properly excluded from version
control. They read project metadata files (`.gitignore`) directly and assert
the presence of required entries; they do not modify any files.
"""

from __future__ import annotations

from pathlib import Path


def test_runtime_artifacts_are_gitignored() -> None:
    """All runtime artefact directories and patterns are in .gitignore."""
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    for entry in ("data/", "artifacts/", "mlruns/", "*.db", ".claude/"):
        assert entry in ignored, f"Expected {entry!r} in .gitignore"
