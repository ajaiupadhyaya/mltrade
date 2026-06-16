"""Safe runtime + git provenance capture for experiment runs.

Captures the git commit, a hash of any uncommitted diff, the Python/runtime
environment, and the installed dependency versions.  It NEVER records
environment variable values, credentials, database URLs, or the diff body
itself — only a hash of the diff.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform as _platform
import subprocess
from pathlib import Path

from mltrade.experiments.records import RunProvenance


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_diff(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _mltrade_version() -> str:
    try:
        return importlib.metadata.version("mltrade")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "0+unknown"


def _dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name is None:
            continue
        versions[name] = dist.version
    return dict(sorted(versions.items()))


def capture_provenance(
    repo_root: Path,
    *,
    command: tuple[str, ...],
) -> RunProvenance:
    """Capture git + runtime provenance for an experiment run.

    Resilient outside a git checkout (e.g. an installed container image): if git
    is unavailable the commit falls back to a sentinel and the run is treated as
    clean, so identity stays content-addressed by spec + dataset.
    """
    try:
        commit = _git_commit(repo_root)
        diff = _git_diff(repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        commit = "0" * 40
        diff = ""
    git_dirty = bool(diff)
    git_diff_sha256 = (
        hashlib.sha256(diff.encode("utf-8")).hexdigest() if git_dirty else None
    )
    return RunProvenance(
        git_commit=commit,
        git_dirty=git_dirty,
        git_diff_sha256=git_diff_sha256,
        python_version=_platform.python_version(),
        platform=_platform.platform(),
        mltrade_version=_mltrade_version(),
        dependencies=_dependencies(),
        command=command,
    )
