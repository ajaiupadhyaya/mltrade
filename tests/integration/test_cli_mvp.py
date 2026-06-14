"""Integration tests for the MLTrade MVP CLI (Task 15).

Tests cover:
- demo run: prints five required summary lines, exit 0, non-empty intents
- paper submit without --submit: exits nonzero, mentions --submit
- status: exits 0, mentions key fields
- data validate: exits 0 on valid fixture data
- paper preview: exits 0, prints risk/order lines
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mltrade.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# demo run
# ---------------------------------------------------------------------------


def test_demo_run_prints_acceptance_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """demo run must print all five required summary lines and exit 0."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["demo", "run"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}.\nstdout:\n{result.stdout}"
    )
    assert "snapshot: ok" in result.stdout
    assert "data quality: pass" in result.stdout
    assert "backtest: complete" in result.stdout
    assert "risk: pass" in result.stdout
    assert "paper orders: preview only" in result.stdout


def test_demo_run_intents_non_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """demo run must produce at least one intent (cold-start = full build)."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["demo", "run"])

    assert result.exit_code == 0, result.stdout
    # Summary line: "paper orders: preview only  (N intents)"
    match = re.search(
        r"paper orders: preview only.*?(\d+) intents", result.stdout
    )
    assert match is not None, (
        f"No intent count found in stdout:\n{result.stdout}"
    )
    n_intents = int(match.group(1))
    assert n_intents > 0, f"Expected > 0 intents, got {n_intents}"


def test_demo_run_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running demo run twice in the same data root must not fail."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    r1 = runner.invoke(app, ["demo", "run"])
    assert r1.exit_code == 0, r1.stdout
    r2 = runner.invoke(app, ["demo", "run"])
    assert r2.exit_code == 0, r2.stdout


# ---------------------------------------------------------------------------
# paper submit
# ---------------------------------------------------------------------------


def test_paper_submit_requires_explicit_flag() -> None:
    """paper submit without --submit must exit nonzero and mention --submit."""
    result = runner.invoke(app, ["paper", "submit"])
    assert result.exit_code != 0
    assert "--submit" in result.stdout


def test_paper_submit_no_submit_flag_explicitly() -> None:
    """paper submit --no-submit is the same as no flag: exit nonzero."""
    result = runner.invoke(app, ["paper", "submit", "--no-submit"])
    assert result.exit_code != 0
    assert "--submit" in result.stdout


def test_paper_submit_with_flag_wrong_environment() -> None:
    """paper submit --submit with non-paper env exits nonzero."""
    result = runner.invoke(app, ["paper", "submit", "--submit"])
    # Default environment is LOCAL, not PAPER -> should fail
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_exits_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """status exits 0 and prints key fields."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "environment:" in result.stdout
    assert "data root:" in result.stdout
    assert "live trading:" in result.stdout
    assert "status: ok" in result.stdout


# ---------------------------------------------------------------------------
# data validate
# ---------------------------------------------------------------------------


def test_data_validate_passes_on_fixture_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """data validate exits 0 on valid fixture data."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["data", "validate"])
    assert result.exit_code == 0, result.stdout
    assert "data quality: pass" in result.stdout


def test_data_validate_prints_session_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """data validate includes last session and row count in output."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["data", "validate"])
    assert result.exit_code == 0, result.stdout
    assert "last session:" in result.stdout
    assert "row count:" in result.stdout


# ---------------------------------------------------------------------------
# paper preview
# ---------------------------------------------------------------------------


def test_paper_preview_exits_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """paper preview exits 0, prints risk: pass and paper orders line."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["paper", "preview"])
    assert result.exit_code == 0, result.stdout
    assert "risk: pass" in result.stdout
    assert "paper orders: preview only" in result.stdout


# ---------------------------------------------------------------------------
# data ingest
# ---------------------------------------------------------------------------


def test_data_ingest_exits_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """data ingest exits 0 and reports quality pass."""
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["data", "ingest"])
    assert result.exit_code == 0, result.stdout
    assert "data quality: pass" in result.stdout
