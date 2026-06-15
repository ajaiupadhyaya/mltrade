"""Integration tests for the dashboard JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from mltrade.config import Settings
from mltrade.export import build_dashboard_payload, write_dashboard_json


def _settings(root: Path) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_root=root,
        database_url=f"sqlite:///{root / 'operations.db'}",
    )


def test_dashboard_payload_has_expected_shape(tmp_path: Path) -> None:
    payload = build_dashboard_payload(_settings(tmp_path))

    assert payload["schema_version"] == 1
    assert set(payload) >= {
        "meta",
        "backtest",
        "portfolio",
        "risk",
        "execution",
        "quality",
        "forecast",
        "experiments",
    }

    meta = payload["meta"]
    assert meta["snapshot_id"] == "fixture-2026-06-12"
    assert meta["live_trading_enabled"] is False
    assert meta["synthetic"] is True
    assert len(meta["universe"]) == 10


def test_backtest_block_carries_real_curve_and_metrics(tmp_path: Path) -> None:
    bt = build_dashboard_payload(_settings(tmp_path))["backtest"]

    # The engineered fixture produces a strongly positive Sharpe; assert a
    # tolerant lower bound rather than a brittle exact value.
    assert bt["sharpe"] > 1.5
    assert bt["sessions"] >= 1300
    assert len(bt["cost_sensitivity"]) == 3

    curve = bt["equity_curve"]
    assert len(curve) >= 2
    assert all(set(point) == {"date", "equity"} for point in curve)
    assert curve[0]["equity"] == 1_000_000.0
    assert curve[-1]["equity"] > curve[0]["equity"]
    # Dates are strictly increasing.
    dates = [point["date"] for point in curve]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_risk_execution_and_portfolio_blocks(tmp_path: Path) -> None:
    payload = build_dashboard_payload(_settings(tmp_path))

    execution = payload["execution"]
    assert execution["count"] == 10
    assert execution["preview_only"] is True

    risk = payload["risk"]
    # Steady-state caps gate two notional checks in the cold-start allocation.
    assert risk["summary"]["block"] == 2
    assert risk["blocked"] is True
    assert len(risk["checks"]) == sum(risk["summary"].values())

    portfolio = payload["portfolio"]
    assert abs(portfolio["cash_weight"] + portfolio["invested_weight"] - 1.0) < 1e-6

    # The platform ships before the experiment registry is merged on this branch.
    assert payload["experiments"]["available"] is False


def test_payload_is_deterministic(tmp_path: Path) -> None:
    first = build_dashboard_payload(_settings(tmp_path / "first"))
    second = build_dashboard_payload(_settings(tmp_path / "second"))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_write_dashboard_json_creates_parents_and_trailing_newline(
    tmp_path: Path,
) -> None:
    out = tmp_path / "nested" / "dashboard.json"
    written = write_dashboard_json(_settings(tmp_path / "data"), out)

    assert written == out
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["meta"]["platform"] == "MLTrade"
