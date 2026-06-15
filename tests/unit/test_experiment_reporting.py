import statistics
from datetime import UTC, date, datetime
from decimal import Decimal

from mltrade.backtest.reporting import BacktestResult, CostSummary, EvaluationWindow
from mltrade.experiments.records import ExperimentRunRecord, RunProvenance
from mltrade.experiments.reporting import build_report_json, build_report_markdown
from mltrade.experiments.runner import build_run_metrics


def _cost(bps: str, sharpe: float) -> CostSummary:
    return CostSummary(
        cost_bps=Decimal(bps),
        annualized_return=0.1,
        annualized_volatility=0.1,
        sharpe=sharpe,
        max_drawdown=-0.2,
        total_costs=10.0,
        turnover=0.3,
        hit_rate=0.5,
    )


def _window(sharpe: float) -> EvaluationWindow:
    return EvaluationWindow(
        start_session=date(2021, 1, 1),
        end_session=date(2021, 12, 31),
        sessions=252,
        annualized_return=0.1,
        annualized_volatility=0.1,
        sharpe=sharpe,
        max_drawdown=-0.2,
        total_costs=10.0,
        turnover=0.3,
        hit_rate=0.5,
    )


def _backtest() -> BacktestResult:
    return BacktestResult(
        sessions=600,
        annualized_return=0.12,
        annualized_volatility=0.1,
        sharpe=1.2,
        max_drawdown=-0.3,
        turnover=0.2,
        total_costs=100.0,
        hit_rate=0.55,
        cost_sensitivity={
            Decimal("2"): _cost("2", 1.3),
            Decimal("5"): _cost("5", 1.2),
            Decimal("10"): _cost("10", 1.1),
        },
        per_symbol_contribution={"SPY": 0.4, "QQQ": 0.6},
        equal_weight_return=0.08,
        cash_return=0.0,
        evaluation_windows=(_window(1.0), _window(1.4)),
    )


def _record(dirty: bool) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id="run-" + "0" * 20,
        experiment_name="ridge-baseline",
        status="complete",
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        dataset_snapshot_id="fixture-2026-06-12",
        compatibility_key="d" * 64,
        seed=42,
        started_at=datetime(2026, 6, 14, tzinfo=UTC),
        finished_at=datetime(2026, 6, 14, tzinfo=UTC),
        provenance=RunProvenance(
            git_commit="c" * 40,
            git_dirty=dirty,
            git_diff_sha256=("e" * 64) if dirty else None,
            python_version="3.13.1",
            platform="test",
            mltrade_version="0.1.0",
            dependencies={"pydantic": "2.11.0"},
            command=("mltrade", "experiment", "run"),
        ),
        parameters={"model.alpha": 1.0},
        metrics=build_run_metrics(_backtest()),
        artifacts=(),
    )


def test_objective_uses_cost_and_window_stability() -> None:
    backtest = _backtest()
    metrics = build_run_metrics(backtest)

    expected = round(
        min(
            backtest.cost_sensitivity[Decimal("5")].sharpe,
            backtest.cost_sensitivity[Decimal("10")].sharpe,
        )
        - statistics.pstdev(w.sharpe for w in backtest.evaluation_windows),
        10,
    )
    assert metrics.robust_sharpe == expected


def test_report_markdown_has_required_sections() -> None:
    markdown = build_report_markdown(_record(dirty=False), backtest=_backtest())

    for needle in (
        "# MLTrade Experiment Report",
        "Dataset snapshot",
        "Dirty worktree",
        "Robust Sharpe",
        "Not a promotion decision",
    ):
        assert needle in markdown


def test_dirty_report_warns_prominently() -> None:
    markdown = build_report_markdown(_record(dirty=True), backtest=_backtest())
    assert "⚠️" in markdown
    assert "Dirty worktree: yes" in markdown


def test_report_json_is_the_canonical_record() -> None:
    record = _record(dirty=False)
    assert build_report_json(record).strip().startswith("{")
    assert ExperimentRunRecord.model_validate_json(
        build_report_json(record)
    ) == record
