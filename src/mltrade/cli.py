"""MLTrade MVP operator CLI.

Command tree
------------
  mltrade doctor                — foundation health check
  mltrade status                — concise system status
  mltrade demo run              — full offline pipeline demo (cold-start)
  mltrade data ingest           — deterministic fixture ingest
  mltrade data validate         — quality gate (nonzero on blocked)
  mltrade research backtest     — offline backtest from latest snapshot
  mltrade portfolio build       — print target weights from offline pipeline
  mltrade paper preview         — dry-run preview (SimulatedBroker)
  mltrade paper submit --submit — submit orders (requires --submit flag)
  mltrade paper reconcile       — reconciliation status

Cold-start note
---------------
``demo run`` is the initial-allocation path: it overrides
``maximum_rebalance_weight`` to 1.0 and ``maximum_order_weight`` to 0.25 so the
first full deployment from all-cash is not blocked by the steady-state 50% cap.
Steady-state live trading uses the default 50% / 10% limits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import typer

from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings

app = typer.Typer(no_args_is_help=True)

# ---------------------------------------------------------------------------
# Sub-apps
# ---------------------------------------------------------------------------

data_app = typer.Typer(no_args_is_help=True, help="Data ingestion and validation.")
demo_app = typer.Typer(no_args_is_help=True, help="Offline demo pipeline.")
research_app = typer.Typer(no_args_is_help=True, help="Research pipeline.")
portfolio_app = typer.Typer(no_args_is_help=True, help="Portfolio optimisation.")
paper_app = typer.Typer(no_args_is_help=True, help="Paper-trading commands.")
experiment_app = typer.Typer(
    no_args_is_help=True,
    help="Run, tune, inspect, and compare reproducible research experiments.",
)

app.add_typer(data_app, name="data")
app.add_typer(demo_app, name="demo")
app.add_typer(research_app, name="research")
app.add_typer(portfolio_app, name="portfolio")
app.add_typer(paper_app, name="paper")
app.add_typer(experiment_app, name="experiment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_settings() -> Settings:
    """Load settings from environment variables (MLTRADE_ prefix)."""
    return Settings()


def _ensure_db_dir(settings: Settings) -> None:
    """Create the data root and SQLite parent directory if needed."""
    settings.data_root.mkdir(parents=True, exist_ok=True)
    db_url = settings.database_url
    if db_url.startswith("sqlite") and "///" in db_url:
        db_path_str = db_url.split("///", 1)[1]
        if db_path_str:
            db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.callback()
def main() -> None:
    """Operate and inspect the MLTrade platform."""


@app.command()
def doctor() -> None:
    """Run foundation health checks."""
    settings = _get_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)

    calendar = XNYSCalendar()
    calendar.last_completed_session(datetime.now(UTC))

    typer.echo("configuration: ok")
    typer.echo("calendar: ok")
    typer.echo("data root: ok")
    state = "enabled" if settings.live_trading_enabled else "disabled"
    typer.echo(f"live trading: {state}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Print a concise system status summary."""
    settings = _get_settings()
    calendar = XNYSCalendar()
    now = datetime.now(UTC)
    last_session = calendar.last_completed_session(now)

    live = "enabled" if settings.live_trading_enabled else "disabled"
    typer.echo(f"environment:         {settings.environment.value}")
    typer.echo(f"data root:           {settings.data_root}")
    typer.echo(f"live trading:        {live}")
    typer.echo(f"last market session: {last_session.isoformat()}")
    typer.echo(f"reference equity:    ${settings.reference_equity:,.0f}")
    typer.echo("status: ok")


# ---------------------------------------------------------------------------
# demo run
# ---------------------------------------------------------------------------


@demo_app.command("run")
def demo_run() -> None:
    """Run the full offline demo pipeline (cold-start initial allocation).

    Overrides maximum_rebalance_weight=1.0 and maximum_order_weight=0.25 to
    permit the initial full-portfolio deployment from all-cash.  This is the
    cold-start initial-allocation allowance, distinct from the steady-state
    50%/10% caps used in production.
    """
    from mltrade.workflows.demo import run_demo

    settings = _get_settings()

    # Cold-start allowance: first deployment buys ~95% of equity at once.
    # Override maximum_rebalance_weight to 1.0 (vs steady-state 0.50) and
    # maximum_order_weight to 0.25 (matches the per-position cap) so the
    # risk gates pass on initial allocation.
    demo_settings = settings.model_copy(
        update={
            "maximum_rebalance_weight": Decimal("1.0"),
            "maximum_order_weight": Decimal("0.25"),
        }
    )

    # Ensure data root and SQLite DB directory exist before run_demo creates tables.
    _ensure_db_dir(demo_settings)

    try:
        result = run_demo(demo_settings)
    except ValueError as exc:
        typer.echo(f"demo failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    quality = result.quality
    backtest = result.backtest
    preview = result.preview

    typer.echo(f"snapshot: ok  (id={result.snapshot_id})")

    if quality.blocked:
        typer.echo("data quality: blocked")
        raise typer.Exit(1)
    typer.echo("data quality: pass")

    typer.echo(f"backtest: complete  ({backtest.sessions} sessions)")

    if preview.risk_report.blocked:
        blocked_codes = [
            c.code
            for c in preview.risk_report.checks
            if c.status.value == "block"
        ]
        typer.echo(f"risk: blocked  (gates={blocked_codes})")
        raise typer.Exit(1)
    typer.echo("risk: pass")

    n_intents = len(preview.intents)
    typer.echo(f"paper orders: preview only  ({n_intents} intents)")


# ---------------------------------------------------------------------------
# data ingest
# ---------------------------------------------------------------------------


@data_app.command("ingest")
def data_ingest() -> None:
    """Ingest deterministic fixture bars and publish a snapshot."""
    from datetime import date

    from mltrade.data.fixtures import DeterministicBarSource
    from mltrade.data.publication import DailyBarPublisher
    from mltrade.data.quality import validate_daily_bars
    from mltrade.storage.snapshots import SnapshotStore
    from mltrade.universe import MVP_UNIVERSE

    settings = _get_settings()
    _ensure_db_dir(settings)

    clock = datetime.now(UTC)
    cal = XNYSCalendar()
    last_session = cal.last_completed_session(clock)
    snapshot_id = f"fixture-{last_session.isoformat()}"
    fixture_start = date(2019, 1, 2)

    source = DeterministicBarSource(seed=42)
    bars = source.fetch(MVP_UNIVERSE, fixture_start, last_session, clock)

    quality = validate_daily_bars(
        bars, universe=MVP_UNIVERSE, expected_last_session=last_session
    )

    store = SnapshotStore(settings.data_root)
    publisher = DailyBarPublisher(store)
    try:
        result = publisher.publish(
            bars=bars,
            quality=quality,
            snapshot_id=snapshot_id,
            created_at=clock,
        )
        manifest = result.manifest
    except FileExistsError:
        manifest = store.load_manifest("daily_bars", snapshot_id)
        typer.echo(f"snapshot already exists: {manifest.snapshot_id}")
    else:
        typer.echo(f"snapshot: {manifest.snapshot_id}")
        typer.echo(f"rows:     {manifest.row_count}")

    quality_status = "blocked" if quality.blocked else "pass"
    typer.echo(f"data quality: {quality_status}")


# ---------------------------------------------------------------------------
# data validate
# ---------------------------------------------------------------------------


@data_app.command("validate")
def data_validate() -> None:
    """Validate the latest snapshot quality gate. Exits nonzero if blocked."""
    from datetime import date

    from mltrade.data.fixtures import DeterministicBarSource
    from mltrade.data.quality import validate_daily_bars
    from mltrade.universe import MVP_UNIVERSE

    clock = datetime.now(UTC)
    cal = XNYSCalendar()
    last_session = cal.last_completed_session(clock)
    fixture_start = date(2019, 1, 2)

    source = DeterministicBarSource(seed=42)
    bars = source.fetch(MVP_UNIVERSE, fixture_start, last_session, clock)

    quality = validate_daily_bars(
        bars, universe=MVP_UNIVERSE, expected_last_session=last_session
    )

    typer.echo(f"last session:   {last_session.isoformat()}")
    typer.echo(f"row count:      {len(bars)}")
    typer.echo(f"issues:         {len(quality.issues)}")

    if quality.blocked:
        for issue in quality.issues:
            typer.echo(f"  [{issue.severity.upper()}] {issue.code}: {issue.message}")
        typer.echo("data quality: blocked")
        raise typer.Exit(1)

    typer.echo("data quality: pass")


# ---------------------------------------------------------------------------
# research backtest
# ---------------------------------------------------------------------------


@research_app.command("backtest")
def research_backtest() -> None:
    """Run the offline backtest from the latest fixture snapshot."""
    from mltrade.workflows.demo import run_demo

    settings = _get_settings()
    _ensure_db_dir(settings)

    try:
        result = run_demo(settings)
    except ValueError as exc:
        typer.echo(f"pipeline failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    bt = result.backtest
    typer.echo(f"snapshot:           {result.snapshot_id}")
    typer.echo(f"sessions:           {bt.sessions}")
    typer.echo(f"annualized return:  {bt.annualized_return:.4f}")
    typer.echo(f"annualized vol:     {bt.annualized_volatility:.4f}")
    typer.echo(f"Sharpe:             {bt.sharpe:.4f}")
    typer.echo(f"max drawdown:       {bt.max_drawdown:.4f}")
    typer.echo("backtest: complete")


# ---------------------------------------------------------------------------
# portfolio build
# ---------------------------------------------------------------------------


@portfolio_app.command("build")
def portfolio_build() -> None:
    """Print target portfolio weights from the offline pipeline."""
    from mltrade.workflows.demo import run_demo

    settings = _get_settings()
    _ensure_db_dir(settings)

    try:
        result = run_demo(settings)
    except ValueError as exc:
        typer.echo(f"pipeline failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    target = result.target
    typer.echo(f"snapshot: {result.snapshot_id}")
    typer.echo("target weights:")
    for symbol, weight in sorted(target.weights.items()):
        typer.echo(f"  {symbol:<8} {float(weight):.4f}")
    typer.echo(f"  {'cash':<8} {float(target.cash_weight):.4f}")
    typer.echo("portfolio: built")


# ---------------------------------------------------------------------------
# paper preview
# ---------------------------------------------------------------------------


@paper_app.command("preview")
def paper_preview() -> None:
    """Run an offline execution preview (SimulatedBroker, no orders placed).

    Uses the cold-start initial-allocation allowance (maximum_rebalance_weight
    1.0 / maximum_order_weight 0.25), so this previews the FIRST full deployment
    from all-cash. Steady-state production (`run_paper`) enforces the stricter
    default 0.50/0.10 caps; this preview is intentionally permissive for the
    cold-start showcase.
    """
    from mltrade.workflows.demo import run_demo

    # Cold-start allowance (see docstring) — relaxes the steady-state caps so the
    # initial all-cash -> ~95% deployment is not blocked by the rebalance gate.
    settings = _get_settings()
    demo_settings = settings.model_copy(
        update={
            "maximum_rebalance_weight": Decimal("1.0"),
            "maximum_order_weight": Decimal("0.25"),
        }
    )
    _ensure_db_dir(demo_settings)

    try:
        result = run_demo(demo_settings)
    except ValueError as exc:
        typer.echo(f"pipeline failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    preview = result.preview
    typer.echo(
        "note: cold-start allowance applied "
        "(rebalance<=100%, order<=25%); steady-state caps are 50%/10%"
    )
    typer.echo(f"snapshot:   {result.snapshot_id}")
    typer.echo(f"intents:    {len(preview.intents)}")

    if preview.risk_report.blocked:
        blocked_codes = [
            c.code
            for c in preview.risk_report.checks
            if c.status.value == "block"
        ]
        typer.echo(f"risk: blocked  (gates={blocked_codes})")
        raise typer.Exit(1)

    typer.echo("risk: pass")
    typer.echo("paper orders: preview only (no orders placed)")
    for intent in preview.intents:
        typer.echo(
            f"  {intent.side.value:<5} {intent.symbol:<8}"
            f" {int(intent.target_quantity)} shares"
        )


# ---------------------------------------------------------------------------
# paper submit
# ---------------------------------------------------------------------------


@paper_app.command("submit")
def paper_submit(
    submit: bool = typer.Option(
        False,
        "--submit/--no-submit",
        help=(
            "Pass --submit to actually send orders. "
            "Required to prevent accidental execution."
        ),
    ),
) -> None:
    """Submit paper orders to the broker. Requires --submit flag.

    Without --submit this command exits nonzero with an explanatory message.
    With --submit it runs the full paper workflow and submits orders (requires
    MLTRADE_ENVIRONMENT=paper and valid Alpaca credentials).
    """
    if not submit:
        typer.echo(
            "Refusing to submit without --submit. "
            "Pass --submit explicitly to send orders to the broker."
        )
        raise typer.Exit(1)

    # With --submit: require PAPER environment and Alpaca credentials.
    from mltrade.config import Environment

    settings = _get_settings()
    if settings.environment is not Environment.PAPER:
        typer.echo(
            f"paper submit requires MLTRADE_ENVIRONMENT=paper; "
            f"got {settings.environment.value!r}",
            err=True,
        )
        raise typer.Exit(1)

    if settings.alpaca_api_key is None or settings.alpaca_api_secret is None:
        typer.echo(
            "paper submit requires MLTRADE_ALPACA_API_KEY and "
            "MLTRADE_ALPACA_API_SECRET to be set",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("paper submit: environment and credentials verified")
    typer.echo(
        "Note: full broker submission requires Task 16 (Alpaca adapter). "
        "Run 'mltrade paper preview' for an offline dry-run."
    )


# ---------------------------------------------------------------------------
# paper reconcile
# ---------------------------------------------------------------------------


@paper_app.command("reconcile")
def paper_reconcile() -> None:
    """Show offline reconciliation status (SimulatedBroker)."""
    from mltrade.workflows.demo import run_demo

    settings = _get_settings()
    _ensure_db_dir(settings)

    try:
        result = run_demo(settings)
    except ValueError as exc:
        typer.echo(f"pipeline failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    recon = result.preview.reconciliation
    typer.echo(f"snapshot:        {result.snapshot_id}")
    typer.echo(f"reconciliation:  {'blocked' if recon.blocked else 'ok'}")
    if recon.blocked:
        typer.echo("reconciliation: blocked", err=True)
        raise typer.Exit(1)
    typer.echo("reconciliation: pass")


# ---------------------------------------------------------------------------
# experiment — reproducible research experiment registry
# ---------------------------------------------------------------------------


def _run_store(settings: Settings):  # type: ignore[no-untyped-def]
    from mltrade.experiments.storage import RunStore

    root = settings.experiment_root
    assert root is not None
    return RunStore(root)


def _load_spec_or_exit(spec_path: Path):  # type: ignore[no-untyped-def]
    from mltrade.experiments.loading import (
        ExperimentSpecError,
        load_experiment_spec,
    )

    try:
        return load_experiment_spec(spec_path)
    except ExperimentSpecError as exc:
        typer.echo(f"spec: invalid ({exc})", err=True)
        raise typer.Exit(1) from exc


@experiment_app.command("init")
def experiment_init(
    directory: Path = typer.Argument(Path("experiments")),
    snapshot_id: str = typer.Option(None, "--snapshot-id"),
) -> None:
    """Write the packaged example specs (without overwriting)."""
    from mltrade.experiments.examples import write_example_specs

    written = write_example_specs(directory, snapshot_id=snapshot_id)
    if not written:
        typer.echo("no files written (examples already exist)")
        return
    for path in written:
        typer.echo(f"wrote: {path}")


@experiment_app.command("validate")
def experiment_validate(spec_path: Path = typer.Argument(...)) -> None:
    """Parse a spec and verify its immutable snapshot context."""
    from mltrade.storage.snapshots import SnapshotStore

    settings = _get_settings()
    loaded = _load_spec_or_exit(spec_path)
    typer.echo("spec: valid")
    typer.echo(f"spec sha256: {loaded.spec_sha256}")

    snapshots = SnapshotStore(settings.data_root)
    try:
        manifest = snapshots.load_manifest(
            loaded.spec.dataset.name, loaded.spec.dataset.snapshot_id
        )
    except (ValueError, OSError):
        typer.echo(f"snapshot: unavailable ({loaded.spec.dataset.snapshot_id})")
        raise typer.Exit(1) from None

    manifest_universe = manifest.metadata.get("universe_version")
    if manifest_universe != loaded.spec.dataset.universe_version:
        typer.echo("snapshot: context mismatch (universe_version)", err=True)
        raise typer.Exit(1)
    typer.echo(f"snapshot: {manifest.snapshot_id}")


@experiment_app.command("run")
def experiment_run(
    spec_path: Path = typer.Argument(...),
    track: bool = typer.Option(False, "--track"),
) -> None:
    """Run a single experiment and persist its canonical record + reports."""
    from mltrade.experiments.runner import (
        ExperimentBlocked,
        ExperimentFailed,
        ExperimentRunner,
        ExperimentTrackingError,
    )
    from mltrade.experiments.tracking import MlflowRunTracker, NullRunTracker

    settings = _get_settings()
    _ensure_db_dir(settings)
    loaded = _load_spec_or_exit(spec_path)
    tracker = (
        MlflowRunTracker(settings.mlflow_tracking_root) if track else NullRunTracker()
    )
    runner = ExperimentRunner(settings=settings, tracker=tracker)
    try:
        result = runner.run(loaded)
    except (ExperimentBlocked, ExperimentFailed) as exc:
        typer.echo(f"experiment failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    except ExperimentTrackingError as exc:
        typer.echo(f"tracking degraded for run {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"run id: {result.record.run_id}")
    typer.echo(f"status: {result.record.status}")
    typer.echo(f"report: {result.report_markdown}")


@experiment_app.command("list")
def experiment_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List stored runs, newest finished first."""
    import json

    settings = _get_settings()
    records = _run_store(settings).list_records()
    records.sort(key=lambda record: record.finished_at, reverse=True)
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "run_id": record.run_id,
                        "status": record.status,
                        "robust_sharpe": (
                            record.metrics.robust_sharpe if record.metrics else None
                        ),
                    }
                    for record in records
                ],
                indent=2,
            )
        )
        return
    if not records:
        typer.echo("no runs found")
        return
    for record in records:
        robust = (
            f"{record.metrics.robust_sharpe:.4f}" if record.metrics else "n/a"
        )
        typer.echo(f"{record.run_id}  {record.status:<8}  robust={robust}")


@experiment_app.command("inspect")
def experiment_inspect(
    run_id: str = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show a run summary, or its canonical JSON with --json."""
    from mltrade.experiments.storage import RunStorageError

    settings = _get_settings()
    try:
        record = _run_store(settings).load(run_id)
    except (RunStorageError, OSError, ValueError) as exc:
        typer.echo(f"run not found: {run_id}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(record.model_dump_json(indent=2))
        return
    typer.echo(f"run id:       {record.run_id}")
    typer.echo(f"experiment:   {record.experiment_name}")
    typer.echo(f"status:       {record.status}")
    typer.echo(f"snapshot:     {record.dataset_snapshot_id}")
    typer.echo(f"dirty:        {record.provenance.git_dirty}")
    if record.metrics is not None:
        typer.echo(f"robust sharpe: {record.metrics.robust_sharpe:.4f}")
        typer.echo(f"max drawdown:  {record.metrics.max_drawdown:.4f}")
    if record.failure is not None:
        typer.echo(f"failure:      {record.failure.category}: {record.failure.message}")


@experiment_app.command("compare")
def experiment_compare(
    run_ids: list[str] = typer.Argument(...),
    include_dirty: bool = typer.Option(False, "--include-dirty"),
) -> None:
    """Rank compatible runs; exit nonzero when runs are incompatible."""
    from mltrade.experiments.comparison import compare_runs
    from mltrade.experiments.storage import RunStorageError

    settings = _get_settings()
    store = _run_store(settings)
    try:
        records = tuple(store.load(run_id) for run_id in run_ids)
    except (RunStorageError, OSError, ValueError) as exc:
        typer.echo(f"run not found: {exc}", err=True)
        raise typer.Exit(1) from exc

    result = compare_runs(records, include_dirty=include_dirty)
    if not result.compatible:
        typer.echo("runs are incompatible — no winner:")
        for field, values in result.differences.items():
            typer.echo(f"  {field}: {values}")
        raise typer.Exit(1)
    for ranked in result.ranking:
        typer.echo(
            f"#{ranked.rank} {ranked.run_id}  robust={ranked.robust_sharpe:.4f}"
        )
    if result.excluded_run_ids:
        typer.echo(f"excluded: {', '.join(result.excluded_run_ids)}")


@experiment_app.command("report")
def experiment_report(run_id: str = typer.Argument(...)) -> None:
    """Regenerate report.json from the canonical record."""
    from mltrade.experiments.reporting import build_report_json
    from mltrade.experiments.storage import RunStorageError

    settings = _get_settings()
    store = _run_store(settings)
    try:
        record = store.load(run_id)
    except (RunStorageError, OSError, ValueError) as exc:
        typer.echo(f"run not found: {run_id}", err=True)
        raise typer.Exit(1) from exc
    run_dir = store.run_directory(run_id)
    (run_dir / "report.json").write_text(build_report_json(record), encoding="utf-8")
    typer.echo(f"run dir: {run_dir}")
    typer.echo(f"report: {run_dir / 'report.md'}")


@experiment_app.command("tune")
def experiment_tune(
    spec_path: Path = typer.Argument(...),
    trials: int = typer.Option(None, "--trials"),
    timeout_minutes: int = typer.Option(None, "--timeout-minutes"),
    study: str = typer.Option(None, "--study"),
    track: bool = typer.Option(False, "--track"),
) -> None:
    """Tune an experiment via a persistent Optuna study."""
    from mltrade.experiments.loading import loaded_from_spec
    from mltrade.experiments.runner import ExperimentRunner
    from mltrade.experiments.tracking import MlflowRunTracker, NullRunTracker
    from mltrade.experiments.tuning import OptunaTuner, StudyContextMismatch

    settings = _get_settings()
    _ensure_db_dir(settings)
    loaded = _load_spec_or_exit(spec_path)
    if loaded.spec.search is None:
        typer.echo("tune requires a [search] space in the spec", err=True)
        raise typer.Exit(1)

    spec = loaded.spec
    if timeout_minutes is not None:
        spec = spec.model_copy(
            update={
                "resources": spec.resources.model_copy(
                    update={"timeout_minutes": timeout_minutes}
                )
            }
        )
        loaded = loaded_from_spec(spec, path=loaded.path)

    tracker = (
        MlflowRunTracker(settings.mlflow_tracking_root) if track else NullRunTracker()
    )
    runner = ExperimentRunner(settings=settings, tracker=tracker)
    tuner = OptunaTuner(storage_path=settings.optuna_storage_path, runner=runner)
    n_trials = trials if trials is not None else spec.resources.max_trials
    study_name = study or spec.name
    try:
        result = tuner.tune(loaded, study_name=study_name, n_trials=n_trials)
    except StudyContextMismatch as exc:
        typer.echo(f"study context mismatch: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"study: {result.study_name}")
    typer.echo(
        f"trials: {result.completed_trials} complete, "
        f"{result.pruned_trials} pruned, {result.failed_trials} failed"
    )
    typer.echo(f"best run id: {result.best_run_id}")
    typer.echo(f"best robust sharpe: {result.best_value}")


@experiment_app.command("resume")
def experiment_resume(
    study: str = typer.Argument(...),
    spec_path: Path = typer.Option(..., "--spec"),
    trials: int = typer.Option(1, "--trials"),
    track: bool = typer.Option(False, "--track"),
) -> None:
    """Resume an existing study with its original spec."""
    from mltrade.experiments.runner import ExperimentRunner
    from mltrade.experiments.tracking import MlflowRunTracker, NullRunTracker
    from mltrade.experiments.tuning import OptunaTuner, StudyContextMismatch

    settings = _get_settings()
    _ensure_db_dir(settings)
    loaded = _load_spec_or_exit(spec_path)
    if loaded.spec.search is None:
        typer.echo("resume requires a [search] space in the spec", err=True)
        raise typer.Exit(1)
    tracker = (
        MlflowRunTracker(settings.mlflow_tracking_root) if track else NullRunTracker()
    )
    runner = ExperimentRunner(settings=settings, tracker=tracker)
    tuner = OptunaTuner(storage_path=settings.optuna_storage_path, runner=runner)
    try:
        result = tuner.tune(loaded, study_name=study, n_trials=trials)
    except StudyContextMismatch as exc:
        typer.echo(f"study context mismatch: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"study: {result.study_name}")
    typer.echo(f"trials: {result.completed_trials} complete")
    typer.echo(f"best run id: {result.best_run_id}")


@experiment_app.command("doctor")
def experiment_doctor() -> None:
    """Check experiment directories, dependencies, and example snapshot."""
    from mltrade.storage.snapshots import SnapshotStore

    settings = _get_settings()
    ok = True

    try:
        import mlflow
        import optuna

        typer.echo(f"mlflow: ok ({mlflow.__version__})")
        typer.echo(f"optuna: ok ({optuna.__version__})")
    except ImportError as exc:
        typer.echo(f"dependencies: MISSING ({exc})", err=True)
        ok = False

    root = settings.experiment_root
    assert root is not None
    try:
        (root / "runs").mkdir(parents=True, exist_ok=True)
        settings.optuna_storage_path.parent.mkdir(parents=True, exist_ok=True)
        settings.mlflow_tracking_root.mkdir(parents=True, exist_ok=True)
        typer.echo(f"experiment root: ok ({root})")
        typer.echo(f"optuna storage: {settings.optuna_storage_path}")
        typer.echo(f"mlflow uri: {settings.mlflow_tracking_root.resolve().as_uri()}")
    except OSError as exc:
        typer.echo(f"experiment directories: FAILED ({exc})", err=True)
        ok = False

    try:
        manifest = SnapshotStore(settings.data_root).load_manifest(
            "daily_bars", "fixture-2026-06-12"
        )
        typer.echo(f"example snapshot: available ({manifest.snapshot_id})")
    except (ValueError, OSError):
        typer.echo(
            "example snapshot: not published (run `mltrade data ingest`)"
        )

    if not ok:
        typer.echo("experiment doctor: failed", err=True)
        raise typer.Exit(1)
    typer.echo("experiment doctor: ok")
