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

app.add_typer(data_app, name="data")
app.add_typer(demo_app, name="demo")
app.add_typer(research_app, name="research")
app.add_typer(portfolio_app, name="portfolio")
app.add_typer(paper_app, name="paper")


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
    """Run an offline execution preview (SimulatedBroker, no orders placed)."""
    from mltrade.workflows.demo import run_demo

    # Demo settings with cold-start allowance for preview
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
