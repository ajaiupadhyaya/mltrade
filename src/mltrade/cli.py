from datetime import UTC, datetime

import typer

from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Operate and inspect the MLTrade platform."""


@app.command()
def doctor() -> None:
    settings = Settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)

    calendar = XNYSCalendar()
    calendar.last_completed_session(datetime.now(UTC))

    typer.echo("configuration: ok")
    typer.echo("calendar: ok")
    typer.echo("data root: ok")
    state = "enabled" if settings.live_trading_enabled else "disabled"
    typer.echo(f"live trading: {state}")
