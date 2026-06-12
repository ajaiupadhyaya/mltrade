from typer.testing import CliRunner

from mltrade.cli import app

runner = CliRunner()


def test_doctor_reports_foundation_health(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "configuration: ok" in result.stdout
    assert "calendar: ok" in result.stdout
    assert "data root: ok" in result.stdout
    assert "live trading: disabled" in result.stdout
