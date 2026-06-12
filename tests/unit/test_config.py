from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from mltrade.config import Environment, Settings


def test_settings_use_safe_local_defaults(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.environment is Environment.LOCAL
    assert settings.live_trading_enabled is False
    assert settings.data_root == tmp_path.absolute()
    assert settings.database_url == "sqlite+pysqlite:///data/operations.db"


def test_secret_values_are_not_revealed(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        alpaca_api_key=SecretStr("paper-key"),
        alpaca_api_secret=SecretStr("paper-secret"),
    )

    rendered = repr(settings)
    assert "paper-key" not in rendered
    assert "paper-secret" not in rendered


def test_database_credentials_are_not_revealed() -> None:
    database_url = "postgresql+psycopg://user:db-password@localhost/mltrade"

    rendered = repr(Settings(database_url=database_url))

    assert database_url not in rendered


def test_environment_variables_use_mltrade_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLTRADE_ENVIRONMENT", "paper")
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))

    settings = Settings()

    assert settings.environment is Environment.PAPER
    assert settings.data_root == tmp_path.absolute()


def test_log_level_is_normalized() -> None:
    settings = Settings(log_level="warning")

    assert settings.log_level == "WARNING"


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings(log_level="TRACE")


def test_live_trading_cannot_be_enabled_in_foundation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not available"):
        Settings(data_root=tmp_path, live_trading_enabled=True)
