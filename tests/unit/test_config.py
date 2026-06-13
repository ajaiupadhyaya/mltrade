import json
from decimal import Decimal
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


def test_mvp_settings_have_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.reference_equity == Decimal("1000000")
    assert settings.maximum_position_weight == Decimal("0.25")
    assert settings.minimum_cash_weight == Decimal("0.05")
    assert settings.target_annual_volatility == Decimal("0.15")
    assert settings.maximum_order_weight == Decimal("0.10")
    assert settings.maximum_rebalance_weight == Decimal("0.50")
    assert settings.minimum_order_notional == Decimal("500")
    assert settings.transaction_cost_bps == Decimal("5")


def test_paper_environment_requires_paper_url() -> None:
    with pytest.raises(ValidationError, match=r"paper-api\.alpaca\.markets"):
        Settings(
            environment=Environment.PAPER,
            alpaca_base_url="https://api.alpaca.markets",
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "reference_equity",
        "maximum_position_weight",
        "minimum_cash_weight",
        "target_annual_volatility",
        "maximum_order_weight",
        "maximum_rebalance_weight",
        "minimum_order_notional",
        "transaction_cost_bps",
    ],
)
def test_mvp_decimal_settings_must_be_strictly_positive(field_name: str) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(**{field_name: Decimal("0")})


def test_maximum_position_weight_reserves_minimum_cash() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_position_weight cannot exceed 1 - minimum_cash_weight",
    ):
        Settings(
            maximum_position_weight=Decimal("0.96"),
            minimum_cash_weight=Decimal("0.05"),
        )


def test_paper_environment_accepts_trailing_slash_on_paper_url() -> None:
    settings = Settings(
        environment=Environment.PAPER,
        alpaca_base_url="https://paper-api.alpaca.markets/",
    )

    assert settings.alpaca_base_url == "https://paper-api.alpaca.markets/"


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
    settings = Settings(database_url=database_url)

    rendered = repr(settings)
    serialized = settings.model_dump()
    serialized_json = settings.model_dump_json()

    assert settings.database_url == database_url
    assert database_url not in rendered
    assert "db-password" not in str(serialized)
    assert "db-password" not in serialized_json
    assert serialized["database_url"] == "[REDACTED]"
    assert json.loads(serialized_json)["database_url"] == "[REDACTED]"


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
