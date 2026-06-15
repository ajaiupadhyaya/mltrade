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


def test_experiment_paths_default_under_data_root(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    assert settings.experiment_root == tmp_path / "experiments"
    assert settings.mlflow_tracking_root == tmp_path / "experiments" / "mlflow"
    assert (
        settings.optuna_storage_path
        == tmp_path / "experiments" / "optuna" / "studies.db"
    )


@pytest.mark.parametrize("experiment_root", ["", "   "])
def test_blank_experiment_root_uses_data_root_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_root: str,
) -> None:
    monkeypatch.setenv("MLTRADE_EXPERIMENT_ROOT", experiment_root)

    settings = Settings(data_root=tmp_path)

    assert settings.experiment_root == tmp_path / "experiments"


def test_explicit_experiment_root_is_absolute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        data_root=tmp_path,
        experiment_root=Path("nested") / ".." / "research",
    )
    assert settings.experiment_root == (tmp_path / "research").resolve()


def test_explicit_experiment_root_expands_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    settings = Settings(data_root=tmp_path, experiment_root=Path("~/research"))

    assert settings.experiment_root == (home / "research").resolve()


def test_model_copy_recomputes_derived_experiment_root(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "original")

    copied = settings.model_copy(update={"data_root": tmp_path / "copied"})

    assert copied.data_root == (tmp_path / "copied").resolve()
    assert copied.experiment_root == (tmp_path / "copied" / "experiments").resolve()


def test_model_copy_preserves_explicit_experiment_root(tmp_path: Path) -> None:
    experiment_root = tmp_path / "custom"
    settings = Settings(
        data_root=tmp_path / "original",
        experiment_root=experiment_root,
    )

    copied = settings.model_copy(update={"data_root": tmp_path / "copied"})

    assert copied.data_root == (tmp_path / "copied").resolve()
    assert copied.experiment_root == experiment_root.resolve()


def test_model_copy_honors_explicit_none_experiment_root(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path / "original",
        experiment_root=tmp_path / "custom",
    )

    copied = settings.model_copy(
        update={
            "data_root": tmp_path / "copied",
            "experiment_root": None,
        }
    )

    assert copied.experiment_root == (tmp_path / "copied" / "experiments").resolve()


def test_model_copy_normalizes_explicit_experiment_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_root=tmp_path / "original")

    copied = settings.model_copy(
        update={
            "data_root": tmp_path / "copied",
            "experiment_root": Path("nested") / ".." / "research",
        }
    )

    assert copied.data_root == (tmp_path / "copied").resolve()
    assert copied.experiment_root == (tmp_path / "research").resolve()


def test_model_copy_revalidates_safety_constraints(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    with pytest.raises(ValidationError, match="not available"):
        settings.model_copy(update={"live_trading_enabled": True})


def test_model_copy_preserves_derived_path_for_risk_updates(tmp_path: Path) -> None:
    database_url = "postgresql+psycopg://user:db-password@localhost/mltrade"
    settings = Settings(data_root=tmp_path, database_url=database_url)

    copied = settings.model_copy(
        update={"maximum_position_weight": Decimal("0.20")}
    )

    assert copied.maximum_position_weight == Decimal("0.20")
    assert copied.experiment_root == tmp_path / "experiments"
    assert copied.database_url == database_url
    assert copied.model_dump()["database_url"] == "[REDACTED]"


def test_settings_tests_ignore_local_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MLTRADE_ENVIRONMENT=live\nMLTRADE_REFERENCE_EQUITY=1\n"
    )

    settings = Settings()

    assert settings.environment is Environment.LOCAL
    assert settings.reference_equity == Decimal("1000000")


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
