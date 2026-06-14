from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import (
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    environment: Environment = Environment.LOCAL
    data_root: Path = Path("data")
    experiment_root: Path | None = None
    database_url: str = Field(
        default="sqlite+pysqlite:///data/operations.db",
        repr=False,
    )
    alpaca_api_key: SecretStr | None = None
    alpaca_api_secret: SecretStr | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    reference_equity: Decimal = Field(default=Decimal("1000000"), gt=0)
    maximum_position_weight: Decimal = Field(default=Decimal("0.25"), gt=0)
    minimum_cash_weight: Decimal = Field(default=Decimal("0.05"), gt=0)
    target_annual_volatility: Decimal = Field(default=Decimal("0.15"), gt=0)
    maximum_order_weight: Decimal = Field(default=Decimal("0.10"), gt=0)
    maximum_rebalance_weight: Decimal = Field(default=Decimal("0.50"), gt=0)
    minimum_order_notional: Decimal = Field(default=Decimal("500"), gt=0)
    transaction_cost_bps: Decimal = Field(default=Decimal("5"), gt=0)
    live_trading_enabled: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="MLTRADE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
    )

    @field_validator("data_root")
    @classmethod
    def make_data_root_absolute(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @field_serializer("database_url")
    def redact_database_url(self, value: str) -> str:
        return "[REDACTED]"

    @model_validator(mode="after")
    def validate_safety_constraints(self) -> "Settings":
        if self.live_trading_enabled:
            raise ValueError("live trading is not available in this release")
        if (
            self.environment is Environment.PAPER
            and self.alpaca_base_url.rstrip("/")
            != "https://paper-api.alpaca.markets"
        ):
            raise ValueError(
                "paper environment requires "
                "https://paper-api.alpaca.markets as alpaca_base_url"
            )
        if self.maximum_position_weight > Decimal("1") - self.minimum_cash_weight:
            raise ValueError(
                "maximum_position_weight cannot exceed 1 - minimum_cash_weight"
            )
        if self.experiment_root is None:
            self.experiment_root = self.data_root / "experiments"
        else:
            self.experiment_root = self.experiment_root.expanduser().resolve()
        return self

    @property
    def mlflow_tracking_root(self) -> Path:
        assert self.experiment_root is not None
        return self.experiment_root / "mlflow"

    @property
    def optuna_storage_path(self) -> Path:
        assert self.experiment_root is not None
        return self.experiment_root / "optuna" / "studies.db"
