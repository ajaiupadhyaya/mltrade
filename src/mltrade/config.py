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
    database_url: str = Field(
        default="sqlite+pysqlite:///data/operations.db",
        repr=False,
    )
    alpaca_api_key: SecretStr | None = None
    alpaca_api_secret: SecretStr | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
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
    def reject_live_trading(self) -> "Settings":
        if self.live_trading_enabled:
            raise ValueError("live trading is not available in this release")
        return self
