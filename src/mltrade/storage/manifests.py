import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecatedSince20,
    field_validator,
)

from mltrade.domain.time import require_utc


def require_safe_path_segment(value: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("value must be a safe path segment")
    return value


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    snapshot_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    created_at: datetime
    source: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    row_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_files: tuple[str, ...] = ()

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("DatasetManifest cannot be updated")
        return super().model_copy(update=update, deep=deep)

    @override
    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            warnings.warn(
                "The `copy` method is deprecated; use `model_copy` instead.",
                category=PydanticDeprecatedSince20,
                stacklevel=2,
            )
            raise TypeError("DatasetManifest cannot be updated")
        return super().copy(
            include=include,
            exclude=exclude,
            update=update,
            deep=deep,
        )

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return require_safe_path_segment(value)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("data_files")
    @classmethod
    def require_relative_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            posix_path = PurePosixPath(value)
            windows_path = PureWindowsPath(value)
            if (
                not value
                or value == "."
                or posix_path.is_absolute()
                or windows_path.is_absolute()
                or windows_path.drive
                or ".." in posix_path.parts
                or ".." in windows_path.parts
            ):
                raise ValueError("data_files must contain safe relative paths")
        return values
