import logging
import sys
from collections.abc import Mapping

import structlog
from structlog.typing import EventDict, WrappedLogger

_SECRET_MARKERS = (
    "secret",
    "password",
    "token",
    "api_key",
    "access_key",
)


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: (
                "[REDACTED]"
                if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_secrets(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    return {
        key: (
            "[REDACTED]"
            if any(marker in key.lower() for marker in _SECRET_MARKERS)
            else _redact_value(value)
        )
        for key, value in event_dict.items()
    }


def configure_logging(level: str) -> None:
    numeric_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }[level]
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
