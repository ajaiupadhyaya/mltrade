import json

import structlog

from mltrade.operations.logging import configure_logging


def test_logging_emits_json_without_secret_values(capsys) -> None:
    configure_logging("INFO")
    logger = structlog.get_logger("test")

    logger.info(
        "data_snapshot_validated",
        dataset="daily_prices",
        api_secret="paper-secret",
        database_url=(
            "postgresql+psycopg://mltrade:database-password@localhost/mltrade"
        ),
        context={"auth_token": "nested-token"},
    )

    output = json.loads(capsys.readouterr().out)
    raw_output = json.dumps(output)
    assert output["event"] == "data_snapshot_validated"
    assert output["dataset"] == "daily_prices"
    assert output["api_secret"] == "[REDACTED]"
    assert output["database_url"] == "[REDACTED]"
    assert output["context"]["auth_token"] == "[REDACTED]"
    assert output["level"] == "info"
    assert "paper-secret" not in raw_output
    assert "nested-token" not in raw_output
    assert "database-password" not in raw_output
