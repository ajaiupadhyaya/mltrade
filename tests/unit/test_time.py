from datetime import UTC, datetime, timedelta, timezone

import pytest

from mltrade.domain.time import require_utc


def test_require_utc_accepts_utc_datetime() -> None:
    value = datetime(2026, 6, 12, 20, 0, tzinfo=UTC)

    assert require_utc(value) is value


def test_require_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        require_utc(datetime(2026, 6, 12, 20, 0))


def test_require_utc_converts_offset_datetime() -> None:
    eastern = timezone(-timedelta(hours=4))
    value = datetime(2026, 6, 12, 16, 0, tzinfo=eastern)

    assert require_utc(value) == datetime(2026, 6, 12, 20, 0, tzinfo=UTC)
