from datetime import UTC, date, datetime

from mltrade.calendar import XNYSCalendar


def test_holiday_is_not_a_session() -> None:
    calendar = XNYSCalendar()

    assert calendar.is_session(date(2024, 7, 4)) is False
    assert calendar.is_session(date(2024, 7, 5)) is True


def test_last_completed_session_before_market_close() -> None:
    calendar = XNYSCalendar()
    now = datetime(2024, 7, 5, 17, 0, tzinfo=UTC)

    assert calendar.last_completed_session(now) == date(2024, 7, 3)


def test_last_completed_session_after_market_close() -> None:
    calendar = XNYSCalendar()
    now = datetime(2024, 7, 5, 21, 0, tzinfo=UTC)

    assert calendar.last_completed_session(now) == date(2024, 7, 5)
