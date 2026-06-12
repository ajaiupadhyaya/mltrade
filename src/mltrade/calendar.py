from datetime import UTC, date, datetime, timedelta
from functools import cached_property
from typing import Any

import exchange_calendars as xcals  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from mltrade.domain.time import require_utc


class XNYSCalendar:
    @cached_property
    def _calendar(self) -> Any:
        return xcals.get_calendar("XNYS")

    def is_session(self, session_date: date) -> bool:
        return bool(self._calendar.is_session(pd.Timestamp(session_date)))

    def last_completed_session(self, now: datetime) -> date:
        utc_now = require_utc(now)
        candidate = utc_now.date()

        while not self.is_session(candidate):
            candidate -= timedelta(days=1)

        session = pd.Timestamp(candidate)
        close = self._calendar.session_close(session).to_pydatetime()
        if close.tzinfo is None:
            close = close.replace(tzinfo=UTC)

        if utc_now >= close.astimezone(UTC):
            return candidate

        previous = self._calendar.previous_session(session)
        previous_date = previous.date()
        if not isinstance(previous_date, date):
            raise TypeError("calendar returned an invalid session date")
        return previous_date
