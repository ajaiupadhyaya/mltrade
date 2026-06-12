from datetime import UTC, datetime


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    if value.tzinfo is UTC:
        return value
    return value.astimezone(UTC)
