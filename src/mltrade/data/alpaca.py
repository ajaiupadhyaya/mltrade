"""Alpaca market-data adapter.

Fetches daily bars from the Alpaca v2 stocks/bars endpoint and converts them
to canonical :class:`~mltrade.data.bars.DailyBar` objects.

Module constants
----------------
- ``ALPACA_DATA_BASE_URL`` — default Alpaca market-data base URL
- ``ALPACA_BARS_PATH``     — path for the bars endpoint

Secrets policy
--------------
The API key and secret are accepted as :class:`~pydantic.SecretStr` and are
only unwrapped at the point of network I/O.  They are **never** logged,
included in exception messages, or exposed in repr output.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from mltrade.data.bars import DailyBar
from mltrade.domain.instruments import AssetType, InstrumentId

if TYPE_CHECKING:
    from mltrade.universe import Universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

ALPACA_DATA_BASE_URL: str = "https://data.alpaca.markets"
ALPACA_BARS_PATH: str = "/v2/stocks/bars"

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_REQUIRED_BAR_FIELDS = frozenset({"t", "o", "h", "l", "c", "v", "vw", "n"})


class AlpacaDataAdapter:
    """Fetches sanitized daily bars from the Alpaca v2 market-data API.

    Parameters
    ----------
    client:
        Synchronous :class:`httpx.Client` used for all requests.  The caller
        is responsible for lifecycle management (context manager recommended).
    api_key:
        Alpaca API key ID as a :class:`~pydantic.SecretStr`.
    api_secret:
        Alpaca API secret key as a :class:`~pydantic.SecretStr`.
    data_base_url:
        Base URL for the Alpaca market-data service.  Defaults to
        :data:`ALPACA_DATA_BASE_URL` (``https://data.alpaca.markets``).
    """

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: SecretStr,
        api_secret: SecretStr,
        data_base_url: str = ALPACA_DATA_BASE_URL,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._api_secret = api_secret
        self._data_base_url = data_base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return Alpaca auth headers without logging secrets."""
        return {
            "APCA-API-KEY-ID": self._api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": self._api_secret.get_secret_value(),
        }

    @staticmethod
    def _parse_bar(
        symbol: str,
        raw: object,
        ingested_at: datetime,
    ) -> DailyBar:
        """Convert a single raw bar dict to a :class:`DailyBar`.

        Raises
        ------
        ValueError
            If *raw* is not a mapping or any required field is missing.
        """
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected dict for bar entry of {symbol!r}, got {type(raw).__name__}"
            )
        missing = _REQUIRED_BAR_FIELDS - raw.keys()
        if missing:
            raise ValueError(
                f"Bar entry for {symbol!r} is missing required fields: "
                f"{sorted(missing)}"
            )

        # Parse session date from ISO timestamp (e.g. "2026-06-11T04:00:00Z")
        try:
            session: date = datetime.fromisoformat(
                raw["t"].replace("Z", "+00:00")
            ).date()
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse timestamp {raw['t']!r} for {symbol!r}: {exc}"
            ) from exc

        try:
            volume = int(raw["v"])
            trade_count = int(raw["n"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-integer volume/trade_count for {symbol!r}: {exc}"
            ) from exc

        try:
            open_ = Decimal(str(raw["o"]))
            high = Decimal(str(raw["h"]))
            low = Decimal(str(raw["l"]))
            close = Decimal(str(raw["c"]))
            vwap = Decimal(str(raw["vw"]))
        except Exception as exc:
            raise ValueError(
                f"Cannot parse price fields for {symbol!r}: {exc}"
            ) from exc

        instrument = InstrumentId(symbol=symbol, asset_type=AssetType.ETF)

        return DailyBar(
            instrument=instrument,
            session=session,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            trade_count=trade_count,
            source="alpaca",
            ingested_at=ingested_at,
        )

    # ------------------------------------------------------------------
    # Public interface (DailyBarSource protocol)
    # ------------------------------------------------------------------

    def fetch(
        self,
        universe: Universe,
        start: date,
        end: date,
        ingested_at: datetime,
    ) -> tuple[DailyBar, ...]:
        """Fetch daily bars for all instruments in *universe*.

        Parameters
        ----------
        universe:
            Target universe; only ``symbols`` is used.
        start:
            First session date (inclusive).
        end:
            Last session date (inclusive).
        ingested_at:
            UTC timestamp to stamp onto every returned bar.

        Returns
        -------
        tuple[DailyBar, ...]
            One :class:`DailyBar` per (symbol, session) pair returned by Alpaca,
            in the order Alpaca returns them.

        Raises
        ------
        ValueError
            If any bar entry is malformed (fail-closed).
        httpx.HTTPStatusError
            On 4xx/5xx responses.
        """
        symbols_csv = ",".join(universe.symbols)
        url = f"{self._data_base_url}{ALPACA_BARS_PATH}"
        params = {
            "symbols": symbols_csv,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": "iex",
        }

        logger.debug(
            "Fetching bars from Alpaca",
            extra={"url": url, "params": params},
        )

        response = self._client.get(
            url,
            params=params,
            headers=self._auth_headers(),
            timeout=httpx.Timeout(30.0),
        )
        response.raise_for_status()
        payload = response.json()

        bars_by_symbol: dict[str, list[object]] = payload.get("bars", {})

        results: list[DailyBar] = []
        for symbol, raw_bars in bars_by_symbol.items():
            if not isinstance(raw_bars, list):
                raise ValueError(
                    f"Expected list of bars for {symbol!r}, got "
                    f"{type(raw_bars).__name__}"
                )
            for raw in raw_bars:
                results.append(self._parse_bar(symbol, raw, ingested_at))

        logger.debug("Fetched %d bars from Alpaca", len(results))
        return tuple(results)
