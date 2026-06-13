"""Immutable Parquet snapshot publication for daily bar data.

``DailyBarPublisher`` serialises a tuple of
:class:`~mltrade.data.bars.DailyBar` objects to a Parquet file using a
fixed explicit PyArrow schema, computes a SHA-256 digest of the exact
on-disk bytes, saves a
:class:`~mltrade.storage.manifests.DatasetManifest`,
and returns a :class:`PublishedSnapshot`.

``load_verified`` recomputes the SHA-256, verifies row count, and reconstructs
the original bars from the Parquet bytes with exact Decimal equality.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from mltrade.data.bars import DailyBar
from mltrade.data.quality import DataQualityReport
from mltrade.domain.instruments import AssetType, InstrumentId
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.universe import MVP_UNIVERSE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET: Final[str] = "daily_bars"
PARQUET_FILENAME: Final[str] = "daily-bars.parquet"
SCHEMA_VERSION: Final[int] = 1
QUALITY_VERSION: Final[str] = "daily-bar-quality-v1"
SOURCE: Final[str] = "mltrade-daily-bar-publisher"

# ---------------------------------------------------------------------------
# Fixed explicit PyArrow schema
#
# Columns:
#   symbol          utf8
#   asset_type      utf8
#   country         utf8
#   session         date32               (calendar date, no timezone)
#   open            utf8                 (Decimal stored as exact string)
#   high            utf8
#   low             utf8
#   close           utf8
#   vwap            utf8
#   volume          int64
#   trade_count     int64
#   source          utf8
#   ingested_at     timestamp[us, tz=UTC]
#
# Rationale for storing Decimal fields as UTF-8 strings: Parquet decimal128
# requires a fixed precision+scale; our fixture bars use Decimal("xxx.yyyy")
# with up to 4 decimal places, but precision and scale vary per value.  Storing
# the canonical ``str(Decimal)`` round-trips without any rounding loss and
# lets us reconstruct the *exact* Decimal object via ``Decimal(str_value)``.
# ---------------------------------------------------------------------------

_ARROW_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("symbol", pa.utf8(), nullable=False),
        pa.field("asset_type", pa.utf8(), nullable=False),
        pa.field("country", pa.utf8(), nullable=False),
        pa.field("session", pa.date32(), nullable=False),
        pa.field("open", pa.utf8(), nullable=False),
        pa.field("high", pa.utf8(), nullable=False),
        pa.field("low", pa.utf8(), nullable=False),
        pa.field("close", pa.utf8(), nullable=False),
        pa.field("vwap", pa.utf8(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("trade_count", pa.int64(), nullable=False),
        pa.field("source", pa.utf8(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublishedSnapshot:
    """Immutable result returned by :meth:`DailyBarPublisher.publish`."""

    manifest: DatasetManifest
    data_files: tuple[Path, ...]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _bars_to_arrow_table(bars: tuple[DailyBar, ...]) -> pa.Table:
    """Convert bars to a PyArrow Table with the fixed schema."""
    symbols: list[str] = []
    asset_types: list[str] = []
    countries: list[str] = []
    sessions: list[date] = []
    opens: list[str] = []
    highs: list[str] = []
    lows: list[str] = []
    closes: list[str] = []
    vwaps: list[str] = []
    volumes: list[int] = []
    trade_counts: list[int] = []
    sources: list[str] = []
    ingested_ats: list[datetime] = []

    for bar in bars:
        symbols.append(bar.instrument.symbol)
        asset_types.append(bar.instrument.asset_type.value)
        countries.append(bar.instrument.country)
        sessions.append(bar.session)
        opens.append(str(bar.open))
        highs.append(str(bar.high))
        lows.append(str(bar.low))
        closes.append(str(bar.close))
        vwaps.append(str(bar.vwap))
        volumes.append(bar.volume)
        trade_counts.append(bar.trade_count)
        sources.append(bar.source)
        ingested_ats.append(bar.ingested_at)

    return pa.table(
        {
            "symbol": pa.array(symbols, type=pa.utf8()),
            "asset_type": pa.array(asset_types, type=pa.utf8()),
            "country": pa.array(countries, type=pa.utf8()),
            "session": pa.array(sessions, type=pa.date32()),
            "open": pa.array(opens, type=pa.utf8()),
            "high": pa.array(highs, type=pa.utf8()),
            "low": pa.array(lows, type=pa.utf8()),
            "close": pa.array(closes, type=pa.utf8()),
            "vwap": pa.array(vwaps, type=pa.utf8()),
            "volume": pa.array(volumes, type=pa.int64()),
            "trade_count": pa.array(trade_counts, type=pa.int64()),
            "source": pa.array(sources, type=pa.utf8()),
            "ingested_at": pa.array(ingested_ats, type=pa.timestamp("us", tz="UTC")),
        },
        schema=_ARROW_SCHEMA,
    )


def _table_to_parquet_bytes(table: pa.Table) -> bytes:
    """Serialise a PyArrow Table to Parquet bytes in memory."""
    buf = io.BytesIO()
    pq.write_table(table, buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arrow_table_to_bars(table: pa.Table) -> tuple[DailyBar, ...]:
    """Reconstruct DailyBar objects from a PyArrow Table."""
    # Convert to Python dicts column-by-column for speed
    col = table.to_pydict()
    bars: list[DailyBar] = []
    n = len(table)
    for i in range(n):
        instrument = InstrumentId(
            symbol=col["symbol"][i],
            asset_type=AssetType(col["asset_type"][i]),
            country=col["country"][i],
        )
        bar = DailyBar(
            instrument=instrument,
            session=col["session"][i],
            open=Decimal(col["open"][i]),
            high=Decimal(col["high"][i]),
            low=Decimal(col["low"][i]),
            close=Decimal(col["close"][i]),
            vwap=Decimal(col["vwap"][i]),
            volume=int(col["volume"][i]),
            trade_count=int(col["trade_count"][i]),
            source=col["source"][i],
            ingested_at=col["ingested_at"][i],
        )
        bars.append(bar)
    return tuple(bars)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class DailyBarPublisher:
    """Publish and verify immutable daily bar Parquet snapshots.

    Parameters
    ----------
    store:
        The :class:`~mltrade.storage.snapshots.SnapshotStore` that owns the
        on-disk snapshot directories.
    """

    def __init__(self, store: SnapshotStore) -> None:
        self._store = store

    def publish(
        self,
        *,
        bars: tuple[DailyBar, ...],
        quality: DataQualityReport,
        snapshot_id: str,
        created_at: datetime,
    ) -> PublishedSnapshot:
        """Serialise *bars* to Parquet and save a verified manifest.

        Parameters
        ----------
        bars:
            The daily bar data to publish.  Must be sorted by (session, symbol).
        quality:
            A passing (non-blocked) :class:`~mltrade.data.quality.DataQualityReport`
            for *bars*.  If the report is blocked, ``ValueError`` is raised
            before any data is written.
        snapshot_id:
            Unique identifier for this snapshot (e.g. ``"fixture-20260612"``).
        created_at:
            UTC timestamp to record as the snapshot creation time.

        Returns
        -------
        PublishedSnapshot
            Immutable value object with ``.manifest`` and ``.data_files``.

        Raises
        ------
        ValueError
            If *quality* is blocked.
        FileExistsError
            If the snapshot already exists on disk.
        """
        if quality.blocked:
            raise ValueError(
                f"quality report is blocked — refusing to publish snapshot "
                f"'{snapshot_id}'; issues: {quality.issues}"
            )

        # Sort bars by (session, symbol) — must already be sorted, but we sort
        # defensively to guarantee deterministic byte output.
        sorted_bars = tuple(
            sorted(bars, key=lambda b: (b.session, b.instrument.symbol))
        )

        # Serialise to Parquet bytes in memory using the fixed explicit schema
        table = _bars_to_arrow_table(sorted_bars)
        parquet_bytes = _table_to_parquet_bytes(table)

        # Write to disk via SnapshotStore (hardened temp+link pattern)
        data_file_path = self._store.save_data_file(
            DATASET,
            snapshot_id,
            PARQUET_FILENAME,
            parquet_bytes,
        )

        # Re-read the exact on-disk bytes to compute the authoritative SHA-256
        on_disk_bytes = data_file_path.read_bytes()
        content_sha256 = _sha256_of_bytes(on_disk_bytes)

        # Build and persist the manifest
        manifest = DatasetManifest(
            dataset=DATASET,
            snapshot_id=snapshot_id,
            created_at=created_at,
            source=SOURCE,
            schema_version=SCHEMA_VERSION,
            row_count=len(sorted_bars),
            content_sha256=content_sha256,
            data_files=(PARQUET_FILENAME,),
            metadata={
                "universe_version": MVP_UNIVERSE.version,
                "schema_version": str(SCHEMA_VERSION),
                "quality_version": QUALITY_VERSION,
            },
        )
        self._store.save_manifest(manifest)

        return PublishedSnapshot(
            manifest=manifest,
            data_files=(data_file_path,),
        )

    def load_verified(self, manifest: DatasetManifest) -> tuple[DailyBar, ...]:
        """Load and verify the bars described by *manifest*.

        Re-reads the Parquet file bytes from disk, verifies the SHA-256
        matches ``manifest.content_sha256``, checks the row count, and
        reconstructs the :class:`~mltrade.data.bars.DailyBar` objects.

        Parameters
        ----------
        manifest:
            The manifest that was returned from :meth:`publish`.

        Returns
        -------
        tuple[DailyBar, ...]
            The bar data in (session, symbol) order.

        Raises
        ------
        ValueError
            If the file hash does not match (``"content hash"`` in message),
            or the row count does not match.
        """
        snapshot_dir = self._store.snapshot_dir(manifest.dataset, manifest.snapshot_id)

        # There must be exactly one data file for daily bars
        if len(manifest.data_files) != 1:
            raise ValueError(
                f"expected exactly 1 data file in manifest, got "
                f"{len(manifest.data_files)}"
            )

        parquet_path = snapshot_dir / manifest.data_files[0]
        on_disk_bytes = parquet_path.read_bytes()

        # Verify content hash
        actual_sha256 = _sha256_of_bytes(on_disk_bytes)
        if actual_sha256 != manifest.content_sha256:
            raise ValueError(
                f"content hash mismatch for '{parquet_path}': "
                f"expected {manifest.content_sha256!r}, "
                f"got {actual_sha256!r}"
            )

        # Deserialise
        buf = io.BytesIO(on_disk_bytes)
        table = pq.read_table(buf, schema=_ARROW_SCHEMA)  # type: ignore[no-untyped-call]

        # Verify row count
        if len(table) != manifest.row_count:
            raise ValueError(
                f"row count mismatch: manifest says {manifest.row_count}, "
                f"file has {len(table)}"
            )

        return _arrow_table_to_bars(table)
