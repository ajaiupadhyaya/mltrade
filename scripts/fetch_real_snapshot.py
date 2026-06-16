"""One-off real-market-data snapshot fetcher (provenance/ops tool).

NOT part of the ``mltrade`` package and NOT a runtime dependency.  Run it once
to freeze a point-in-time panel of real, split/dividend-adjusted daily bars into
the repository.  The platform runtime then reads only the frozen Parquet file
(:class:`mltrade.data.snapshot.SnapshotBarSource`), so every downstream result
stays fully deterministic and offline — exactly how a desk freezes a research
dataset.

Source:  Yahoo Finance via ``yfinance`` (auto-adjusted OHLC).
Run:     uv run --with yfinance python scripts/fetch_real_snapshot.py

The output is a rectangular panel over the set of sessions for which *every*
symbol has a bar (an inner join across symbols), guaranteeing the completeness
the data-quality gate and walk-forward engine expect.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Cross-asset risk-premia universe (all liquid since 2007); SPY doubles as the
# benchmark.  Order is fixed for reproducibility.
SYMBOLS: tuple[str, ...] = (
    "SPY",  # US large cap (and benchmark)
    "QQQ",  # US large-cap growth / tech
    "IWM",  # US small cap
    "EFA",  # Developed ex-US equity
    "EEM",  # Emerging-market equity
    "TLT",  # Long US Treasuries
    "IEF",  # Intermediate US Treasuries
    "GLD",  # Gold
    "DBC",  # Broad commodities
    "VNQ",  # US REITs
)

START = "2007-01-03"
AS_OF = "2026-06-13"  # frozen point-in-time as-of date (inclusive end)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "snapshots" / "real"
PANEL_PATH = OUT_DIR / f"daily_bars_{AS_OF}.parquet"
MANIFEST_PATH = OUT_DIR / f"daily_bars_{AS_OF}.manifest.json"


def _fetch_symbol(symbol: str) -> pd.DataFrame:
    """Download one symbol's auto-adjusted daily OHLCV as a tidy frame."""
    raw = yf.download(
        symbol,
        start=START,
        end=AS_OF,
        interval="1d",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise SystemExit(f"no data returned for {symbol}")
    # yfinance returns a column MultiIndex (field, ticker) for single symbols too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.columns = ["open", "high", "low", "close", "volume"]
    frame.index = pd.to_datetime(frame.index).date  # type: ignore[assignment]
    frame.index.name = "session"
    # Drop any session with a non-finite or non-positive price.
    frame = frame.dropna()
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_symbol: dict[str, pd.DataFrame] = {s: _fetch_symbol(s) for s in SYMBOLS}

    # Inner-join sessions: keep only dates present for EVERY symbol so the panel
    # is a complete rectangle (no holes for the quality gate / engine).
    common: set[date] | None = None
    for frame in per_symbol.values():
        idx = set(frame.index)
        common = idx if common is None else (common & idx)
    assert common is not None
    sessions = sorted(common)
    if not sessions:
        raise SystemExit("no overlapping sessions across the universe")

    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        frame = per_symbol[symbol].loc[sessions]
        for session, row in frame.iterrows():
            o = round(float(row["open"]), 4)
            h = round(float(row["high"]), 4)
            low = round(float(row["low"]), 4)
            c = round(float(row["close"]), 4)
            # Guarantee OHLC invariants survive rounding (nudge by <= 1e-4).
            h = max(h, o, low, c)
            low = min(low, o, h, c)
            vwap = round((h + low + c) / 3.0, 4)
            rows.append(
                {
                    "session": session,
                    "symbol": symbol,
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": int(row["volume"]),
                    "vwap": vwap,
                    "trade_count": 0,
                    "source": "yfinance-adjusted",
                }
            )

    panel = pd.DataFrame(rows).sort_values(["session", "symbol"]).reset_index(drop=True)
    panel.to_parquet(PANEL_PATH, index=False)

    content_sha256 = hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest()
    manifest = {
        "dataset": "real_daily_bars",
        "source": "yfinance (Yahoo Finance), auto_adjust=True",
        "adjustment": "split+dividend (total-return adjusted)",
        "as_of": AS_OF,
        "fetched_at": datetime.now(UTC).isoformat(),
        "symbols": list(SYMBOLS),
        "benchmark": "SPY",
        "start_session": sessions[0].isoformat(),
        "end_session": sessions[-1].isoformat(),
        "session_count": len(sessions),
        "row_count": len(panel),
        "rows_per_symbol": {s: len(per_symbol[s].loc[sessions]) for s in SYMBOLS},
        "content_sha256": content_sha256,
        "panel_file": PANEL_PATH.name,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {PANEL_PATH.relative_to(REPO_ROOT)}  ({len(panel):,} rows)")
    print(f"  sessions: {sessions[0]} → {sessions[-1]}  ({len(sessions):,})")
    print(f"  sha256:   {content_sha256}")


if __name__ == "__main__":
    main()
