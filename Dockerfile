FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Data root used by all mltrade commands inside the container.
# The demo writes Parquet snapshots and the SQLite operations DB here.
ENV MLTRADE_DATA_ROOT=/app/data
ENV MLTRADE_DATABASE_URL=sqlite+pysqlite:////app/data/operations.db

# uv: don't mutate the lockfile or re-sync at runtime, and keep the cache
# inside /app (owned by the non-root user) so `uv run` works without a
# writable HOME.
ENV UV_FROZEN=1
ENV UV_NO_SYNC=1
ENV UV_CACHE_DIR=/app/.uv-cache

WORKDIR /app

# Install uv and sync runtime dependencies (no dev extras).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY experiments ./experiments

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# Create a non-root user and make the app tree (venv, data, uv cache) writable
# by it, so `uv run` and the demo's snapshot/SQLite writes all succeed.
RUN groupadd --system mltrade \
    && useradd --system --gid mltrade --no-create-home mltrade \
    && mkdir -p /app/data /app/.uv-cache \
    && chown -R mltrade:mltrade /app

USER mltrade

ENTRYPOINT ["uv", "run", "mltrade"]
CMD ["demo", "run"]
