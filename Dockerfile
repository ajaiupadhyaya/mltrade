FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Data root used by all mltrade commands inside the container.
# The demo writes Parquet snapshots and the SQLite operations DB here.
ENV MLTRADE_DATA_ROOT=/app/data
ENV MLTRADE_DATABASE_URL=sqlite+pysqlite:////app/data/operations.db

WORKDIR /app

# Install uv and sync runtime dependencies (no dev extras).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# Create a non-root user and a writable data directory owned by that user.
RUN groupadd --system mltrade \
    && useradd --system --gid mltrade --no-create-home mltrade \
    && mkdir -p /app/data \
    && chown -R mltrade:mltrade /app/data

USER mltrade

ENTRYPOINT ["uv", "run", "mltrade"]
CMD ["demo", "run"]
