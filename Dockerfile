FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HOME=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (without the project) for better layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the actual project and install it.
COPY src ./src
COPY app ./app
COPY data ./data
COPY models ./models
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "--no-sync", "streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
