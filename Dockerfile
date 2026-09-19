FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/camptodata/forest-cover-app" \n      org.opencontainers.image.description="Streamlit app for forest cover type classification" \n      org.opencontainers.image.licenses="MIT"

# uv is copied from its official image, pinned for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# libgomp1 is required by XGBoost; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user that owns /app from the start, so no recursive chown
# (which would duplicate the whole virtual environment layer) is needed later.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir /app \
    && chown appuser:appuser /app

WORKDIR /app
USER appuser
ENV HOME=/home/appuser

# Install dependencies first (without the project) for better layer caching.
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the actual project and install it.
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

# Use the project's virtual environment directly: nothing is resolved or
# downloaded when the container starts.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
