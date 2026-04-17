# syntax=docker/dockerfile:1.6
# ──────────────────────────────────────────────────────────────────────────────
# AI Labor Market Impact Observatory — Dashboard image.
#
# Build:
#   docker build -t ai-labor-impact .
# Run:
#   docker run --rm -p 8501:8501 ai-labor-impact
#   # → open http://localhost:8501
#
# Image stays lean (~600 MB) by installing the runtime requirements only.
# For the Bayesian / MLflow notebooks, run them outside Docker or extend
# this Dockerfile with `requirements-dev.txt`.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# OS deps: build-essential needed for some scipy/numpy wheel scenarios.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached when only source changes).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code.
COPY src/   src/
COPY app/   app/
COPY data/  data/
COPY pyproject.toml README.md ./

# Streamlit health endpoint for orchestrators / `docker compose` healthchecks.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py"]
