FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv
# HF_HUB_DISABLE_SYMLINKS: ONNX Runtime rejects external weight files reached through HF cache symlinks.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" HF_HOME=/models HF_HUB_DISABLE_SYMLINKS=1 PYTHONUNBUFFERED=1

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --extra cuda
COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
