FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    SIGNATURE_DEVICE=cpu \
    SIGNATURE_TORCH_THREADS=2
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN groupadd --gid 10001 api && useradd --uid 10001 --gid api --no-create-home api
COPY --chown=api:api app ./app
COPY --chown=api:api vendor ./vendor
COPY --chown=api:api signature_yolox/api_model ./signature_yolox/api_model

USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
