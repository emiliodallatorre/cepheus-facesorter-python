FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=run_directives/default.yml

COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
RUN uv sync --frozen --no-dev

COPY src ./src
COPY run_directives ./run_directives
COPY prefect.yaml ./prefect.yaml

CMD ["sh", "-c", "uv run python src/pipeline.py --config \"$CONFIG_PATH\""]
