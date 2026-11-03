# One base for both stages: the venv's python symlinks into this interpreter.
FROM docker.io/library/python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

FROM base AS build

COPY --from=uv /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY pinball.yaml ./
COPY src ./src
# Non-editable: pinball and its force-included pinball.yaml land in
# site-packages, so the venv does not depend on /app/src.
RUN uv sync --locked --no-dev --no-editable

FROM base

# Bytecode is compiled at build time; nothing is written at runtime, so the
# image runs with a read-only root filesystem.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=build /app/.venv /app/.venv

USER 1000:1000
EXPOSE 9201
CMD ["python", "-m", "pinball"]
