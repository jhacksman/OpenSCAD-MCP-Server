FROM python:3.13-slim-bookworm AS builder

# Match the lockfile tooling used for the native installation.
RUN python -m pip install --no-cache-dir uv==0.6.3
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/openscad_mcp ./src/openscad_mcp
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
RUN uv sync --locked --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    openscad xvfb xauth libgl1-mesa-dri fonts-liberation tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data /tmp/runtime-app \
    && chown app:app /data /tmp/runtime-app \
    && chmod 700 /tmp/runtime-app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    OPENSCAD_OUTPUT_DIR=/data \
    LIBGL_ALWAYS_SOFTWARE=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-app \
    PYTHONUNBUFFERED=1
USER app
WORKDIR /data
# MCP uses stdin/stdout. Xvfb supplies a headless display; no GPU is required.
# tini forwards termination to the entire process group, including renderers.
ENTRYPOINT ["tini", "-g", "--", "xvfb-run", "-a", "-s", "-screen 0 1024x768x24 -nolisten tcp", "openscad-mcp"]
