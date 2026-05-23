# Gateway production image.
#
# Phase 3.2: multi-stage build that produces a slim runtime image.
#
#   1. frontend     - pulls the prebuilt /dist from
#                     guitar-detect/frontend-assets:dev (Task 3.1).
#   2. python-build - installs the gateway package + deps into /install
#                     on a thin slim base. Build-only apt libs
#                     (-dev variants, build-essential) live ONLY here.
#   3. runtime      - copies /install + frontend /dist into a fresh
#                     python:3.11-slim with only the runtime apt libs
#                     aiortc needs.
#
# Size target (DESIGN.md §6.1): < 350 MB.
#
# Build context: services/gateway. The frontend image must be built first
# (see services/frontend/Dockerfile).

ARG FRONTEND_IMAGE=guitar-detect/frontend-assets:dev

# ---- frontend stage ----
FROM ${FRONTEND_IMAGE} AS frontend
# Nothing to do; the export stage of the frontend image is just /dist.

# ---- python-build stage ----
# Pin to a specific patch release; never the floating 3.11-slim tag.
FROM python:3.11.10-slim AS python-build

# Build-time deps for wheels that compile from source on this platform
# (av/aiortc on some arches). Kept out of the runtime stage so they don't
# bloat the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavutil-dev \
        libswscale-dev \
        libsrtp2-dev \
        libopus-dev \
        libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy project metadata + source, then install into /install. Using
# --prefix lets the runtime stage pick up dependencies AND the gateway
# package itself with a single COPY --from=... layer.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

# ---- runtime stage ----
FROM python:3.11.10-slim AS runtime

# aiortc runtime libs only (the *-dev variants are not needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libavformat59 \
        libavcodec59 \
        libavutil57 \
        libswscale6 \
        libsrtp2-1 \
        libopus0 \
        libvpx7 \
    && rm -rf /var/lib/apt/lists/*

# Deps + the gateway package itself (installed into site-packages) come
# in via /install -> /usr/local in one shot.
COPY --from=python-build /install /usr/local

WORKDIR /app

# Static SPA bundle from the frontend builder. main.py mounts this at
# /app/static (relative to WORKDIR) only if the directory exists.
COPY --from=frontend /dist /app/static

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Single worker by design (DESIGN.md §6.1): the WebRTC peer + Redis
# clients are process-local; horizontal scale-out happens at the
# orchestrator layer, not via uvicorn workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
