# Gateway production image.
#
# Three stages:
#   1. frontend     — pulls the prebuilt /dist from the frontend-assets
#                     image (services/frontend/Dockerfile).
#   2. python-build — installs deps + the gateway package into /install
#                     on a thin slim base. Build-only apt libs (-dev
#                     variants, build-essential) live ONLY here.
#   3. runtime      — copies /install + frontend /dist into a fresh
#                     python:3.11-slim with only the runtime apt libs
#                     aiortc needs.
#
# Layer ordering inside python-build:
#   a. apt build deps                          (rarely changes)
#   b. pyproject.toml only                     (changes per dep bump)
#   c. pip install runtime deps (no app code)  (~200 MB; cached
#                                               across source edits)
#   d. app/ source + pip install --no-deps .   (small, fast layer)
#
# This means a normal source edit invalidates only step (d), not the
# (large) dep install step (c).
#
# Size target (DESIGN.md §6.1): < 350 MB.
#
# Build context: services/gateway. The frontend image must be built
# first (see services/frontend/Dockerfile + the Makefile's
# build-images target).

ARG PYTHON_VERSION=3.11.10
ARG FRONTEND_IMAGE=guitar-detect/frontend-assets:dev

# ---------- frontend stage ----------
FROM ${FRONTEND_IMAGE} AS frontend
# Nothing to do; the export stage of the frontend image is just /dist.

# ---------- python-build stage ----------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS python-build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-time deps for wheels that compile from source on this platform
# (av/aiortc on some arches). Kept out of the runtime stage so they
# don't bloat the final image.
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

# --- layer: pyproject metadata only (caches dep install) ---
COPY pyproject.toml ./

# --- layer: install runtime deps WITHOUT app code ---
# Same tomllib-extraction trick as the worker image so a source edit
# doesn't invalidate the (large) dep install layer.
RUN python -c "import tomllib, sys; \
        data = tomllib.load(open('pyproject.toml','rb')); \
        deps = list(data['project']['dependencies']); \
        sys.stdout.write('\n'.join(deps))" > /tmp/deps.txt \
    && pip install --prefix=/install -r /tmp/deps.txt \
    && rm /tmp/deps.txt

# --- layer: app source + self-install (no deps; deps already there) ---
COPY app ./app
RUN pip install --prefix=/install --no-deps .

# ---------- runtime stage ----------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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

EXPOSE 8000

# Single worker by design (DESIGN.md §6.1): the WebRTC peer + Redis
# clients are process-local; horizontal scale-out happens at the
# orchestrator layer, not via uvicorn workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
