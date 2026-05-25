# Inference worker production image.
#
# Layer ordering goal: any single source-code edit invalidates only the
# small "app source + package self-install" tail, not the heavy
# dependency install or the multi-hundred-MB model bakes. Concretely
# the layers go (cheapest invalidation last):
#
#   1. apt deps                                  (rarely changes)
#   2. pyproject.toml only                       (changes per dep bump)
#   3. pip install runtime deps (no app code)    (~3 GB layer; cached
#                                                 across source edits)
#   4. YOLO + MobileCLIP bake (download_models.py)   (cached unless
#                                                 the script changes)
#   5. SigLIP-2 pre-warm into HF cache           (cached unless model
#                                                 id env changes)
#   6. app/ source + pip install --no-deps .     (this is the small
#                                                 fast layer that
#                                                 every source edit
#                                                 ends up busting)
#
# Final image carries deps + baked models + the warmed HF cache, but
# leaves behind apt build chain and pyc droppings in the builder.
#
# Deviation from DESIGN.md §6.2 (which names openvino/ubuntu22_runtime
# as the base): we use python:3.11-slim and install openvino via pip.
# The OpenVINO base ships Python 3.10, which conflicts with the
# worker's `requires-python = >=3.11,<3.12` pin.
#
# Build context: services/inference-worker.

ARG PYTHON_VERSION=3.11.10
ARG SIGLIP_MODEL_ID=google/siglip2-base-patch16-256
ARG HF_HOME_DIR=/opt/hf-cache

# ---------- builder stage ----------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder

ARG SIGLIP_MODEL_ID
ARG HF_HOME_DIR
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=${HF_HOME_DIR}

# Build + runtime system deps. build-essential is needed for any wheel
# that has to compile from source; libgl1 / libglib2.0-0 / libgomp1
# are needed by opencv-python-headless and openvino at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# --- layer: pyproject metadata only (caches dep install) ---
COPY pyproject.toml ./

# --- layer: install runtime deps WITHOUT app code ---
# Extract [project.dependencies] + [project.optional-dependencies].export
# from pyproject.toml and install just those. Python 3.11 has tomllib in
# stdlib, no extra deps needed for the extraction. This keeps the (large)
# dep install layer stable across source-only edits.
RUN python -c "import tomllib, sys; \
        data = tomllib.load(open('pyproject.toml','rb')); \
        deps = list(data['project']['dependencies']) + \
               list(data['project']['optional-dependencies'].get('export', [])); \
        sys.stdout.write('\n'.join(deps))" > /tmp/deps.txt \
    && pip install --prefix=/install -r /tmp/deps.txt \
    && rm /tmp/deps.txt

# Make the deps importable for the next two bake steps.
ENV PATH="/install/bin:${PATH}"
ENV PYTHONPATH="/install/lib/python3.11/site-packages"

# --- layer: bake YOLO + MobileCLIP into /models ---
# Slow (download + ONNX export), worth its own cache layer. Only scripts/
# is in the build context for this step so app/ edits don't invalidate.
COPY scripts ./scripts
RUN mkdir -p /models && python scripts/download_models.py all --out /models \
    && rm -rf /models/raw
# /models/raw holds the source PyTorch .pt downloads (~250 MB combined).
# Only the OpenVINO IR dirs are loaded at runtime — strip the raws.

# --- layer: pre-warm SigLIP-2 weights into HF cache ---
# ~370 MB into ${HF_HOME}/hub. Subsequent worker boots load from disk
# instead of pulling from huggingface.co. Cached unless SIGLIP_MODEL_ID
# changes via --build-arg.
RUN mkdir -p ${HF_HOME_DIR} && python -c "import os; \
        from transformers import AutoModel, AutoProcessor; \
        mid = '${SIGLIP_MODEL_ID}'; \
        AutoModel.from_pretrained(mid); \
        AutoProcessor.from_pretrained(mid); \
        print(f'siglip cached to {os.environ[\"HF_HOME\"]}')"

# --- layer: app source + self-install (no deps; deps already there) ---
# This is the fast layer every source edit will rebuild.
COPY app ./app
RUN pip install --prefix=/install --no-deps .

# ---------- runtime stage ----------
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime

ARG HF_HOME_DIR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/models \
    PROMPTS_FILE=/config/prompts.md \
    HF_HOME=${HF_HOME_DIR} \
    TRANSFORMERS_OFFLINE=1
# TRANSFORMERS_OFFLINE=1 prevents the worker from trying to call
# huggingface.co at runtime — everything must load from the baked cache.

# Runtime-only system libs. Notably NO build-essential / gcc — those
# stayed in the builder stage and don't ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Deps + worker package + baked models + warmed HF cache, in one COPY
# per source to minimize the runtime layer count.
COPY --from=builder /install /usr/local
COPY --from=builder /models /models
COPY --from=builder ${HF_HOME_DIR} ${HF_HOME_DIR}

# Readiness: app/main.py touches /tmp/ready once the pipeline is loaded
# (DESIGN.md §7.3). The k8s exec probe `cat /tmp/ready` will succeed
# once that file exists.

CMD ["python", "-m", "app.main"]
