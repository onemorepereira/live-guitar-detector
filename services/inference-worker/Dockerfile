# Inference worker production image.
#
# Phase 3.3: builds a slim runtime with model weights baked in.
#
# Deviation from DESIGN.md §6.2 (which names openvino/ubuntu22_runtime
# as the base): we use python:3.11-slim and install openvino via pip.
# The openvino base ships Python 3.10, which conflicts with the worker's
# `>=3.11,<3.12` requires-python pin. Pip-installed openvino is
# functionally equivalent; revisit if benchmarks ever justify the base.
#
# Size target (DESIGN.md §6.2): < 1.5 GB.
#
# Build context: services/inference-worker.

# ---- builder stage: install deps + bake models ----
# Pin to a specific patch release; never the floating 3.11-slim tag.
FROM python:3.11.10-slim AS builder

# System deps needed for model download/export + opencv runtime probes.
# build-essential / gcc are required for any wheels that compile from
# source on this platform; libgl1 / libglib2.0-0 / libgomp1 are needed
# by opencv-python-headless and openvino at import time (which the
# export script triggers).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Install Python deps FIRST (cached layer). The model download below
# changes far less often than source code, but we still want source-only
# edits to NOT invalidate the (large) pip install layer. The [export]
# extra pulls in onnx + onnxscript needed by download_models.py.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install ".[export]"

# Make the just-installed package available to the model-download step.
ENV PATH="/install/bin:${PATH}"
ENV PYTHONPATH="/install/lib/python3.11/site-packages"

# Bake models into /models. This is the slowest layer (~30s download +
# ~30s ONNX -> IR export, network-dependent). Ordered AFTER pip install
# so source-only changes don't invalidate it. scripts/ is copied last so
# edits to non-script files in the source tree don't bust this layer.
COPY scripts ./scripts
RUN mkdir -p /models && python scripts/download_models.py all --out /models \
    && rm -rf /models/raw
# The /models/raw subdir holds the source PyTorch .pt downloads (~250 MB
# combined: YOLO + MobileCLIP state dict). Only the OpenVINO IR dirs
# (yolov8n-oiv7-fp32/, mobileclip-image-fp16/, mobileclip-text-fp16/)
# are loaded at runtime — strip the raws to keep the image lean.

# ---- runtime stage ----
FROM python:3.11.10-slim AS runtime

# Runtime-only system libs for opencv-python-headless + openvino.
# build-essential / gcc are intentionally NOT carried over.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in deps + the worker package itself (installed via --prefix in
# builder) in one shot.
COPY --from=builder /install /usr/local
# Bring in baked models.
COPY --from=builder /models /models

ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/models
ENV PROMPTS_FILE=/config/prompts.md
# OpenVINO threading default of "0 (auto)" works on Linux containers.

# Readiness: app/main.py touches /tmp/ready once the pipeline is loaded
# (DESIGN.md §7.3). The k8s exec probe `cat /tmp/ready` will succeed
# once that file exists.

CMD ["python", "-m", "app.main"]
