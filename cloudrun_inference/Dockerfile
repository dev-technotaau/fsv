# Cloud Run GPU container image for F-Stain DINOv3 inference.
#
# Base image choice: python:3.11-slim instead of nvidia/cuda. The nvidia-cu12
# pip packages bundle every user-space CUDA .so file onnxruntime-gpu needs;
# the GPU host (Cloud Run L4) provides only the kernel driver. Avoiding the
# nvidia/cuda base image saves ~1 GB of layer size and skips the externally-
# managed-environment pip restriction.
FROM python:3.11-slim

WORKDIR /app

# curl + ca-certs for downstream debugging (e.g. curling the health endpoint
# from inside the container during a `gcloud run services proxy` session).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in their own layer so requirements changes don't bust
# the model-file layers below — model file is 2.5 GB so reusing that layer
# across builds is a big speed-up.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# CRITICAL: LD_LIBRARY_PATH must be set BEFORE python starts so the dynamic
# linker can find the nvidia-cu12 .so files. Setting os.environ inside
# Python is too late — glibc's linker initializes once at process start and
# won't re-read the env var. Without this, CUDAExecutionProvider silently
# falls back to CPU (~25x slower) on most container configurations.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib

# Bake the model files into the image. Cloud Run scales container instances
# horizontally; every cold-start reads these from the local filesystem.
# Pulling from GCS at startup would add 5-15s per cold start AND incur
# per-request egress costs. The image gets larger (~2.5 GB), but Cloud Run
# caches images at the regional layer so pulls are fast within the region.
#
# Build-context expectation: both files must be in the same directory as
# this Dockerfile. See deploy.sh for the expected layout.
COPY fence_dinov3_phase1.onnx        /model/fence_dinov3_phase1.onnx
COPY fence_dinov3_phase1.onnx.data   /model/fence_dinov3_phase1.onnx.data

# Application code (last layer — cheapest to rebuild on code-only changes).
COPY app.py .

# Cloud Run sends SIGTERM with a 10s grace period before SIGKILL. uvicorn's
# default signal handler honors this — finishes in-flight requests then
# exits cleanly via the lifespan after-yield block.
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Single uvicorn worker. Multiple workers would each load a copy of the
# 2.5 GB ONNX session into VRAM, which doesn't fit on a single L4 (24 GB).
# Concurrency is configured at the Cloud Run service level instead.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info
