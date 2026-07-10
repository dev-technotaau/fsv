# Cloud Run L4 image: DINOv3 segmentation (/detect) + Qwen renovation (/render).
#
# Same python:3.11-slim base + nvidia-cu12 pip CUDA libs as the working /detect service —
# torch (cu124) reuses those same libs, so /detect's GPU path is unchanged. The PRE-QUANTIZED
# 4-bit Qwen checkpoint (~12-15GB) is baked in like the ONNX, so it loads from local disk in
# ~1-2 min — no GCS bucket, no runtime quantization. Image ~25GB (bakeable; the full 40GB bf16
# was not).
FROM python:3.11-slim

WORKDIR /app

# curl/ca-certs for debugging; libglib2.0-0 + libgl1 for opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps in their own layer (torch cu124 + Qwen stack + onnxruntime-gpu).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# CRITICAL (unchanged from /detect): LD_LIBRARY_PATH must be set before python starts so the
# dynamic linker finds the nvidia-cu12 .so files onnxruntime-gpu dlopens. torch installs its
# CUDA libs into these same nvidia/*/lib dirs.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/site-packages/torch/lib

# Bake the PRE-QUANTIZED 4-bit Qwen checkpoint (~12-15GB) + the Lightning LoRA into the image.
# local_dir gives clean real-file dirs the engine loads from /model/qwen-edit + /model/qwen-lightning.
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ovedrive/Qwen-Image-Edit-2509-4bit', local_dir='/model/qwen-edit'); \
    snapshot_download('lightx2v/Qwen-Image-Lightning', allow_patterns=['Qwen-Image-Edit-2509/*'], \
                      local_dir='/model/qwen-lightning')"

# DINOv3 model files (unchanged).
COPY fence_dinov3_phase1.onnx        /model/fence_dinov3_phase1.onnx
COPY fence_dinov3_phase1.onnx.data   /model/fence_dinov3_phase1.onnx.data

# Application code (last layer). color_finish + qwen_engine are imported by app.py.
COPY color_finish.py qwen_engine.py app.py ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Reduce CUDA fragmentation during the model load (the diffusers caching-allocator warmup
# can spike; this lets reserved-but-unallocated blocks be reused). Must be set before torch inits.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single uvicorn worker (one copy of each model in VRAM). Cloud Run concurrency=1.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info
