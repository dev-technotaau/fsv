# Cloud Run L4 image: DINOv3 segmentation (/detect) + Qwen renovation (/render).
#
# python:3.11-slim + nvidia-cu12 (cu128) pip CUDA libs. torch (cu128) + onnxruntime-gpu share those
# libs. RENDER ENGINE = Nunchaku SVDQuant INT4 (fast fused INT4 kernels, ~3x vs bnb-NF4, full 20-step
# sharpness): the 12.7GB INT4 transformer + the ovedrive snapshot (reused only for its 4-bit text
# encoder + VAE + configs) are baked in like the ONNX -> load from local disk, no bucket. Image ~30GB.
FROM python:3.11-slim

WORKDIR /app

# curl/ca-certs for debugging; libglib2.0-0 + libgl1 for opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps in their own layer (torch cu128 + Nunchaku + Qwen stack + onnxruntime-gpu).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# Fail the build early if the Nunchaku wheel didn't install against this exact torch, or if ORT
# didn't install cleanly. Import the REAL class the engine uses (nunchaku has no __version__).
# (GPU providers can't be verified here — no GPU at build; checked at runtime via the startup log.)
RUN python -c "import torch, onnxruntime; \
    from nunchaku import NunchakuQwenImageTransformer2DModel; from nunchaku.utils import get_precision; \
    print('OK torch', torch.__version__, '| ort', onnxruntime.__version__, onnxruntime.get_available_providers())"

# CRITICAL (unchanged from /detect): LD_LIBRARY_PATH must be set before python starts so the
# dynamic linker finds the nvidia-cu12 .so files onnxruntime-gpu dlopens. torch installs its
# CUDA libs into these same nvidia/*/lib dirs.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/site-packages/torch/lib

# Bake the render weights (loaded from local disk, no bucket):
#  * Nunchaku INT4 transformer (12.7GB, r128 = max sharpness) -> /model/nunchaku-qwen
#  * ovedrive 4-bit snapshot MINUS its transformer -> /model/qwen-edit (reused only for the
#    4-bit text_encoder + VAE + tokenizer + scheduler configs; the Nunchaku transformer overrides it).
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nunchaku-tech/nunchaku-qwen-image-edit-2509', \
                      allow_patterns=['svdq-int4_r128-qwen-image-edit-2509.safetensors'], \
                      local_dir='/model/nunchaku-qwen'); \
    snapshot_download('ovedrive/Qwen-Image-Edit-2509-4bit', local_dir='/model/qwen-edit', \
                      ignore_patterns=['transformer/*'])"

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
