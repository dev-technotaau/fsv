"""F-Stain server-side inference on Modal — DINOv3-L variant (Phase 1).

Architecture
------------
Same protocol as app.py but with the flagship DINOv3-L + ViT-Adapter +
MSDeform + Mask2Former + refinement head model. Browser uploads a fence
photo via multipart POST → this Modal container runs the ONNX model
(GPU T4) → returns a PNG-encoded grayscale mask (512x512, uint8 =
sigmoid_prob * 255).

Why a separate app from app.py
------------------------------
- Keeps the production DINOv2-small endpoint untouched while we test
  the heavier model.
- Different model size (2.47 GB vs 135 MB), input size (512 vs 518),
  and compute requirements (GPU vs CPU).
- Different APP_NAME so Modal deploys it as its own URL.

Deploy
------
    pip install modal
    modal token new                # one-time browser auth
    cd modal_inference/
    modal deploy app_dinov3.py

After deploy, Modal prints the URL. Paste `<URL>/detect` into the
matching CONFIG.MODAL_ENDPOINT inside the DINOv3 copy of index2.html.

Endpoints
---------
    GET  /        -> health JSON (also wakes a cold container)
    POST /detect  -> multipart 'image' field, returns image/png mask
"""

import modal

APP_NAME = "f-stain-dinov3-inference"

# Browsers will POST from these origins. Same set as app.py — covers
# huggingface.co + localhost dev servers + any *.static.hf.space subdomain.
ALLOWED_ORIGINS = [
    "https://huggingface.co",
    "https://ninjafencestaining.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGIN_REGEX = r"https://[A-Za-z0-9._-]+\.static\.hf\.space"

MODEL_FILE_REMOTE = "/model/fence_dinov3_phase1.onnx"
MODEL_DATA_REMOTE = "/model/fence_dinov3_phase1.onnx.data"
INPUT_SIZE = 512                                 # DINOv3 patch_size=16, 512 = 32*16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

app = modal.App(APP_NAME)

# Container image: debian + GPU-capable onnxruntime + both ONNX files (the
# graph .onnx AND its external-data .onnx.data — the model uses external
# data format because total weights exceed the 2GB ONNX protobuf limit).
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime-gpu==1.24.4",        # GPU runtime (CUDA 12.x compatible)
        # CUDA 12 + cuDNN 9 runtime libs — onnxruntime-gpu 1.24 needs these
        # at runtime to create CUDAExecutionProvider. debian_slim base image
        # doesn't include CUDA toolkit; gpu="T4" provides only the kernel
        # driver. These pip packages provide the user-space .so files
        # (cublasLt, cudnn, etc.) under site-packages/nvidia/<lib>/lib/.
        # Without them, ORT silently falls back to CPU (~25x slower).
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cudnn-cu12==9.10.2.21",   # pin to the version used in training
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-nvjitlink-cu12",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
    )
    # CRITICAL: LD_LIBRARY_PATH must be set at CONTAINER START so the dynamic
    # linker can find the nvidia-cu12 .so files bundled in the pip packages.
    # Setting os.environ['LD_LIBRARY_PATH'] inside Python at runtime is TOO
    # LATE — the linker initializes once when Python starts and won't re-read
    # the env var. Without this, CUDAExecutionProvider silently falls back to
    # CPU (we saw this on first deploy attempt).
    .env({"LD_LIBRARY_PATH": ":".join([
        f"/usr/local/lib/python3.11/site-packages/nvidia/{_lib}/lib"
        for _lib in ["cublas", "cuda_runtime", "cuda_nvrtc",
                      "cudnn", "cufft", "curand", "nvjitlink"]
    ])})
    .add_local_file(
        "../models/fence_dinov3_phase1.onnx",
        MODEL_FILE_REMOTE,
    )
    .add_local_file(
        "../models/fence_dinov3_phase1.onnx.data",
        MODEL_DATA_REMOTE,
    )
)

# Hoist runtime imports to MODULE scope (only execute inside the Modal
# container — locally during `modal deploy` they are skipped). FastAPI's
# type introspection resolves endpoint parameter types from each function's
# __globals__, so types like `UploadFile` MUST be at module scope or the
# route registration crashes with a ForwardRef error.
with app_image.imports():
    import io
    import os
    import sys
    import glob
    import ctypes
    import time
    import logging

    # ── Diagnostics: print what the container actually sees ──────────────
    print(f"[startup] python={sys.version.split()[0]}  exe={sys.executable}", flush=True)
    print(f"[startup] LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '<<UNSET>>')}", flush=True)
    print(f"[startup] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<<UNSET>>')}", flush=True)
    # Find nvidia .so files actually present in the image
    _nv_libs_found = []
    for _pat in [
        "/usr/local/lib/python*/site-packages/nvidia/*/lib/lib*.so*",
        "/usr/lib/python*/site-packages/nvidia/*/lib/lib*.so*",
    ]:
        _nv_libs_found.extend(glob.glob(_pat))
    print(f"[startup] nvidia .so files found ({len(_nv_libs_found)}):", flush=True)
    for _so in sorted(set(_nv_libs_found))[:20]:
        print(f"  {_so}", flush=True)

    # Explicitly dlopen the critical CUDA libs with RTLD_GLOBAL so symbols
    # resolve in subsequently-loaded ORT CUDA provider. This is more
    # reliable than relying on the linker to find them via LD_LIBRARY_PATH.
    _CRITICAL_LIBS = [
        "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12",
        "libcudnn.so.9", "libcufft.so.11", "libcurand.so.10",
        "libnvJitLink.so.12", "libnvrtc.so.12",
    ]
    for _libname in _CRITICAL_LIBS:
        # find the actual path among the discovered libs
        _matches = [p for p in _nv_libs_found if p.endswith(f"/{_libname}") or p.endswith(f"/{_libname}.0")]
        if _matches:
            try:
                ctypes.CDLL(_matches[0], mode=ctypes.RTLD_GLOBAL)
                print(f"[startup] dlopen OK: {_libname} -> {_matches[0]}", flush=True)
            except OSError as _e:
                print(f"[startup] dlopen FAIL: {_libname}: {_e}", flush=True)
        else:
            print(f"[startup] dlopen SKIP: {_libname} not found in image", flush=True)

    import numpy as np
    import onnxruntime as ort
    print(f"[startup] onnxruntime version: {ort.__version__}", flush=True)
    print(f"[startup] onnxruntime available providers: {ort.get_available_providers()}", flush=True)
    from PIL import Image as PILImage
    from fastapi import FastAPI, UploadFile, File, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware


@app.function(
    image=app_image,
    gpu="T4",                     # T4 16GB — cheapest Modal GPU, fits our 2.47GB model + activations easily
    cpu=2.0,                      # CPU is just orchestration (GPU does the inference)
    memory=4096,                  # 4 GB RAM (CPU side — for image decode + pre/post processing)
    scaledown_window=600,         # keep warm 10 min after last request (GPU cold-start hurts more than CPU)
    min_containers=0,             # scale to zero when idle (free)
    max_containers=4,             # cap concurrent containers (cost guardrail — T4 is more expensive than CPU)
)
@modal.concurrent(max_inputs=4)   # one container can handle 4 parallel requests (16GB VRAM accommodates)
@modal.asgi_app()
def web():
    logger = logging.getLogger("f-stain-dinov3")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load the ONNX model ONCE per container (cached in this closure across
    # subsequent requests handled by the same container). CUDA provider
    # primary, CPU fallback in case CUDA init fails on a degraded node.
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        MODEL_FILE_REMOTE,
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active_providers = session.get_providers()
    mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
    logger.info(
        f"[startup] ONNX session ready  "
        f"providers={active_providers}  "
        f"inputs={[i.name for i in session.get_inputs()]}  "
        f"outputs={[o.name for o in session.get_outputs()]}"
    )

    fastapi_app = FastAPI(title="F-Stain DINOv3 Inference", docs_url=None, redoc_url=None)

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Inference-Ms", "X-Upload-Bytes", "X-Provider"],
        max_age=86400,
    )

    @fastapi_app.get("/")
    async def health():
        """Health + wake. Browser hits this on page load to start the container
        warming so the user's first `Detect` click hits a hot session."""
        return {
            "status": "ok",
            "service": APP_NAME,
            "model_input_size": INPUT_SIZE,
            "channel_order": "RGB",
            "providers": active_providers,
        }

    @fastapi_app.post("/detect")
    async def detect(image: UploadFile = File(...)):
        t0 = time.time()

        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload too large (>20 MB)")

        try:
            img = PILImage.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # Preprocess → NCHW [1,3,512,512] ImageNet-normalized fp32.
        # Bilinear resize is what the model was trained against; do NOT
        # use nearest-neighbor here or accuracy degrades on small fences.
        img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0   # HWC fp32
        arr = arr.transpose(2, 0, 1)[None]                         # 1xCHW
        arr = (arr - mean_arr) / std_arr

        # Inference (output name is 'mask_prob' for the refined-head export)
        out_name = session.get_outputs()[0].name
        in_name = session.get_inputs()[0].name
        out = session.run([out_name], {in_name: arr})[0]   # (1,1,512,512)
        probs = out[0, 0]                                    # (512, 512) sigmoid

        # Encode as PNG grayscale (compact lossless within uint8 precision —
        # 1/255 ≈ 0.4% loss, well below any threshold the browser applies).
        mask_uint8 = (probs * 255.0).clip(0, 255).astype(np.uint8)
        mask_img = PILImage.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG", optimize=True, compress_level=6)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[detect] in={len(contents)/1024:.1f}KB  "
            f"out={buf.tell()/1024:.1f}KB  total={elapsed_ms}ms  "
            f"provider={active_providers[0] if active_providers else 'unknown'}"
        )

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Inference-Ms": str(elapsed_ms),
                "X-Upload-Bytes": str(len(contents)),
                "X-Provider": active_providers[0] if active_providers else "unknown",
                "Cache-Control": "no-cache, no-store",
            },
        )

    return fastapi_app
