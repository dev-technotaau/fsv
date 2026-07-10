"""F-Stain server-side inference on Google Cloud Run with L4 GPU.

Architecture
------------
Same protocol as the Modal version (`modal_inference/app_dinov3.py`) but
without any Modal-specific framework: plain FastAPI + uvicorn in a Docker
container. Browser uploads a fence photo via multipart POST → this Cloud Run
container runs the ONNX model on an L4 GPU → returns a PNG-encoded grayscale
mask (512x512, uint8 = sigmoid_prob * 255).

Why a separate copy of the inference code
-----------------------------------------
- Cloud Run wants a plain ASGI app, not Modal's `@app.function(...)` decorators
- Container image is built via Cloud Build / Dockerfile, not Modal's `image=`
- Model files are baked into the container layer instead of Modal's
  `.add_local_file()` (Cloud Run filesystem persists across requests on the
  same instance, which is what we want — no per-request file pulls)
- Lifecycle hooks are FastAPI's `lifespan` context manager rather than Modal's
  per-call function decoration. The ORT session is loaded ONCE per cold-start,
  then reused across all subsequent requests handled by that instance.

Endpoints
---------
    GET  /        -> health JSON (also wakes a cold instance — browser hits
                     this on page load so the instance starts warming before
                     the user clicks Apply Stain)
    POST /detect  -> multipart 'image' field, returns image/png mask
"""

import ctypes
import glob
import io
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage

# Render pipeline (added alongside /detect — Qwen renovation + exact-swatch finish).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color_finish as cf
import qwen_engine

# ─── Configuration ────────────────────────────────────────────────────
MODEL_FILE = os.environ.get("FSV_MODEL", "/model/fence_dinov3_phase1.onnx")
INPUT_SIZE = 512                                 # DINOv3 patch_size=16, 32*16=512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── render (/render) config ──────────────────────────────────────────
WORKING_RES = int(os.environ.get("FSV_WORKING_RES", "1024"))   # Qwen render resolution
FAMILY_CONTRAST = {"general": 1.0, "semi-transparent": 1.12, "semi-solid": 0.95}
_qwen_lock = threading.Lock()                    # concurrency=1 GPU + serialises Qwen lazy-load

# Browsers will POST from these origins. Same set as the Modal version.
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

# ─── Logging ──────────────────────────────────────────────────────────
# Cloud Logging captures stdout/stderr automatically; the explicit stream
# handler makes uvicorn + our logs land in one place with consistent format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("f-stain-dinov3")

# ─── Startup environment diagnostics ──────────────────────────────────
# Print enough to debug cold-start GPU issues without an SSH session.
# Cloud Logging captures these; you'll find them under "service: fsv-dinov3
# severity: INFO" in the Cloud Run logs for the cold-start instance.
logger.info(f"[startup] python={sys.version.split()[0]}  exe={sys.executable}")
logger.info(f"[startup] LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '<<UNSET>>')}")
logger.info(f"[startup] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<<UNSET>>')}")

# ─── CUDA library preload ─────────────────────────────────────────────
# onnxruntime-gpu needs cuBLAS, cuDNN, etc. dlopen'd at startup. Preloading
# them with RTLD_GLOBAL makes their symbols available to ORT's CUDA provider
# regardless of how it resolves its own libs. Without this, ORT silently
# falls back to CPU on some image configurations (~25x slower).
_CRITICAL_LIBS = [
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcufft.so.11",
    "libcurand.so.10",
    "libnvJitLink.so.12",
    "libnvrtc.so.12",
]
_nv_libs_found = []
for _pat in [
    "/usr/local/lib/python*/site-packages/nvidia/*/lib/lib*.so*",
    "/usr/lib/python*/site-packages/nvidia/*/lib/lib*.so*",
]:
    _nv_libs_found.extend(glob.glob(_pat))
logger.info(f"[startup] nvidia .so files found ({len(_nv_libs_found)}):")
for _so in sorted(set(_nv_libs_found))[:20]:
    logger.info(f"  {_so}")
for _libname in _CRITICAL_LIBS:
    _matches = [
        p for p in _nv_libs_found
        if p.endswith(f"/{_libname}") or p.endswith(f"/{_libname}.0")
    ]
    if _matches:
        try:
            ctypes.CDLL(_matches[0], mode=ctypes.RTLD_GLOBAL)
            logger.info(f"[startup] dlopen OK: {_libname}")
        except OSError as e:
            logger.warning(f"[startup] dlopen FAIL: {_libname}: {e}")
    else:
        logger.warning(f"[startup] dlopen SKIP: {_libname} not bundled in image")

# Log ORT version + provider availability AFTER the dlopen preload.
# get_available_providers() reflects what ORT can actually create sessions
# with — if "CUDAExecutionProvider" is NOT in this list, no amount of
# session-level provider= overriding will save us; the .so preload failed
# and ORT will silently fall back to CPU when the session is built.
logger.info(f"[startup] onnxruntime version: {ort.__version__}")
logger.info(f"[startup] onnxruntime available providers: {ort.get_available_providers()}")

# ─── Model + session lifecycle ────────────────────────────────────────
# Loaded once per cold-start via the lifespan context manager. Cloud Run
# reuses container instances across requests, so this stays warm until the
# instance is scaled down (after `scaledown_window` of no traffic).
_session: ort.InferenceSession | None = None
_input_name: str | None = None
_output_name: str | None = None
_active_providers: list[str] = []
_mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_std_arr  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 3, 1, 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX session at container boot. yield hands control to
    request handling; the after-yield block runs on graceful shutdown."""
    global _session, _input_name, _output_name, _active_providers
    logger.info(f"[startup] loading {MODEL_FILE}")
    t0 = time.time()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Seg stays on GPU (DINOv3L, fp32, unchanged). Keep onnxruntime LEAN so Qwen fits beside it on
    # the 24GB L4: kSameAsRequested (no arena doubling) + no cudnn max-workspace hold ORT to its
    # true measured working set ~5.3GB (2.66GB weights + 2.7GB activations) instead of the ~6.2GB
    # default. That ~0.9GB saving is what lets seg (5.3GB) + Qwen (15.7GB load peak) fit under
    # 21.96GB — a thin but real ~0.9GB margin. NO gpu_mem_limit (a cap below 5.3GB would starve
    # seg). If a render OOMs on VRAM, drop FSV_WORKING_RES to 768 to shrink Qwen's activations.
    _cuda_opts = {"arena_extend_strategy": "kSameAsRequested",
                  "cudnn_conv_use_max_workspace": "0",
                  "cudnn_conv_algo_search": "HEURISTIC"}
    _session = ort.InferenceSession(
        MODEL_FILE, sess_options=opts,
        providers=[("CUDAExecutionProvider", _cuda_opts), "CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    _active_providers = _session.get_providers()
    logger.info(
        f"[startup] ONNX session ready in {time.time() - t0:.1f}s  "
        f"providers={_active_providers}  "
        f"input={_input_name}  output={_output_name}"
    )
    yield
    logger.info("[shutdown] container stopping (SIGTERM)")


app = FastAPI(
    title="F-Stain DINOv3 Inference (Cloud Run)",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Ms", "X-Upload-Bytes", "X-Provider",
                    "X-DeltaE", "X-Seg-Ms", "X-Render-Ms", "X-Total-Ms"],
    max_age=86400,
)


@app.get("/")
async def health():
    """Health + wake. Browser hits this on page load to start an instance
    warming so the user's first Apply Stain click hits a hot session."""
    return {
        "status": "ok",
        "service": "f-stain-dinov3-cloudrun",
        "model_input_size": INPUT_SIZE,
        "channel_order": "RGB",
        "providers": _active_providers,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    if _session is None:
        # Should never happen — lifespan runs before request handling.
        raise HTTPException(status_code=503, detail="Model not loaded")

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
    # Bilinear resize is what the model was trained against; do NOT use
    # nearest-neighbor here or accuracy degrades on small fences.
    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0   # HWC fp32
    arr = arr.transpose(2, 0, 1)[None]                         # 1xCHW
    arr = (arr - _mean_arr) / _std_arr

    out = _session.run([_output_name], {_input_name: arr})[0]  # (1,1,512,512)
    probs = out[0, 0]                                           # (512, 512) sigmoid

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
        f"provider={_active_providers[0] if _active_providers else 'unknown'}"
    )

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Inference-Ms": str(elapsed_ms),
            "X-Upload-Bytes": str(len(contents)),
            "X-Provider": _active_providers[0] if _active_providers else "unknown",
            "Cache-Control": "no-cache, no-store",
        },
    )


# ─── render pipeline: renovate (Qwen) + exact-swatch finish ────────────
def _segment_native(img_rgb: np.ndarray) -> np.ndarray:
    """Reuse the loaded DINOv3 session -> soft mask [0,1] at native (H,W)."""
    H, W = img_rgb.shape[:2]
    im = PILImage.fromarray(img_rgb).resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
    arr = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
    arr = (arr - _mean_arr) / _std_arr
    probs = _session.run([_output_name], {_input_name: arr.astype(np.float32)})[0][0, 0]
    if probs.min() < 0 or probs.max() > 1:
        probs = 1.0 / (1.0 + np.exp(-probs))
    return np.clip(cv2.resize(probs.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)


@app.post("/render")
async def render(image: UploadFile = File(...), colorHex: str = Form(...),
                 family: str = Form("general"), tone: str = Form("warm reddish cedar brown"),
                 seed: int = Form(0), mask: UploadFile = File(None)):
    """Renovate the fence (Qwen) -> re-impose the EXACT swatch colour + composite over the
    original. Reuses the loaded DINOv3 session for the mask (or an optional supplied one).
    Qwen is lazy-loaded on the first call so /detect is unaffected until then."""
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.time()
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "upload too large (>20MB)")
    try:
        orig = np.array(PILImage.open(io.BytesIO(data)).convert("RGB"))
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")
    H, W = orig.shape[:2]

    if mask is not None:                          # reuse the /detect mask if the browser sends it
        md = await mask.read()
        m = np.array(PILImage.open(io.BytesIO(md)).convert("L"), np.float32) / 255.0
        mask_arr = m if m.shape[:2] == (H, W) else cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_arr = np.clip(mask_arr, 0, 1)
    else:                                         # else segment here with the same GPU model
        mask_arr = _segment_native(orig)
    if (mask_arr > 0.5).mean() < 0.01:
        raise HTTPException(422, "no fence detected")
    t_seg = time.time()

    scale = WORKING_RES / max(H, W)
    wW, wH = max(64, round(W * scale)), max(64, round(H * scale))
    work = PILImage.fromarray(orig).resize((wW, wH), PILImage.LANCZOS)
    with _qwen_lock:                              # concurrency=1 GPU + serialises the lazy first-load
        ren = qwen_engine.renovate(work, tone=tone, seed=seed)
    t_ren = time.time()

    final = cf.finish(orig, np.array(ren.convert("RGB")), mask_arr, colorHex,
                      contrast=FAMILY_CONTRAST.get(family, 1.0))
    de = cf.delta_e_median(final, mask_arr, colorHex)
    buf = io.BytesIO()
    PILImage.fromarray(final).save(buf, "JPEG", quality=92)
    logger.info(f"[render] {colorHex} {family} seg={int((t_seg-t0)*1000)}ms "
                f"render={int((t_ren-t_seg)*1000)}ms total={int((time.time()-t0)*1000)}ms dE={de:.2f}")
    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-DeltaE": f"{de:.2f}", "X-Seg-Ms": str(int((t_seg - t0) * 1000)),
        "X-Render-Ms": str(int((t_ren - t_seg) * 1000)), "X-Total-Ms": str(int((time.time() - t0) * 1000)),
        "Cache-Control": "no-store"})
