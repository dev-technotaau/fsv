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
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
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
# WORKING_RES defines the render AREA TARGET (WORKING_RES², ~1MP at 1024 = diffusers' stock output
# area). The crop-to-fence path renders the fence bbox at this same pixel scale — fewer tokens, same
# sharpness. NOTE: values >1024 do NOT sharpen full-frame renders (the pipeline caps area at its
# trained ~1MP); values <1024 (e.g. 896) trade a little sharpness for real speed.
WORKING_RES = int(os.environ.get("FSV_WORKING_RES", "1024"))
CROP_MAX_FRAC = float(os.environ.get("FSV_CROP_MAX_FRAC", "0.8"))  # >this bbox fraction => full frame
# Response JPEG quality. Measured: the returned file is the SECOND biggest cost after
# diffusion (a 3MB upload came back as a 2.84MB JPEG at q92, ~8s of network). q85 roughly
# halves that with no visible difference — the wood detail comes from the render, not the
# encoder. Env-tunable so it can be changed without rebuilding the image.
JPEG_QUALITY = int(os.environ.get("FSV_JPEG_QUALITY", "85"))
# contrast scales the luminance SPREAD (grain shadows, plank-gap depth, board-to-board variation)
# for more realistic depth/shadow/contrast — dE-SAFE (the median colour stays the exact swatch).
# general nudged 1.0 -> 1.08 (gentle; not aggressive). Override per-family via env if needed.
FAMILY_CONTRAST = {"general": float(os.environ.get("FSV_CONTRAST", "1.08")),
                   "semi-transparent": 1.15, "semi-solid": 1.0}
# chroma_retain lets a touch of the render's real colour variation through for "depth of colour".
# >0 raises dE slightly (watch X-DeltaE stays <=3); 0.06 is subtle. 0 = pure exact swatch.
CHROMA_RETAIN = float(os.environ.get("FSV_CHROMA_RETAIN", "0.06"))
_qwen_lock = threading.Lock()                    # concurrency=1 GPU + serialises Qwen lazy-load
_qwen_ready = threading.Event()                  # set once the background warm finishes

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


def _weight_paths() -> list[str]:
    paths = []
    tf = os.environ.get("NUNCHAKU_TRANSFORMER",
                        "/model/nunchaku-qwen/svdq-int4_r128-qwen-image-edit-2509.safetensors")
    if os.path.exists(tf):
        paths.append(tf)
    paths += sorted(glob.glob("/model/qwen-edit/text_encoder/*.safetensors"))
    return paths


def _mem_mb() -> int:
    """Container memory usage in MB from the cgroup — what Cloud Run's 32Gi limit actually meters.
    Tries cgroup v2 then v1 (Cloud Run gen2 has been seen as v1 -> the v2-only path returned -1)."""
    for path in ("/sys/fs/cgroup/memory.current",                # cgroup v2
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):  # cgroup v1
        try:
            with open(path) as f:
                return int(f.read().strip()) >> 20
        except Exception:
            continue
    return -1


def _vram() -> str:
    """GPU memory in MB. Three different numbers, because they answer different questions:
      used      - driver-level, WHOLE process: Qwen + the ONNX CUDA arena + the CUDA context.
                  This is what nvidia-smi shows and the only one that tells you what GPU to buy.
      peak_torch- torch's high-water ALLOCATED mark since the last reset, so per-render.
      reserved  - what torch's caching allocator holds. Expect this to sit near the high-water mark
                  and not fall: qwen_engine no-ops empty_cache() during a render on purpose, so
                  freed blocks are kept for reuse rather than returned. High reserved is not a leak.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "vram=n/a"
        free, total = torch.cuda.mem_get_info()
        return (f"vram_used={(total - free) >> 20}MB peak_torch={torch.cuda.max_memory_allocated() >> 20}MB "
                f"reserved={torch.cuda.memory_reserved() >> 20}MB vram_total={total >> 20}MB")
    except Exception:
        return "vram=n/a"


def _reset_vram_peak():
    """Make peak_torch mean 'this render' rather than 'since the container started'."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _drop_weight_cache():
    """Evict the baked weight files from the page cache once the model is loaded — the bytes now
    live in the model's own memory and the ~18GB of file cache (parts of it pinned by nunchaku's
    mmap load) is pure pressure. Without this, the 12.7GB transformer bouncing GPU->CPU at the END
    of a render allocates faster than cgroup reclaim frees cache -> signal-9 OOM at 32Gi (observed
    rev 00012: killed ~18s AFTER denoise 20/20). Advisory + harmless if already evicted."""
    for path in _weight_paths():
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass


def _prefetch_weights():
    """Warm the page cache for the big baked Qwen files by reading them in parallel slices,
    overlapped with ONNX init. Cloud Run streams the container image lazily — the first read of the
    12.7GB INT4 file is network-bound (~150-300MB/s single-stream); 16 concurrent ranged reads both
    overlap that wait with ONNX/GPU init and (if the streaming backend parallelizes ranges like GCS
    does) can multiply throughput. Read-and-discard: the payoff is the warm page cache. The cache
    is dropped again by _drop_weight_cache() as soon as the model has loaded."""
    from concurrent.futures import ThreadPoolExecutor
    paths = _weight_paths()
    t0 = time.time()
    for path in paths:                                   # big file first; sequential across files
        try:
            size = os.path.getsize(path)
            n = 16
            sl = size // n + 1
            def _read_slice(i, _p=path, _sl=sl, _size=size):
                fd = os.open(_p, os.O_RDONLY)
                try:
                    off, end = i * _sl, min((i + 1) * _sl, _size)
                    while off < end:
                        chunk = os.pread(fd, min(8 << 20, end - off), off)
                        if not chunk:
                            break
                        off += len(chunk)
                finally:
                    os.close(fd)
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(_read_slice, range(n)))
        except Exception as e:
            logger.warning(f"[startup] prefetch {path} failed: {e}")
            return
    logger.info(f"[startup] weight prefetch done ({len(paths)} files) in {time.time() - t0:.1f}s")


# Throwaway denoise run at startup so the first REAL render doesn't pay one-time GPU costs.
# Measured on the L40S: the same photo took 15.2s as a container's first render and 11.7s as its
# fourth — identical output dims, ~3.5s of pure one-time cost. Loading the weights (what the warm
# thread used to do on its own) does NOT trigger it: nunchaku's INT4 CUDA modules are loaded
# lazily on first use, the attention backend is chosen on first call, and cuBLAS/cuDNN pick
# algorithms and allocate workspaces on first call. All of that landed on whoever clicked first.
# 512x512 not 1024x896: most of the cost is shape-independent (module load, backend selection), so
# a small run captures it at a fraction of the time. Steps=2 because true_cfg_scale>1 makes each
# step two transformer forwards — two steps is enough to exercise the loop, the scheduler and the
# VAE decode.
WARM_RENDER_PX = int(os.environ.get("FSV_WARM_RENDER_PX", "512"))   # 0 disables


def _warm_render():
    """Run one tiny render to force lazy GPU init. Best-effort: never fatal, never blocks readiness
    on success of the render itself — a failure here only means the first user pays what they used
    to pay. Takes _qwen_lock because renovate() patches module-level globals (VAE_IMAGE_SIZE and
    torch.cuda.empty_cache) and documents that the caller must hold it."""
    if WARM_RENDER_PX <= 0:
        return
    try:
        t = time.time()
        blank = PILImage.new("RGB", (WARM_RENDER_PX, WARM_RENDER_PX), (128, 110, 90))
        with _qwen_lock:
            qwen_engine.renovate(blank, steps=2, height=WARM_RENDER_PX, width=WARM_RENDER_PX)
        logger.info(f"[startup] warm render ({WARM_RENDER_PX}px, 2 steps) in {time.time() - t:.1f}s "
                    f"— first real render no longer pays lazy CUDA init  {_vram()}")
    except Exception as e:
        logger.warning(f"[startup] warm render failed ({e}); the first /render will be ~3.5s slower")


def _warm_segmentation():
    """One dummy /detect-shaped inference so the first REAL one doesn't pay lazy init.

    Measured: first /detect on a fresh container 1131ms vs 335ms once warm. ORT defers cuDNN
    algorithm selection and workspace allocation to the first run of a given input shape — and
    under FSV_ORT_FAST=1 that first run is an EXHAUSTIVE benchmark of every conv algorithm, which
    is markedly slower still. The shape is always INPUT_SIZE x INPUT_SIZE, so warming it once here
    covers every subsequent request. Runs BEFORE the Qwen load because /detect is already being
    served by the time this thread starts, and it needs no lock (Qwen owns _qwen_lock, not ORT)."""
    try:
        t = time.time()
        _segment_native(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8))
        logger.info(f"[startup] segmentation warmed in {time.time() - t:.1f}s")
    except Exception as e:
        logger.warning(f"[startup] segmentation warm failed ({e}); first /detect will be slower")


def _warm_qwen():
    """Background Qwen warm — runs while the server is ALREADY serving /detect. A /render that
    arrives mid-warm simply blocks on _qwen_lock until the load finishes."""
    try:
        tw = time.time()
        _warm_segmentation()
        with _qwen_lock:
            qwen_engine.load()
        _drop_weight_cache()   # weights are in the model now; free ~18GB of cache headroom
        logger.info(f"[startup] Qwen loaded in {time.time() - tw:.1f}s (background)  mem={_mem_mb()}MB")
        # Only NOW is the service genuinely fast, so qwen_ready flips after this rather than after
        # the load — the frontend polls that flag to decide the user won't hit a slow first render.
        _warm_render()
        _qwen_ready.set()
        logger.info(f"[startup] Qwen warm complete in {time.time() - tw:.1f}s  mem={_mem_mb()}MB")
    except Exception as e:
        # Deliberately leave qwen_ready false, as before: /render still lazy-loads and sets the flag
        # itself, and reporting "ready" for a model that failed to load would be a lie.
        logger.warning(f"[startup] Qwen warm failed ({e}); will lazy-load on first /render")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX session at container boot. yield hands control to
    request handling; the after-yield block runs on graceful shutdown."""
    global _session, _input_name, _output_name, _active_providers
    # Kick off the weight prefetch FIRST so the network-bound file reads overlap the ONNX init.
    if os.environ.get("QWEN_WARM_ON_START", "1") == "1":
        threading.Thread(target=_prefetch_weights, daemon=True, name="qwen-prefetch").start()
    logger.info(f"[startup] loading {MODEL_FILE}")
    t0 = time.time()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Seg stays on GPU (DINOv3L, fp32, unchanged). Two profiles, because this file is deployed
    # VERBATIM to both a 24GB L4 (Cloud Run) and a 48GB L40S (Modal) and they want opposite things.
    #
    # LEAN (default, REQUIRED on the L4): kSameAsRequested (no arena doubling) + no cudnn
    #   max-workspace hold ORT to its true measured working set ~5.3GB (2.66GB weights + 2.7GB
    #   activations) instead of the ~6.2GB default. That ~0.9GB saving is what lets seg (5.3GB) +
    #   Qwen (15.7GB load peak) fit under 21.96GB — a thin but real ~0.9GB margin. NO gpu_mem_limit
    #   (a cap below 5.3GB would starve seg). If a render OOMs on VRAM, drop FSV_WORKING_RES.
    #
    # FAST (FSV_ORT_FAST=1, big-GPU only): lets ORT have what it actually wants. kNextPowerOfTwo is
    #   ORT's own default arena growth; max-workspace + EXHAUSTIVE let cuDNN benchmark every conv
    #   algorithm and keep the fastest instead of guessing from a heuristic. Costs VRAM (~1GB) and a
    #   ONE-OFF benchmarking pass on the first inference of each input shape — the shape here is
    #   always 512x512, so it happens once per container and _warm_render() absorbs it. Measured
    #   steady-state /detect is ~335ms end to end, of which the ONNX run is a fraction, so expect
    #   tens of milliseconds from this, not a step change. Do NOT set this on the L4.
    if os.environ.get("FSV_ORT_FAST", "0") == "1":
        _cuda_opts = {"arena_extend_strategy": "kNextPowerOfTwo",
                      "cudnn_conv_use_max_workspace": "1",
                      "cudnn_conv_algo_search": "EXHAUSTIVE"}
        logger.info("[startup] ORT profile=FAST (max workspace + EXHAUSTIVE conv search)")
    else:
        _cuda_opts = {"arena_extend_strategy": "kSameAsRequested",
                      "cudnn_conv_use_max_workspace": "0",
                      "cudnn_conv_algo_search": "HEURISTIC"}
        logger.info("[startup] ORT profile=LEAN (VRAM-capped; set FSV_ORT_FAST=1 on a >=40GB GPU)")
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
    # Warm Qwen in the BACKGROUND: the port opens (and /detect serves) as soon as ONNX is ready
    # (~25s) instead of after the full ~2min Qwen load. The frontend pings GET / on page load /
    # photo select, so the warm overlaps the time the user spends masking + picking a colour.
    if os.environ.get("QWEN_WARM_ON_START", "1") == "1":
        threading.Thread(target=_warm_qwen, daemon=True, name="qwen-warm").start()
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
                    "X-DeltaE", "X-Seg-Ms", "X-Render-Ms", "X-Total-Ms",
                    "X-Crop-Frac", "X-Render-Dims"],
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
        "qwen_ready": _qwen_ready.is_set(),   # background warm finished => /render is instant-start
        "host_mb": _mem_mb(),                 # cgroup usage — what the memory= limit meters
        "gpu": _vram(),                       # right-size the GPU/RAM from real numbers, not guesses
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
async def render(request: Request,
                 image: UploadFile = File(...), colorHex: str = Form(...),
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

    # Crop-to-fence: only masked fence pixels survive the composite, so renovate JUST the fence
    # bbox (+margin) at the SAME pixel scale a full-frame render would use — near-linear token/time
    # savings (fence at 40% of frame ≈ 2.6x faster), identical fence sharpness by construction.
    # Falls back to the whole frame when the fence dominates (frac > CROP_MAX_FRAC).
    ys, xs = np.where(mask_arr > 0.5)             # non-empty: coverage checked above
    mg = max(32, round(0.04 * max(H, W)))         # margin >> the 2px composite feather
    cy0, cx0 = max(0, int(ys.min()) - mg), max(0, int(xs.min()) - mg)
    cy1, cx1 = min(H, int(ys.max()) + 1 + mg), min(W, int(xs.max()) + 1 + mg)
    frac = ((cy1 - cy0) * (cx1 - cx0)) / (H * W)
    if frac > CROP_MAX_FRAC:
        cy0, cx0, cy1, cx1, frac = 0, 0, H, W, 1.0
    cw, ch = cx1 - cx0, cy1 - cy0
    area = WORKING_RES * WORKING_RES              # output area target (1024² ≈ diffusers' stock 1MP)
    s0 = (area / (W * H)) ** 0.5                  # full-frame pixel scale => crop keeps parity
    out_w = max(64, int(round(cw * s0 / 32)) * 32)
    out_h = max(64, int(round(ch * s0 / 32)) * 32)
    if out_w * out_h > area * 1.05:               # rounding guard: never exceed the area target
        k = (area / (out_w * out_h)) ** 0.5
        out_w = max(64, int(round(out_w * k / 32)) * 32)
        out_h = max(64, int(round(out_h * k / 32)) * 32)
    work = PILImage.fromarray(orig[cy0:cy1, cx0:cx1])
    # Zombie-render guard: if the client already gave up (timeout/cancel), don't spend 1-2min of
    # GPU + a 12.7GB end-of-render offload on a response nobody reads — retries stack behind the
    # lock and the summed memory of dead + live requests is what OOMs a 32Gi instance.
    if await request.is_disconnected():
        raise HTTPException(499, "client disconnected before render")
    _reset_vram_peak()
    logger.info(f"[render] pre-renovate mem={_mem_mb()}MB {_vram()} crop={frac:.2f} out={out_w}x{out_h}")
    with _qwen_lock:                              # concurrency=1 GPU + serialises the lazy first-load
        if await request.is_disconnected():       # re-check: we may have queued behind a long render
            raise HTTPException(499, "client disconnected while queued")
        ren = qwen_engine.renovate(work, tone=tone, seed=seed, height=out_h, width=out_w)
    if not _qwen_ready.is_set():
        _qwen_ready.set()                         # lazy first-load also counts as warmed
        _drop_weight_cache()
    logger.info(f"[render] post-renovate mem={_mem_mb()}MB {_vram()}")
    t_ren = time.time()

    # Paste the renovated crop back into a full-size canvas; finish() color-locks + composites
    # using ONLY mask pixels (all inside the bbox), so outside-bbox content is never consulted.
    ren_np = np.array(ren.convert("RGB"))
    interp = cv2.INTER_LANCZOS4 if (cw * ch) > (out_w * out_h) else cv2.INTER_AREA
    ren_np = cv2.resize(ren_np, (cw, ch), interpolation=interp)
    canvas = orig.copy()
    canvas[cy0:cy1, cx0:cx1] = ren_np
    # bbox: every mask pixel is inside it by construction (it IS the mask bbox, grown by mg >= 32px,
    # far more than the composite's 2px feather), so finish() emits a bit-identical image while
    # skipping two full-image LAB conversions + a full-image blur over pixels that cannot change.
    final = cf.finish(orig, canvas, mask_arr, colorHex,
                      contrast=FAMILY_CONTRAST.get(family, 1.08), chroma_retain=CHROMA_RETAIN,
                      bbox=(cy0, cx0, cy1, cx1))
    de = cf.delta_e_median(final, mask_arr, colorHex)
    buf = io.BytesIO()
    PILImage.fromarray(final).save(buf, "JPEG", quality=JPEG_QUALITY)
    logger.info(f"[render] {colorHex} {family} seg={int((t_seg-t0)*1000)}ms "
                f"render={int((t_ren-t_seg)*1000)}ms total={int((time.time()-t0)*1000)}ms dE={de:.2f} "
                f"crop={frac:.2f} out={out_w}x{out_h} mem={_mem_mb()}MB")
    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-DeltaE": f"{de:.2f}", "X-Seg-Ms": str(int((t_seg - t0) * 1000)),
        "X-Render-Ms": str(int((t_ren - t_seg) * 1000)), "X-Total-Ms": str(int((time.time() - t0) * 1000)),
        "X-Crop-Frac": f"{frac:.2f}", "X-Render-Dims": f"{out_w}x{out_h}",
        "Cache-Control": "no-store"})
