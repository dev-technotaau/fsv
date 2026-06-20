"""Local FastAPI inference server for the DINOv3-L flagship ONNX (Phase 1).

Mirrors app_dinov3.py's /detect protocol exactly so the same
index2_dinov3.html can hit it by just changing CONFIG.MODAL_ENDPOINT
to http://localhost:8001/detect.

Why this exists
---------------
Test the freshly-exported ONNX against the visualizer locally, on YOUR
GPU, without going through a Modal redeploy cycle. Useful when iterating
on the model checkpoint before committing to a cloud deploy.

Run
---
    conda activate ml
    python modal_inference/local_server.py

Open the visualizer at http://localhost:5500/index2_dinov3.html (or
similar) after editing its CONFIG.MODAL_ENDPOINT to:

    http://localhost:8001/detect

Endpoints
---------
    GET  /        -> health JSON + provider info
    POST /detect  -> multipart 'image' field, returns image/png mask
"""
from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

# Windows: onnxruntime-gpu 1.24 needs CUDA 12.x runtime DLLs (cublasLt64_12.dll,
# cudnn64_9.dll, etc.). When installed via pip (nvidia-cublas-cu12,
# nvidia-cudnn-cu12, ...) the DLLs land in site-packages/nvidia/<lib>/bin.
# Add those directories to the DLL search path BEFORE importing onnxruntime.
if sys.platform == "win32":
    import os
    import site
    _added = []
    for _site_pkg in site.getsitepackages():
        _nv_root = Path(_site_pkg) / "nvidia"
        if _nv_root.exists():
            for _sub in _nv_root.iterdir():
                _bin = _sub / "bin"
                if _bin.exists():
                    try:
                        os.add_dll_directory(str(_bin))
                        _added.append(str(_bin))
                    except (OSError, FileNotFoundError):
                        pass
    if _added:
        print(f"[startup] Added {len(_added)} NVIDIA DLL search paths "
              f"(e.g. {_added[0]})", file=sys.stderr)

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage

# ── Config ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = PROJECT_ROOT / "models" / "fence_dinov3_phase1.onnx"

INPUT_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

ALLOWED_ORIGINS = [
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8001", "http://127.0.0.1:8001",
]

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("f-stain-dinov3-local")

# ── Load ONNX once at startup ────────────────────────────────────────────
if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX file not found at {ONNX_PATH}. Run tools.export_onnx first."
    )

logger.info(f"Loading ONNX: {ONNX_PATH}  ({ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB graph)")
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
active_providers = session.get_providers()
in_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name
logger.info(f"Session ready  providers={active_providers}  in={in_name}  out={out_name}")

if "CUDAExecutionProvider" not in active_providers:
    logger.warning(
        "CUDAExecutionProvider NOT active — falling back to CPU. "
        "Inference will be ~25 sec per image instead of ~1 sec. "
        "Check that onnxruntime-gpu is installed and CUDA libs are on PATH."
    )

# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title="F-Stain DINOv3 Local Inference", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Ms", "X-Upload-Bytes", "X-Provider"],
    max_age=86400,
)


@app.get("/")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "f-stain-dinov3-inference (LOCAL)",
        "model_input_size": INPUT_SIZE,
        "channel_order": "RGB",
        "providers": active_providers,
        "onnx_path": str(ONNX_PATH),
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> Response:
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

    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    out = session.run([out_name], {in_name: arr})[0]
    probs = out[0, 0]

    mask_uint8 = (probs * 255.0).clip(0, 255).astype(np.uint8)
    mask_img = PILImage.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG", optimize=True, compress_level=6)

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(
        f"[detect] in={len(contents)/1024:.1f}KB  "
        f"out={buf.tell()/1024:.1f}KB  total={elapsed_ms}ms  "
        f"provider={active_providers[0]}"
    )

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Inference-Ms": str(elapsed_ms),
            "X-Upload-Bytes": str(len(contents)),
            "X-Provider": active_providers[0],
            "Cache-Control": "no-cache, no-store",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
