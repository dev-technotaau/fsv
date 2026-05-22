"""F-Stain server-side inference on Modal.

Architecture
------------
Browser uploads a fence photo via multipart POST → this Modal container
runs the ONNX model (CPU) → returns a PNG-encoded grayscale mask
(518x518, uint8 = sigmoid_prob * 255). Browser decodes the PNG, converts
back to float probabilities, and applies the existing post-processing
pipeline (soft-mask, CC cleanup, vegetation filter) locally.

Why this layout
---------------
- Heavy bit (the 135 MB ONNX model) lives on the server. Browser never
  downloads it — page is instant.
- Lightweight bit (recoloring, color picker, blend modes) stays in the
  browser, so changing color/opacity/blend is interactive (no network).
- PNG-encoded grayscale mask is ~30-100 KB (vs ~1 MB raw float array).
- Model is loaded once per container at warm-start, reused across requests.

Deploy
------
    pip install modal
    modal token new                # one-time browser auth
    cd modal_inference/
    modal deploy app.py

After deploy, Modal prints the URL. Paste `<URL>/detect` into
CONFIG.MODAL_ENDPOINT inside fence-staining-visualizer/index2.html.

Endpoints
---------
    GET  /        -> health JSON (also wakes a cold container)
    POST /detect  -> multipart 'image' field, returns image/png mask
"""

import modal

APP_NAME = "f-stain-inference"

# Browsers will POST from these origins. We use BOTH an exact-match list
# (for huggingface.co + localhost) AND a regex (for any *.static.hf.space
# subdomain we might ever deploy to — covers f-stain, n-stain, and any
# future rename without needing another redeploy of this app).
ALLOWED_ORIGINS = [
    "https://huggingface.co",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
ALLOWED_ORIGIN_REGEX = r"https://[A-Za-z0-9._-]+\.static\.hf\.space"

MODEL_FILE_REMOTE = "/model/fence_model_dinov2.onnx"
INPUT_SIZE = 518                                 # snapped to DINOv2 patch_size=14 multiple
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

app = modal.App(APP_NAME)

# Container image: small Debian base + the inference deps + the bundled ONNX.
# Bundling the model in the image keeps cold-starts fast (no extra download).
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime==1.20.0",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
    )
    .add_local_file(
        "../fence-staining-visualizer/fence_model_dinov2.onnx",
        MODEL_FILE_REMOTE,
    )
)

# Hoist runtime imports to MODULE scope (only execute inside the Modal
# container — locally during `modal deploy` they are skipped). FastAPI's
# type introspection resolves endpoint parameter types from each function's
# __globals__, so types like `UploadFile` MUST be at module scope or the
# route registration crashes with a ForwardRef error.
with app_image.imports():
    import io
    import time
    import logging
    import numpy as np
    import onnxruntime as ort
    from PIL import Image as PILImage
    from fastapi import FastAPI, UploadFile, File, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware


@app.function(
    image=app_image,
    cpu=2.0,                      # 2 CPU cores — fits DINOv2-S ORT comfortably
    memory=2048,                  # 2 GB RAM
    scaledown_window=300,         # keep container warm 5 min after last request
    min_containers=0,             # scale to zero when idle (free)
    max_containers=10,            # cap concurrent containers (cost guardrail)
)
@modal.concurrent(max_inputs=4)   # one container can handle 4 parallel requests
@modal.asgi_app()
def web():
    logger = logging.getLogger("f-stain")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load the ONNX model ONCE per container (cached in this closure across
    # subsequent requests handled by the same container).
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        MODEL_FILE_REMOTE,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
    logger.info(
        f"[startup] ONNX session ready  "
        f"inputs={[i.name for i in session.get_inputs()]}  "
        f"outputs={[o.name for o in session.get_outputs()]}"
    )

    fastapi_app = FastAPI(title="F-Stain Inference", docs_url=None, redoc_url=None)

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Inference-Ms", "X-Upload-Bytes"],
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

        # Preprocess → NCHW [1,3,518,518] ImageNet-normalized fp32.
        # Bilinear resize is what the model was trained against; do NOT
        # use nearest-neighbor here or accuracy degrades on small fences.
        img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0   # HWC fp32
        arr = arr.transpose(2, 0, 1)[None]                         # 1xCHW
        arr = (arr - mean_arr) / std_arr

        # Inference
        out = session.run(["output"], {"input": arr})[0]   # (1,1,518,518)
        probs = out[0, 0]                                    # (518, 518) sigmoid

        # Encode as PNG grayscale (compact lossless within uint8 precision —
        # 1/255 ≈ 0.4% loss, well below any threshold the browser applies).
        mask_uint8 = (probs * 255.0).clip(0, 255).astype(np.uint8)
        mask_img = PILImage.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG", optimize=True, compress_level=6)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[detect] in={len(contents)/1024:.1f}KB  "
            f"out={buf.tell()/1024:.1f}KB  total={elapsed_ms}ms"
        )

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Inference-Ms": str(elapsed_ms),
                "X-Upload-Bytes": str(len(contents)),
                "Cache-Control": "no-cache, no-store",
            },
        )

    return fastapi_app
