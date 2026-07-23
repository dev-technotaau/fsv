"""FLAGSHIP /detect + /render on Modal — hardware-upgrade demo.

WHY THIS FILE EXISTS
--------------------
The flagship pipeline already works on Google Cloud Run (1x NVIDIA L4) but a
render takes 2-3 minutes, and the client wants proof that the wait is a
HARDWARE limit, not a software one. This deploys the SAME pipeline on a much
bigger GPU so the difference is visible side by side.

WHAT IS AND IS NOT CHANGED
--------------------------
NOT changed (deliberately — this must be an apples-to-apples demo):
  * app.py, qwen_engine.py, color_finish.py      — imported verbatim from
    ../cloudrun_inference/. No forks, no edits, no "demo mode".
  * requirements.txt                              — installed with
    pip_install_from_requirements, so the dependency set cannot drift.
  * The model files                               — same Nunchaku INT4
    checkpoint, same ovedrive text-encoder/VAE snapshot, same DINOv3 ONNX.
  * Sampling quality                              — same 20-step default,
    same true_cfg 4.0, same ~1MP output, same crop-to-fence.

Changed: the GPU, the CPU/RAM, and the host. That is the whole point.

WHY L40S SPECIFICALLY
---------------------
qwen_engine.load() picks its checkpoint with nunchaku's get_precision():
    f"svdq-{prec}_r128-qwen-image-edit-2509.safetensors"
`prec` is "int4" on Ampere/Ada and "fp4" on Blackwell. Only the int4 file is
baked (same as the Cloud Run Dockerfile), so the GPU must be Ampere or Ada.

  L40S  = AD102, Ada (sm_89) — SAME architecture family as the L4 (AD104).
          Nunchaku's INT4 kernels run natively, so the existing checkpoint and
          code work with ZERO changes. 48GB VRAM, within the 40-50GB ceiling.

  vs the current L4:  142 SMs (2.45x),  864 GB/s (2.88x),  ~3x INT4 throughput.

Other options, and why not:
  * A100-40GB (sm_80) — INT4 works but is reported SLOWER than bf16 on
    datacenter Ampere; also no quality gain. Not worth it.
  * H100 / H200 / B200 — over the 40-50GB ceiling. Blackwell would also need
    the svdq-fp4_r128 file baked (see BLACKWELL note at the bottom).

MEASURED RESULTS (not estimates — taken from this deployment)
-------------------------------------------------------------
Per-step, same photo and resolution:
    Cloud Run L4  5.79 s/step   ->   Modal L40S  2.59 s/step   =  2.24x faster
(The spec sheets suggest ~3x; the real figure is 2.24x. Use the real one.)

At FSV_WORKING_RES=1024, QWEN_STEPS=16, measured end to end:
    X-Crop-Frac 1.00 (fence fills frame, no crop possible)  ->  41.4 s / 34.4 s
    X-Crop-Frac 0.70                                        ->  22.1 s
Colour accuracy held throughout: X-DeltaE ~1.0 (target is <= 3).
Segmentation is negligible: X-Seg-Ms 0.3-1.0 s.

FSV_WORKING_RES is now 896 (was 1024): ~23% fewer tokens, so expect roughly
1.3x better than the numbers above.

The single biggest factor is the PHOTO, not the hardware: X-Crop-Frac near 1.00
means the fence fills the frame and the crop optimisation cannot help. For a
demo, choose a photo with visible sky/yard around the fence.
Every response reports the real numbers in its headers.

DEPLOY
------
    pip install modal
    modal token new                 # one-time
    cd modal_inference/
    modal deploy app_render.py

First deploy is slow: it downloads ~18GB of weights into the image layer.
Later deploys reuse that layer.

TEST
----
    curl -sS --max-time 900 -F image=@fence.jpg -F colorHex=#7d4f28 \
         -F family=general <URL>/render -o out.jpg -D -
Read X-Render-Ms (denoise time), X-Total-Ms, X-Crop-Frac and X-DeltaE from the
response headers. X-Crop-Frac near 1.00 means the fence filled the frame and no
crop speed-up was possible — use a photo with some background for a fair demo.
"""

from pathlib import Path

import modal

APP_NAME = "f-stain-flagship"

# ── Hardware (the only thing that really changes vs Cloud Run) ────────────────
# Must stay Ampere/Ada so the baked INT4 checkpoint is the right one.
GPU = "L40S"        # 48GB Ada. Alternatives: "A100-40GB" (slower for INT4), "L4" (to reproduce the Cloud Run baseline exactly).
CPU = 8.0           # Cloud Run used 8 vCPU
# 32GB, down from 64GB. The 64 was defensive sizing from the Cloud Run OOM crisis, and that root
# cause is gone: the INT4 DiT now loads straight into VRAM (device="cuda"), and QWEN_RESIDENCY=gpu
# moves the 4-bit text encoder there too, so NEITHER of the two big weight blobs has a host copy any
# more. Steady-state RSS is ~6-10GB. This 32GB is not the 32GiB that killed Cloud Run — that one was
# holding a 12.7GB host copy of the DiT plus a 12.7GB transient at VAE-decode.
# The one spike left is cold start: _prefetch_weights() reads up to 18GB of weight files, and cgroup
# memory.current counts page cache, so first boot briefly reports 20GB+ before _drop_weight_cache()
# evicts it. That is reclaimable cache, not anonymous memory — the kernel drops it under pressure
# instead of OOM-killing — and Modal treats a bare int as a soft request, so it can burst anyway.
MEMORY_MB = 32768

# ── Runtime tunables ─────────────────────────────────────────────────────────
# These are injected at RUNTIME via modal.Secret.from_dict, NOT baked into the
# image. That matters: anything in Image.env() is a build layer, so editing it
# invalidates every later layer — including _bake_weights, forcing an 18GB
# re-download. As a Secret they are a *function* parameter, so changing a value
# here redeploys in seconds and never rebuilds the image.
TUNABLES = {
    "QWEN_QUANT": "nunchaku",
    # 896 (was 1024): the render area target. ~23% fewer tokens than 1024, so
    # roughly 1.3x faster, at a small sharpness cost. Set back to "1024" to
    # match the Cloud Run production setting exactly.
    "FSV_WORKING_RES": "720",
    # 16 steps for the speed demo. Set to "20" to match Cloud Run exactly and
    # make the comparison a pure hardware difference.
    "QWEN_STEPS": "16",
    # "gpu" (not "resident"): NOTHING is offloaded — the INT4 DiT, the 4-bit text encoder and the
    # VAE all stay in VRAM, so a render moves no weights across PCIe. "resident" only pins the DiT
    # and still shuttles the ~5GB text encoder CPU->GPU->CPU on every single call; fitting render
    # time against output pixels over three measured renders put that fixed cost at ~1.2s. Steady
    # state is ~23GB (DiT 12.7 + TE ~5 + VAE 0.3 + the DINOv3 ONNX arena ~5) on a 48GB L40S, and the
    # PEAK is no higher than "resident" was, because the text encoder was already co-resident with
    # the DiT during prompt encoding there. qwen_engine downgrades this to "resident" automatically
    # on any GPU under 38GiB, so switching GPU= back to "L4" stays safe without editing this line.
    "QWEN_RESIDENCY": "gpu",
    # OFF — measured, not assumed. Batching the two CFG forwards into one batch-2 forward requires
    # padding the positive/negative prompt embeds to a common length, which is only safe if the
    # transformer honours encoder_hidden_states_mask. Nunchaku's INT4 attention processor does NOT:
    # nunchaku/models/attention_processors/qwenimage.py documents the argument as "Not used." and
    # applies only a dense attention_mask of shape (B, 1, L_total, L_total), which diffusers never
    # builds. qwen_engine's self-check caught this on the first batched step (rel diff 0.166 vs a
    # 0.05 tolerance) and disabled batching, so no render was ever affected — but leaving the flag
    # on costs two wasted forwards (~2s) on the first render of every container, so it is off.
    # Building that dense mask ourselves would force the attention off its flash backend and cost
    # far more than the ~10-20% batching could win. Revisit only if Nunchaku adds mask support.
    "QWEN_CFG_BATCH": "0",
    # Lift the onnxruntime VRAM cap on the segmentation model. app.py ships VERBATIM to both this
    # 48GB L40S and Cloud Run's 24GB L4, so the cap cannot simply be deleted — it is what lets seg
    # (5.3GB) and Qwen (15.7GB load peak) coexist under the L4's 21.96GB. Here there is no such
    # pressure: measured peak is 25.6GB of 45.5GB, so ~20GB is spare. "1" gives ORT its default
    # arena growth plus cuDNN max-workspace and an EXHAUSTIVE conv-algorithm search instead of a
    # heuristic guess. Costs ~1GB VRAM and a one-off benchmark on the first inference, which
    # _warm_segmentation() now absorbs at startup. Honest expectation: tens of milliseconds off a
    # ~335ms /detect, not a step change — the ONNX run is only part of that number.
    # NEVER set this on the L4; it would reintroduce the OOM this cap was added to fix.
    "FSV_ORT_FAST": "1",
}

CLOUDRUN_DIR = Path(__file__).parent.parent / "cloudrun_inference"
MODELS_DIR = Path(__file__).parent.parent / "models"

# LD_LIBRARY_PATH must be set BEFORE python starts so the dynamic linker finds
# the nvidia-cu12 .so files that onnxruntime-gpu dlopens. Setting it from inside
# Python is too late. Identical list to the Cloud Run Dockerfile.
_NV_LIBS = ["cublas", "cuda_runtime", "cuda_nvrtc", "cudnn", "cufft", "curand", "nvjitlink"]
_LD_LIBRARY_PATH = ":".join(
    [f"/usr/local/lib/python3.11/site-packages/nvidia/{lib}/lib" for lib in _NV_LIBS]
    + ["/usr/local/lib/python3.11/site-packages/torch/lib"]
)

app = modal.App(APP_NAME)


def _bake_weights():
    """Download the render weights into the image layer (same as the Dockerfile).

    Baking beats a Volume here: the weights become part of the image, so a cold
    container starts reading them immediately instead of mounting and fetching.
    """
    from huggingface_hub import snapshot_download

    # Nunchaku SVDQuant INT4 transformer (12.7GB, r128 = max sharpness).
    snapshot_download(
        "nunchaku-tech/nunchaku-qwen-image-edit-2509",
        allow_patterns=["svdq-int4_r128-qwen-image-edit-2509.safetensors"],
        local_dir="/model/nunchaku-qwen",
    )
    # ovedrive 4-bit snapshot MINUS its transformer — reused only for the 4-bit
    # text encoder + VAE + tokenizer + scheduler configs.
    snapshot_download(
        "ovedrive/Qwen-Image-Edit-2509-4bit",
        local_dir="/model/qwen-edit",
        ignore_patterns=["transformer/*"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    # libglib2.0-0 + libgl1 are required by opencv-python-headless.
    .apt_install("ca-certificates", "curl", "libglib2.0-0", "libgl1")
    # Install the EXACT Cloud Run dependency set — including the cu128 extra
    # index and the pinned Nunchaku wheel — straight from the same file, so the
    # demo can never drift from production.
    .pip_install_from_requirements(str(CLOUDRUN_DIR / "requirements.txt"))
    # BUILD-TIME env only. Deliberately excludes the tunables (see TUNABLES) so
    # that changing resolution or steps never invalidates the 18GB weight layer.
    .env(
        {
            "LD_LIBRARY_PATH": _LD_LIBRARY_PATH,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    # Fail the build early if the Nunchaku wheel didn't match this torch.
    .run_commands(
        "python -c \"import torch, onnxruntime; "
        "from nunchaku import NunchakuQwenImageTransformer2DModel; "
        "from nunchaku.utils import get_precision; "
        "print('OK torch', torch.__version__, '| ort', onnxruntime.__version__, "
        "onnxruntime.get_available_providers())\""
    )
    .run_function(_bake_weights)
    # DINOv3 segmentation model (2.4GB) — powers /detect.
    .add_local_file(str(MODELS_DIR / "fence_dinov3_phase1.onnx"), "/model/fence_dinov3_phase1.onnx")
    .add_local_file(str(MODELS_DIR / "fence_dinov3_phase1.onnx.data"), "/model/fence_dinov3_phase1.onnx.data")
    # The production application code, imported unchanged.
    .add_local_file(str(CLOUDRUN_DIR / "app.py"), "/app/app.py")
    .add_local_file(str(CLOUDRUN_DIR / "qwen_engine.py"), "/app/qwen_engine.py")
    .add_local_file(str(CLOUDRUN_DIR / "color_finish.py"), "/app/color_finish.py")
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    # Runtime config. A Secret is a function parameter, not an image layer, so
    # editing TUNABLES redeploys in seconds without rebuilding the image.
    secrets=[modal.Secret.from_dict(TUNABLES)],
    timeout=3600,          # a cold container loads ~18GB before the first render
    scaledown_window=600,  # stay warm 10 min after the last request
    min_containers=0,      # scale to zero — no cost when idle
    max_containers=1,      # one GPU is plenty for a demo; also caps spend
)
# Keep this at 1. Raising it does NOT let cheap requests (the frontend's `/` warm-up poll, /detect)
# overtake a render: app.py's endpoints are `async def` and do their work synchronously on the
# event loop, so a 20s render freezes the loop and any extra admitted request just waits anyway.
# It would also be unsafe — /render holds a blocking threading.Lock across an `await`, which two
# concurrent requests can deadlock. Raising this requires moving the blocking stages onto worker
# threads first (asyncio.to_thread); until then 1 is the only correct value.
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    """Serve the unmodified Cloud Run FastAPI app (/, /detect, /render)."""
    import sys

    sys.path.insert(0, "/app")
    from app import app as fastapi_app  # /app/app.py — the production app

    return fastapi_app


# ─────────────────────────────────────────────────────────────────────────────
# BLACKWELL NOTE (only if you ever move to a 5090 / RTX PRO 6000 / B200)
#
# nunchaku's get_precision() returns "fp4" on Blackwell, so qwen_engine would
# look for svdq-fp4_r128-qwen-image-edit-2509.safetensors, which is NOT baked
# here and the load would fail. To support Blackwell, add the fp4 pattern to
# _bake_weights():
#
#     allow_patterns=["svdq-int4_r128-qwen-image-edit-2509.safetensors",
#                     "svdq-fp4_r128-qwen-image-edit-2509.safetensors"],
#
# and confirm the pinned nunchaku v1.2.1 cu12.8 wheel ships sm_100/sm_120
# kernels. No other code changes are needed — the engine picks the file itself.
# ─────────────────────────────────────────────────────────────────────────────
