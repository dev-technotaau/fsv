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
    # NOTE: QWEN_STEPS / QWEN_CFG / NUNCHAKU_TRANSFORMER are NOT set here — they come from the
    # SAMPLING profile below, so the step count can never disagree with the checkpoint.
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
    # heuristic guess.
    #
    # TURNED BACK OFF after measuring it. EXHAUSTIVE benchmarks every conv algorithm on the first
    # inference, and _warm_segmentation() logged "segmentation warmed in 15.9s" against ~0.3-1s on
    # the heuristic path. Paying ~15s on EVERY cold start to save tens of ms per /detect is a bad
    # trade on a scale-to-zero container — the ONNX run is only a fraction of the ~335ms /detect,
    # and this service cold-starts far more often than it serves a latency-critical detect.
    # Worth revisiting ONLY with min_containers>=1, where the startup cost is paid once.
    # NEVER set this on the L4 either; there it would reintroduce the OOM the cap exists to fix.
    "FSV_ORT_FAST": "0",
}

# ── Sampling profile — the single biggest speed lever, and the only one that touches QUALITY ──
# Change this ONE word and redeploy. quality and lightning8 are a ~30s redeploy (both checkpoints
# are baked); lightning4 needs its checkpoint uncommented in BAKED_TRANSFORMERS first.
#
#   profile      checkpoint            steps  CFG   forwards/image   measured/est. denoise
#   ---------    ------------------    -----  ----  --------------   ---------------------
#   quality      base                   16    4.0        32          6.6s / 12.1s  (MEASURED)
#   lightning8   lightning-8steps       8     1.0         8          ~2.0-2.4s / ~3.5-4.0s (est.)
#   lightning4   lightning-4steps       4     1.0         4          ~1.2-1.5s / ~2.0-2.5s (NOT BAKED)
#
# (the two numbers are a crop-frac 0.39 photo and a crop-frac 0.72 photo)
#
# Forwards is what actually costs: true CFG > 1.0 runs the transformer TWICE per step, the
# distilled checkpoints are guidance-distilled so they run it once. 16x2 -> 8x1 is 4x fewer.
#
# QUALITY IS THE OPEN QUESTION, and X-DeltaE will NOT answer it — the LAB colour-lock pins dE near
# 1.0 whatever the sampler does. Judge on plank-edge crispness and wood grain at 100% zoom, on the
# same photo and seed. Step distillation trades high-frequency detail, and grain is exactly that,
# so expect lightning4 to be visibly softer. lightning8 is the one with a real chance of being
# indistinguishable. An earlier in-house verdict that "Lightning is soft" was measuring a
# quantization mismatch (a bf16 LoRA on an INT4 transformer, which nunchaku cannot apply at all)
# and should not be treated as evidence about the distillation itself.
SAMPLING = "lightning8"

_LIGHTNING_DIR = "/model/nunchaku-qwen/lightning-251115"

# Transformer checkpoints baked into the image. THIS LIST IS THE COLD-START COST: each file is
# 12.65GB and has to be pulled before a container can serve, so every entry slows the first request
# after a scale-to-zero. Baking more than one is what makes switching SAMPLING a ~30s redeploy
# instead of a ~13GB rebuild — that convenience is not free.
#
# The 4-step checkpoint is deliberately NOT baked. It is used by exactly one profile, and 4-step
# distillation is expected to be visibly soft on wood grain, which is the detail this product is
# judged on. To try it: uncomment the line, redeploy (~13GB, slow, once), then set SAMPLING.
BAKED_TRANSFORMERS = [
    "svdq-int4_r128-qwen-image-edit-2509.safetensors",
    "lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
    # "lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
]

_SAMPLING_PROFILES = {
    # Base checkpoint: NUNCHAKU_TRANSFORMER unset, so qwen_engine builds the default path.
    "quality": {"QWEN_STEPS": "16", "QWEN_CFG": "4.0"},
    # Distilled: steps and CFG are deliberately NOT set. qwen_engine reads "8steps"/"4steps" out of
    # the filename and derives steps + CFG 1.0 + the shift=3 scheduler from that, so the sampler
    # cannot silently disagree with the weights.
    "lightning8": {"NUNCHAKU_TRANSFORMER":
                   f"{_LIGHTNING_DIR}/svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors"},
    "lightning4": {"NUNCHAKU_TRANSFORMER":
                   f"{_LIGHTNING_DIR}/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors"},
}
if SAMPLING not in _SAMPLING_PROFILES:
    raise ValueError(f"SAMPLING must be one of {sorted(_SAMPLING_PROFILES)}, got {SAMPLING!r}")

# Fail at DEPLOY time, not at container start. Selecting a profile whose checkpoint was never baked
# would otherwise surface as a FileNotFoundError inside the background warm thread, minutes later,
# in the logs of a container that appears to boot fine — exactly the "it won't start" symptom that
# is hardest to diagnose.
_selected = _SAMPLING_PROFILES[SAMPLING].get("NUNCHAKU_TRANSFORMER")
if _selected and not any(_selected.endswith(p.rsplit("/", 1)[-1]) for p in BAKED_TRANSFORMERS):
    raise ValueError(
        f"SAMPLING={SAMPLING!r} needs {_selected!r}, which is NOT in BAKED_TRANSFORMERS.\n"
        f"Uncomment it there and redeploy (~13GB rebuild, once), or choose a profile whose "
        f"checkpoint is baked: {[p.rsplit('/', 1)[-1] for p in BAKED_TRANSFORMERS]}")

TUNABLES.update(_SAMPLING_PROFILES[SAMPLING])

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

    # Nunchaku SVDQuant INT4 transformers, r128 = max sharpness.
    #   base                 12.7GB  20/16-step, true CFG 4.0  -> 32 forwards
    #   lightning 8-step     12.7GB  guidance-distilled, CFG 1.0 ->  8 forwards
    #   lightning 4-step     12.7GB  ditto                       ->  4 forwards
    # All three are baked so SAMPLING can be switched with a ~30s redeploy and no rebuild. They are
    # ~38GB of image layer, downloaded once. IMPORTANT: the "lightning-251115" folder is fused with
    # the Qwen-Image-EDIT-2509 Lightning LoRA; the older top-level "lightningv2.0" files are fused
    # with the LoRA for the base text-to-image Qwen-Image and are the wrong distillation for an
    # editing pipeline. Only the 251115 ones are fetched here, deliberately.
    snapshot_download(
        "nunchaku-tech/nunchaku-qwen-image-edit-2509",
        allow_patterns=BAKED_TRANSFORMERS,
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
