"""Qwen-Image-Edit-2509 renovation engine (GPU).

Uses the PRE-QUANTIZED NF4 checkpoint `ovedrive/Qwen-Image-Edit-2509-4bit` (~12-15 GB, quality
layers kept full-precision so no stipple noise). It's BAKED into the image (like the DINOv3 ONNX),
so it loads from local disk in ~1-2 min — no GCS bucket, no runtime quantization, no OOM warmup.
Renovates the fence (fresh wood); exact colour + pristine background are handled by color_finish.py.

Env: QWEN_MODEL (baked path), QWEN_QUANT (prequant|4bit|none), QWEN_LIGHTNING_LORA (''=off),
     QWEN_LIGHTNING_WEIGHT, QWEN_STEPS, QWEN_CFG.
"""
from __future__ import annotations
import os
import logging
import threading

log = logging.getLogger("qwen-engine")
_pipe = None
_load_lock = threading.Lock()   # load() is hit by the startup warm thread AND request threads
_lightning_active = False   # set True only if the Lightning LoRA actually loaded

RENOVATE_PROMPT = (
    "Restain this wooden privacy fence so it looks freshly and evenly re-stained with fresh, "
    "clean, brand-new {tone} wood. Regenerate the wood surface as newly sanded lumber with fine, "
    "richly detailed natural vertical wood grain and visible timber texture, one uniform even tone "
    "across all boards, with natural depth and dimension. REMOVE all grey weathering, water stains, "
    "green algae, mildew and peeling paint. Preserve realistic soft natural daylight, gentle shadows "
    "in the plank gaps and along the rails and posts, and lifelike depth. Keep the EXACT same fence — "
    "same planks, boards, gaps, rails, posts, dog-ear tops and perspective — and keep every branch, "
    "leaf, the ground and the background identical. Photorealistic, sharp crisp focus, high detail, "
    "professional photograph."
)
NEGATIVE = ("weathered, faded, grey, peeling, flaking, cracked, old, worn, dirty, mildew, algae, "
            "water stains, blotchy, patchy, uneven, different fence, missing planks, extra planks, "
            "cartoon, painting, blurry, low quality")

# Pre-quantized checkpoint + Lightning LoRA, both baked into the image (see Dockerfile).
QWEN_MODEL_DEFAULT = "/model/qwen-edit"
QWEN_LORA_REPO = os.environ.get("QWEN_LIGHTNING_LORA", "/model/qwen-lightning").strip()
QWEN_LORA_WEIGHT = os.environ.get(
    "QWEN_LIGHTNING_WEIGHT",
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")


def _default_steps() -> int:
    v = os.environ.get("QWEN_STEPS")
    return int(v) if v else (8 if _lightning_active else 20)   # 8 w/ Lightning, else 20-step base


def _default_cfg() -> float:
    v = os.environ.get("QWEN_CFG")
    return float(v) if v else (1.0 if _lightning_active else 4.0)


def load():
    """Build the pipeline once. Safe to call repeatedly and from multiple threads."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _load_lock:           # double-checked: warm thread + first request race here
        if _pipe is not None:
            return _pipe
        return _load_locked()


def _load_locked():
    global _pipe, _lightning_active
    import torch
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        log.warning("[qwen] transformers %s detected; diffusers 0.39 Qwen path is validated against "
                    "4.x — a 5.x forward-path break causes a first-inference 500. Pin <5.",
                    transformers.__version__)
    try:
        from diffusers import QwenImageEditPlusPipeline as QPipe   # 2509 = 'Plus' (multi-image)
    except Exception:
        from diffusers import QwenImageEditPipeline as QPipe
    model = os.environ.get("QWEN_MODEL", QWEN_MODEL_DEFAULT)
    quant = os.environ.get("QWEN_QUANT", "nunchaku")

    if quant == "nunchaku":
        # FAST path: Nunchaku SVDQuant INT4 transformer (fused INT4 tensor-core GEMMs — ~3x vs bnb
        # NF4, at FULL 20-step sharpness). Load the INT4 DiT, then build the pipe reusing the baked
        # ovedrive snapshot ONLY for its 4-bit text_encoder + VAE + tokenizer + configs.
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from nunchaku.utils import get_precision
        prec = get_precision()   # 'int4' on L4/Ada (NOT fp4, which is Blackwell-only)
        tf = os.environ.get("NUNCHAKU_TRANSFORMER",
                            f"/model/nunchaku-qwen/svdq-{prec}_r128-qwen-image-edit-2509.safetensors")
        log.info("[qwen] loading Nunchaku %s transformer %s", prec, tf)
        # CRITICAL host-RAM fix: load the 12.7GB INT4 DiT DIRECTLY to the GPU (device='cuda').
        # from_pretrained then does to_empty(device='cuda') + load_state_dict, so the weights
        # materialize in VRAM and the transient host state-dict is freed — NO persistent 12.7GB
        # host copy. The default (device='cpu') load left that copy resident, so the 20GB baseline +
        # the pipeline's end-of-render 12.7GB transformer move blew past Cloud Run's 32Gi host cap
        # (signal-9 ~20s after denoise, rev 00012-00014 all). Loading to GPU drops the baseline to
        # ~8GB so even a worst-case burst stays ~20GB. Fall back to CPU load only if the kwarg is
        # unsupported (older nunchaku) — that path OOMs, but keeps the service importable.
        try:
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(tf, device="cuda")
            log.info("[qwen] DiT materialized directly in VRAM (no host copy)")
        except Exception as e:
            log.warning("[qwen] device='cuda' load failed (%s); falling back to host load (OOM risk)", e)
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(tf)
        pipe = QPipe.from_pretrained(model, transformer=transformer, torch_dtype=torch.bfloat16)
    else:
        kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if quant == "4bit":
            # Runtime-quantize a NON-prequantized base model (fallback). Quantize BOTH big components;
            # skip img_mod to avoid stipple noise.
            from diffusers import PipelineQuantizationConfig, BitsAndBytesConfig as DiffusersBnb
            from transformers import BitsAndBytesConfig as TransformersBnb
            kw["quantization_config"] = PipelineQuantizationConfig(quant_mapping={
                "transformer": DiffusersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                            llm_int8_skip_modules=["transformer_blocks.0.img_mod"]),
                "text_encoder": TransformersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.bfloat16)})
        # quant == "prequant": checkpoint is already NF4 on disk -> no quant config, load as-is.
        # quant == "none": bf16 (needs >24GB).
        log.info("[qwen] loading %s (quant=%s)", model, quant)
        pipe = QPipe.from_pretrained(model, **kw)

    # Lightning is the FAST 8-step path but softer than full base sampling. OFF by default so the
    # engine uses sharp, high-quality 20-step base (what produced the loved output). Opt in with
    # QWEN_LIGHTNING=1 only when speed matters more than sharpness.
    if QWEN_LORA_REPO and os.environ.get("QWEN_LIGHTNING", "0") == "1":
        try:
            import math
            from diffusers import FlowMatchEulerDiscreteScheduler
            # Live adapter, NOT fused (fusing into 4-bit is lossy — PEFT #2321).
            pipe.load_lora_weights(QWEN_LORA_REPO, weight_name=QWEN_LORA_WEIGHT, adapter_name="lightning")
            pipe.set_adapters(["lightning"], adapter_weights=[1.0])
            _lightning_active = True
            # Lightning = shift=3 distillation -> base_shift/max_shift = log(3); scalar 'shift' stays 1.0.
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
                "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
                "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
                "time_shift_type": "exponential", "use_beta_sigmas": False,
                "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
            log.info("[qwen] Lightning adapter active (no fuse): %s/%s steps=%s cfg=%s",
                     QWEN_LORA_REPO, QWEN_LORA_WEIGHT, _default_steps(), _default_cfg())
        except Exception as e:  # LoRA optional — fall back to more steps (auto via _lightning_active)
            log.warning("[qwen] Lightning LoRA not applied (%s); falling back to %s-step base",
                        e, _default_steps())

    # Placement: 4-bit (prequant/4bit) uses enable_model_cpu_offload() — pipe.to('cuda') raises on a
    # bnb model. Only bf16 (quant=none) may use .to('cuda').
    #
    # QWEN_RESIDENCY (nunchaku only) controls how the 12.7GB INT4 DiT is placed:
    #   resident (DEFAULT, REQUIRED on Cloud Run's 32Gi host) — pipe._exclude_from_cpu_offload keeps
    #            the DiT on the GPU for the WHOLE render; only TE + VAE offload. This is the only mode
    #            that survives 32Gi: with plain offload, nunchaku mmaps the 12.7GB weights (those file
    #            pages stay host-resident, unreclaimable) AND at VAE-decode the DiT moves GPU->CPU,
    #            allocating ANOTHER 12.7GB anon => ~34GB transient => cgroup SIGKILL ~20s after
    #            denoise finishes (verified rev 00012/00013, twice each). Resident removes that burst:
    #            the DiT never leaves the GPU. VRAM peak is TE-encode (DiT 12.7 + TE ~5 + seg ~2.7 =
    #            ~20-23GB) — fits the 24GB L4 (every crash so far was HOST OOM, never CUDA OOM).
    #   offload  — nunchaku's stock >18GB-GPU config; correct on a normal 64GB host but OOMs Cloud
    #            Run's 32Gi (see above). Only use on a >40Gi instance.
    #   pinned   — DO NOT USE at 32Gi: pins a SECOND 12.7GB copy in locked pages -> load-time OOM
    #            (rev 00011). Needs >40Gi.
    residency = os.environ.get("QWEN_RESIDENCY", "resident")
    if quant == "none" and os.environ.get("QWEN_FULL") == "1":
        pipe.to("cuda")
    elif quant == "nunchaku" and residency == "resident":
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_model_cpu_offload()
        log.info("[qwen] residency=resident (DiT stays on GPU; only TE/VAE offload — no host burst)")
    else:
        pipe.enable_model_cpu_offload()
        if quant == "nunchaku" and residency == "pinned":
            try:
                t0 = __import__("time").time()
                n = 0
                for p in pipe.transformer.parameters():
                    if p.data.device.type == "cpu" and not p.data.is_pinned():
                        p.data = p.data.pin_memory(); n += 1
                for b in pipe.transformer.buffers():
                    if b.data.device.type == "cpu" and not b.data.is_pinned():
                        b.data = b.data.pin_memory(); n += 1
                log.info("[qwen] residency=pinned (%d tensors pinned in %.1fs — fast DMA shuttle)",
                         n, __import__("time").time() - t0)
            except Exception as e:   # pinning is an optimization only — never fatal
                log.warning("[qwen] pin_memory failed (%s); falling back to pageable offload", e)
    _pipe = pipe
    log.info("[qwen] ready (lightning=%s)", _lightning_active)
    return _pipe


_DEFAULT_VAE_AREA = 1024 * 1024   # diffusers' hardcoded output/condition area when no dims passed


def renovate(image_pil, tone: str = "warm reddish cedar brown",
             steps: int | None = None, true_cfg: float | None = None, seed: int = 0,
             height: int | None = None, width: int | None = None):
    """Return a PIL image of the fence renovated to fresh wood.

    height/width (multiples of 32): render at EXACT dims — used by the crop-to-fence path so a
    bbox crop costs proportionally fewer tokens. When set, the Plus pipeline's module-level
    VAE_IMAGE_SIZE is patched to the same area so the condition-image latents get the SAME grid
    as the output (mismatched grids halve the crop speedup and worsen pixel-shift). Caller must
    hold the render lock (app._qwen_lock) — the patch is a module global.
    """
    import torch
    pipe = load()
    is_plus = pipe.__class__.__name__ == "QwenImageEditPlusPipeline"
    g = torch.Generator(device="cpu").manual_seed(seed)
    im = [image_pil] if is_plus else image_pil            # list form is correct for the Plus/2509 pipe
    cfg = true_cfg if true_cfg is not None else _default_cfg()
    kwargs = dict(image=im, prompt=RENOVATE_PROMPT.format(tone=tone),
                  num_inference_steps=steps or _default_steps(),
                  true_cfg_scale=cfg, generator=g)        # true_cfg_scale is the CFG knob for Qwen-Edit
    if cfg > 1.0:                                          # negative_prompt only consumed when CFG on
        kwargs["negative_prompt"] = NEGATIVE

    # Condition-grid patch: keep condition dims == output dims (see docstring).
    qep = None
    if is_plus:
        try:
            from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as qep
        except Exception:
            qep = None
    if height and width:
        kwargs["height"], kwargs["width"] = height, width
        if qep is not None and hasattr(qep, "VAE_IMAGE_SIZE"):
            qep.VAE_IMAGE_SIZE = int(height) * int(width)
    elif qep is not None and hasattr(qep, "VAE_IMAGE_SIZE"):
        qep.VAE_IMAGE_SIZE = _DEFAULT_VAE_AREA            # restore stock behavior for full-frame

    # Nunchaku's transformer calls torch.cuda.empty_cache() at the END OF EVERY FORWARD (v1.2.1
    # transformer_qwenimage.py:558) — 2x/step. Under expandable_segments each call is a synchronizing
    # allocator teardown + page re-map on the next forward: 10-40s of pure overhead per render.
    # No-op it for the duration of the pipe call; restore + flush once after.
    _ec = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: None
    try:
        out = pipe(**kwargs).images[0]
    finally:
        torch.cuda.empty_cache = _ec
        _ec()
    return out
