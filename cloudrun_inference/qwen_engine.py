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

log = logging.getLogger("qwen-engine")
_pipe = None
_lightning_active = False   # set True only if the Lightning LoRA actually loaded

RENOVATE_PROMPT = (
    "Restain this wooden privacy fence so it looks freshly and evenly re-stained with fresh, "
    "clean, brand-new {tone} wood. Regenerate the wood surface as newly sanded lumber with fine "
    "natural vertical grain and one uniform even tone across all boards. REMOVE all grey "
    "weathering, water stains, green algae, mildew and peeling paint. Keep the EXACT same fence — "
    "same planks, boards, gaps, rails, posts, dog-ear tops and perspective — and keep every branch, "
    "leaf, the ground and the background identical. Photorealistic, sharp, high detail."
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
    """Build the pipeline once. Safe to call repeatedly."""
    global _pipe, _lightning_active
    if _pipe is not None:
        return _pipe
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
    quant = os.environ.get("QWEN_QUANT", "prequant")

    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if quant == "4bit":
        # Runtime-quantize a NON-prequantized base model (fallback path, not used with the baked
        # ovedrive checkpoint). Quantize BOTH big components; skip img_mod to avoid stipple noise.
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

    if QWEN_LORA_REPO:
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
    if quant == "none" and os.environ.get("QWEN_FULL") == "1":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    _pipe = pipe
    log.info("[qwen] ready (lightning=%s)", _lightning_active)
    return _pipe


def renovate(image_pil, tone: str = "warm reddish cedar brown",
             steps: int | None = None, true_cfg: float | None = None, seed: int = 0):
    """Return a PIL image of the fence renovated to fresh wood."""
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
    return pipe(**kwargs).images[0]
