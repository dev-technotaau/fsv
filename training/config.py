"""training/config.py — YAML-driven configuration for the training pipeline.

All hyperparameters are gathered into a single TrainingConfig dataclass.
Configs live in configs/*.yaml. Override any field via CLI: `--key value`.

Example:
    python -m training.train --config configs/phase1.yaml \
        --batch-size 8 --backbone-size large
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════
# Sub-configs
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    # Backbone
    backbone_name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    # Options:
    #   DINOv3 (recommended — gated; run `huggingface-cli login` first):
    #     facebook/dinov3-vits16-pretrain-lvd1689m       (~22M)
    #     facebook/dinov3-vitb16-pretrain-lvd1689m       (~86M)
    #     facebook/dinov3-vitl16-pretrain-lvd1689m       (~300M)
    #     facebook/dinov3-vith16plus-pretrain-lvd1689m   (~840M, default)
    #     facebook/dinov3-vit7b16-pretrain-lvd1689m      (~6.7B)
    #   DINOv2 (open access, fallback):
    #     facebook/dinov2-{small,base,large,giant}
    backbone_freeze_first_n_layers: int = 0
    # 0 = train all layers; useful 12-18 for smaller datasets.
    backbone_drop_path_rate: float = 0.1

    # Multi-block aggregation (DINOv3 paper recommendation for dense tasks)
    multi_block_n: int = 4                 # fuse last N transformer blocks
    aggregation_type: str = "weighted_sum" # 'sum' | 'concat_proj' | 'weighted_sum'

    # Pixel decoder: 'fpn' (lightweight) or 'msdeform' (real M2F via HF impl)
    pixel_decoder_type: str = "msdeform"
    pixel_decoder_layers: int = 6           # M2F default = 6
    pixel_decoder_heads: int = 8
    pixel_decoder_ffn_dim: int = 1024

    # Decoder
    decoder_type: str = "mask2former"
    # Options: 'mask2former' (transformer queries + cross-attn) or 'upernet'
    decoder_dim: int = 256
    decoder_num_queries: int = 16          # mask2former only
    decoder_num_layers: int = 6            # transformer decoder layers
    decoder_dropout: float = 0.0
    use_masked_attention: bool = True       # M2F's defining feature
    use_global_tokens: bool = True          # CLS + register tokens as memory
    mask_attention_threshold: float = 0.5   # sigmoid threshold for masked-attn

    # Output
    num_classes: int = 1                   # binary fence vs background
    output_logit: bool = True              # raw logits (use BCEWithLogitsLoss)

    # Refinement head (optional, learnable mask cleanup)
    use_refinement_head: bool = True
    refinement_channels: int = 32
    refinement_num_blocks: int = 3
    # A. Inject pixel decoder's high-res features into the refinement head
    #    (instead of running it on raw RGB only). The features are projected
    #    to refinement_decoder_feature_channels, detached, upsampled.
    refinement_use_decoder_features: bool = True
    refinement_decoder_feature_channels: int = 64
    # B. Edge prediction head — supervised by Sobel-derived GT edges.
    #    Forces refinement features to be edge-aware.
    refinement_use_edge_head: bool = True
    # C. Iterative refinement: run the head N times, each pass feeding the
    #    previous output as the "coarse mask" input. Each iteration detached
    #    from the previous (bounded gradient depth).
    refinement_iterations: int = 2

    # Depth / position cues for the refinement head:
    #   - y_coord: free, normalized vertical position (0=top, 1=bottom).
    #              Captures sky/ground correlation. Always-on by default.
    #   - depth:   frozen DPT/MiDaS teacher predicts per-pixel depth.
    #              Distinguishes near-vs-far same-material objects (e.g.
    #              foreground garden bed vs background fence). +25-30%
    #              training time, +120 MB model, +80-120 ms inference.
    #              Opt-in.
    refinement_use_y_coord: bool = True
    refinement_use_depth: bool = False
    depth_model_name: str = "Intel/dpt-hybrid-midas"

    # Memory / speed knobs
    gradient_checkpointing: bool = False    # halves backbone activation mem
    torch_compile: bool = False             # torch.compile(model) — 10-30% speedup
    torch_compile_mode: str = "default"     # 'default' | 'reduce-overhead' | 'max-autotune'


@dataclass
class LossConfig:
    # Combined loss = sum(weight_i * loss_i). Set weight=0 to disable.
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.5
    # Lovasz: direct IoU surrogate. ON by default — directly optimizes the
    # primary metric. Tiny extra cost, +0.5-1% IoU lift.
    lovasz_weight: float = 0.5
    # Tversky: recall-friendly Dice variant. Penalizes false negatives more
    # than false positives so the model doesn't miss fence pixels at the
    # boundaries (critical for the staining use case — gaps show through).
    tversky_weight: float = 0.5
    tversky_alpha: float = 0.7              # FN penalty
    tversky_beta: float = 0.3               # FP penalty
    connectivity_weight: float = 0.0        # experimental — leave 0 to start

    # BCE specifics
    use_pos_weight: bool = True             # auto-computed at startup if True
    pos_weight: Optional[float] = None      # override; None = auto
    # Focal-loss exponent: 0 = standard BCE; 2 = Lin et al. 2017 standard.
    # Down-weights easy pixels; concentrates loss on hard look-alikes
    # (cedar fence vs other wood, fence vs grass/ground/sky boundary).
    focal_gamma: float = 2.0
    # OHEM (Online Hard Example Mining): keep top-K% highest-loss BCE pixels
    # per image. 0 = disabled; 0.25 = top 25%. Stacks with focal_gamma.
    # Independent of PointRend — they target different pixel populations.
    ohem_top_k_ratio: float = 0.0

    # Boundary loss
    boundary_kernel_size: int = 5

    # Dice specifics
    dice_smooth: float = 1.0

    # Per-sample loss weighting from dataset's review_source field
    weight_by_review_source: bool = True

    # Deep supervision (Mask2Former-style): also apply BCE+Dice to each
    # decoder layer's mask prediction. Total deep-sup contribution is
    # `deep_supervision_weight` (split evenly across the aux layers).
    # 0.0 = disabled (only final layer supervised, the original behavior).
    # 1.0 = aux losses summed at full weight; standard M2F uses ~1.0.
    deep_supervision_weight: float = 1.0

    # PointRend importance sampling — replaces dense BCE+Dice with sampling
    # at N informative (boundary-heavy) points per image. Used by real
    # Mask2Former training. Faster (less compute on easy interior pixels)
    # AND better boundaries.
    use_pointrend: bool = True
    pointrend_n_points: int = 12544        # 112^2 — M2F default
    pointrend_oversample_ratio: float = 3.0
    pointrend_importance_ratio: float = 0.75

    # Edge-aware auxiliary loss (refinement head's edge prediction).
    # 0 = disabled.
    edge_loss_weight: float = 0.5
    edge_loss_dilate: int = 3              # widen GT edge band by this many px
    edge_loss_pos_weight: float = 8.0      # edges are sparse (~1-3% of pixels)

    # Iterative refinement: weight on aux losses for non-final iterations.
    # Only applies when model.refinement_iterations > 1.
    refinement_iter_aux_weight: float = 0.5

    # EMA self-distillation: consistency loss between live model output and
    # the EMA-teacher output on the same input. Mean-Teacher style. The EMA
    # forward runs in no_grad — we only push the live model toward the
    # teacher's prediction. ~0.5-1% IoU lift on hard-case generalization.
    # 0 = disabled; 0.5 = typical. Only active if `train.use_ema=True`.
    ema_distill_weight: float = 0.0


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    base_lr: float = 1e-4                  # head LR
    backbone_lr: float = 1e-5              # backbone LR (lower)
    backbone_lr_decay: float = 0.9         # layer-wise LR decay multiplier
    weight_decay: float = 0.05
    momentum: float = 0.9                  # for SGD; ignored for AdamW
    betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler
    warmup_epochs: int = 2
    warmup_lr: float = 1e-7
    lr_min: float = 1e-7
    schedule: str = "cosine"               # 'cosine' | 'constant' | 'linear'

    # Gradient
    grad_clip_norm: float = 1.0
    grad_accumulation_steps: int = 1

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bf16"                # 'bf16' (Ampere+) or 'fp16'


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 8                    # per-device
    val_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Validation
    val_every_n_epochs: int = 1
    val_inference_size: int = 0            # 0 = match train size

    # Multi-scale training
    multi_scale_train: bool = False        # randomly resize within range during train
    multi_scale_min_factor: float = 0.75
    multi_scale_max_factor: float = 1.25

    # CutMix augmentation (pair-wise in the batch). 0 = disabled, 0.3-0.5 = typical.
    # Cuts a random rectangle from sample B and pastes it onto sample A (with
    # mask). Strong regularizer, helps generalization on hard images.
    cutmix_p: float = 0.3
    cutmix_alpha: float = 1.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 1000

    # Seed
    seed: int = 42
    deterministic: bool = False            # True = slower but bit-exact

    # Phase 2 init from phase 1 best ckpt (override at CLI)
    init_from: Optional[str] = None

    # Test-time augmentation (TTA) at val/test
    use_tta: bool = False
    tta_scales: tuple[float, ...] = (1.0,)
    tta_flip: bool = False
    # Force TTA ON for the FINAL post-training test-set evaluation
    # (regardless of `use_tta`). Per-epoch val still respects `use_tta`
    # (TTA at val is slow and we don't need it during training).
    tta_at_final_test: bool = True

    # Robustness / safety knobs
    skip_step_on_nonfinite_loss: bool = True   # silently drop NaN/Inf batches
    log_grad_norm: bool = True                 # log clipped grad-norm to TB

    # Early stopping (set patience > 0 to enable)
    early_stop_patience: int = 0               # epochs of no-improvement to wait
    early_stop_min_delta: float = 1e-4         # minimum metric delta to count

    # End-of-training test-set evaluation (uses best.pt)
    run_test_eval_on_finish: bool = True


@dataclass
class DataConfig:
    splits_dir: str = "dataset/splits"
    image_size: int = 512                  # phase1=512, phase2=1024
    train_split: str = "train"             # phase2 sets this to 'train_hq'
    val_split: str = "val"                 # phase2 sets this to 'val_hq'
    test_split: str = "test"               # phase2 sets this to 'test_hq'


@dataclass
class PostProcessConfig:
    """Inference-time post-processing. Off during training (val metrics still
    measure raw model output; post-processing is applied at inference scripts
    via --post-process flag, OR auto-applied at the final test eval if
    `enabled` is true).

    Defaults are now ON with fence-tuned parameters:
      - DenseCRF bilateral params tightened for fence's narrow shape
        (smaller spatial radius, slightly stronger color guidance)
      - Guided filter ON for boundary smoothing
      - Morphology ON for tiny-speck cleanup
    """
    enabled: bool = True
    use_morphology: bool = True
    morphology_kernel: int = 3
    use_guided_filter: bool = True
    guided_filter_radius: int = 4
    guided_filter_eps: float = 1e-4
    # DenseCRF: dramatically sharpens fence-vs-grass / fence-vs-sky edges via
    # bilateral image-color guidance. Enabled by default; requires pydensecrf.
    use_dense_crf: bool = True
    crf_iterations: int = 5
    crf_gauss_sxy: int = 3
    crf_gauss_compat: int = 3
    # Tighter spatial radius (fence boundaries are usually within ~50 px,
    # not 80) and slightly stronger color guidance for fence vs grass/sky.
    crf_bilateral_sxy: int = 50
    crf_bilateral_srgb: int = 10
    crf_bilateral_compat: int = 12
    # Connected-component cleanup (fence-domain post-process):
    #   - drop tiny false-positive blobs (in trees / grass)
    #   - optionally fill tiny background holes inside the fence mask
    #   - optionally keep only the K largest blobs
    use_cc_cleanup: bool = True
    cc_min_blob_area: int = 200             # drop blobs smaller than N pixels
    cc_fill_holes_smaller_than: int = 0     # 0 = preserve picket gaps; raise to fill
    cc_keep_top_k_blobs: int = 0            # 0 = unlimited; e.g. 5 = up to 5 fence regions


@dataclass
class LogConfig:
    log_dir: str = "outputs/training_v2"
    run_name: str = "phase1"               # subdirectory name
    log_every_n_steps: int = 50
    use_tensorboard: bool = True
    save_sample_predictions: int = 8       # save N val sample predictions per epoch as PNG


@dataclass
class CheckpointConfig:
    save_every_n_epochs: int = 5           # in addition to latest+best
    save_best_metric: str = "val_iou"      # which val metric drives 'best'
    save_best_mode: str = "max"            # 'max' or 'min'
    keep_last_n: int = 3                   # how many periodic ckpts to retain
    save_optimizer_state: bool = True
    resume_from: Optional[str] = None      # path to .pt to resume; None = start fresh


# ══════════════════════════════════════════════════════════════════════
# Top-level config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    post: PostProcessConfig = field(default_factory=PostProcessConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ── Serialization ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return _asdict_safe(self)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Install pyyaml: pip install pyyaml") from e
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Install pyyaml: pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _from_dict(cls, data)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return _from_dict(cls, d)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Override fields via dotted path. e.g. {'train.batch_size': 16}."""
        for key, value in overrides.items():
            target: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], value)


# ══════════════════════════════════════════════════════════════════════
# Helpers (recursively build nested dataclasses from dicts)
# ══════════════════════════════════════════════════════════════════════

def _asdict_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _asdict_safe(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _asdict_safe(v) for k, v in obj.items()}
    return obj


def _from_dict(cls, data: dict) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            v = data[f.name]
            if dataclasses.is_dataclass(f.type) or (
                isinstance(f.type, str) and f.type.endswith("Config")
            ):
                # Resolve string forward references
                inner_cls = f.default_factory().__class__ \
                    if f.default_factory is not dataclasses.MISSING \
                    else None
                if inner_cls and isinstance(v, dict):
                    kwargs[f.name] = _from_dict(inner_cls, v)
                else:
                    kwargs[f.name] = v
            else:
                kwargs[f.name] = v
    return cls(**kwargs)
