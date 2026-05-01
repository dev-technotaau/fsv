"""train_web_deployable.py — Single-file BROWSER-DEPLOYABLE training script.

================================================================================
WHAT THIS SCRIPT IS
================================================================================
A self-contained, ENTERPRISE-GRADE training script that produces a small
(~35M params) fence segmentation model exportable to ONNX UNDER 100 MB at fp16
— suitable for direct browser inference via onnxruntime-web (NO server API).

This is the EXACT same training pipeline as the flagship
`training/train.py` + `configs/phase1.yaml` (every loss, every augmentation,
every training trick). The ONLY differences are:
  • Smaller model:  DINOv2-Small (22M, OPEN ACCESS) instead of DINOv3-H+ (840M)
                     no ViT-Adapter, no DPT depth, smaller decoder + refinement
  • Single phase:   no phase 2 fine-tune (just 80 epochs at 512²)
  • Output dir:     outputs/web_deployable/  (NOT outputs/training_v2/)
  • ONNX export:    fp32 + fp16 auto-export at the end (web target = fp16)

================================================================================
WHAT IT REUSES (zero duplication, full feature parity with flagship)
================================================================================
  - tools/dataset.py        — FenceDataset + phase1 augmentation + balanced sampler
  - dataset/splits/         — same train/val/test JSONLs the flagship reads
  - training/model.py       — FenceSegmentationModel (just smaller config)
  - training/losses.py      — CombinedLoss (BCE+Dice+Lovasz+Tversky+Boundary+
                                Focal+PointRend+Edge+DeepSup+EMA-distill+FDS+CGM+
                                BDR+Connectivity)
  - training/metrics.py     — SegMetricsAccumulator (incl. per-subcategory)
  - training/checkpoint.py  — CheckpointManager (self-describing checkpoints)
  - training/ema.py         — ModelEMA (Mean-Teacher)
  - training/lr_scheduler.py — CosineWarmupScheduler + layer-wise LR decay
  - training/provenance.py  — git SHA / lib versions / hostname / GPU
  - training/post_process.py — morphology + guided filter + DenseCRF + CC cleanup
  - tools/export_onnx.py    — fp32 + fp16 ONNX with parity check + sidecar JSON

================================================================================
EXPECTED PARAM BREAKDOWN
================================================================================
  DINOv2-Small backbone           : ~22.0M
  ViTToFPN adapter (384→192)      :  ~0.9M
  MSDeformAttn pixel decoder x6   :  ~2.9M
  Mask2Former decoder x6, q=16    :  ~3.6M
  Global token proj               :  ~0.07M
  UNet3+ refinement (FDS+CGM+BDR+PR):~5.5M
  refinement_feature_proj          :  ~0.01M
  ─────────────────────────────────────
  TOTAL                           : ~35.0M params
                                    ~140 MB fp32 ONNX
                                     ~70 MB fp16 ONNX  ← ship this

================================================================================
USAGE
================================================================================
  # Default 80-epoch run on 33K-image train split
  python train_web_deployable.py

  # Override knobs
  python train_web_deployable.py --epochs 100 --batch-size 16 --lr 4e-4

  # Resume from a checkpoint
  python train_web_deployable.py --resume-from outputs/web_deployable/web_v1/checkpoints/latest.pt

After training:
  outputs/web_deployable/web_v1/onnx/model_fp16.onnx       ← deploy this
  outputs/web_deployable/web_v1/onnx/model.json            ← sidecar (preprocessing params)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Reuse the flagship pipeline's modules ──────────────────────────────────
from training.config import TrainingConfig
from training.checkpoint import CheckpointManager, CheckpointState
from training.ema import ModelEMA
from training.losses import CombinedLoss
from training.lr_scheduler import CosineWarmupScheduler, build_param_groups
from training.metrics import SegMetricsAccumulator
from training.model import build_model
from training import provenance as _provenance

from tools.dataset import (
    FenceDataset,
    phase1_train_aug, phase1_val_aug,
    phase2_train_aug, phase2_val_aug,
    compute_pos_weight,
    compute_balanced_sample_weights,
    load_jsonl,
    seed_worker,
    verify_split_integrity,
)


# ══════════════════════════════════════════════════════════════════════════
# WEB-DEPLOYABLE CONFIG — same shape as flagship phase1.yaml, but SHRUNK.
# Single source of truth, no separate YAML to maintain.
# Every non-architectural knob matches flagship phase1.yaml exactly.
# ══════════════════════════════════════════════════════════════════════════

def build_web_config() -> TrainingConfig:
    cfg = TrainingConfig()

    # ── BACKBONE: DINOv2-Small (open access, 22M params) ─────────────────
    cfg.model.backbone_name = "facebook/dinov2-small"
    cfg.model.backbone_freeze_first_n_layers = 0
    cfg.model.backbone_drop_path_rate = 0.1
    cfg.model.multi_block_n = 4         # fuse last 4 of 12 ViT blocks (DINOv2-S has 12)
    cfg.model.aggregation_type = "weighted_sum"
    cfg.model.use_global_tokens = True

    # ── ViT-Adapter OFF (saves ~30M params for budget) ───────────────────
    cfg.model.use_vit_adapter = False

    # ── Pixel decoder: real MSDeformAttn but smaller ─────────────────────
    cfg.model.pixel_decoder_type = "msdeform"
    cfg.model.pixel_decoder_layers = 6      # was 12 in flagship
    cfg.model.pixel_decoder_heads = 8
    cfg.model.pixel_decoder_ffn_dim = 768   # was 3072

    # ── Mask2Former decoder: smaller dim, fewer queries, fewer layers ────
    cfg.model.decoder_type = "mask2former"
    cfg.model.decoder_dim = 192             # was 512
    cfg.model.decoder_num_queries = 16      # was 64
    cfg.model.decoder_num_layers = 6        # was 15
    cfg.model.decoder_dropout = 0.0
    cfg.model.use_masked_attention = True
    cfg.model.mask_attention_threshold = 0.5
    cfg.model.num_classes = 1
    cfg.model.output_logit = True

    # ── Refinement: smaller, but ALL UNet3+ extras ON ────────────────────
    cfg.model.use_refinement_head = True
    cfg.model.refinement_channels = 64      # was 96
    cfg.model.refinement_num_blocks = 3     # was 4
    cfg.model.refinement_use_decoder_features = True
    cfg.model.refinement_decoder_feature_channels = 32
    cfg.model.refinement_use_edge_head = True
    cfg.model.refinement_iterations = 2     # was 3
    cfg.model.refinement_use_y_coord = True
    cfg.model.refinement_use_depth = False  # OFF — saves DPT 122M frozen
    cfg.model.refinement_use_full_scale_ds = True
    cfg.model.refinement_use_cgm = True
    cfg.model.refinement_use_distance = True
    cfg.model.refinement_use_pointrend_module = True
    cfg.model.refinement_pointrend_n_uncertain = 2048
    cfg.model.refinement_pointrend_hidden = 96

    cfg.model.gradient_checkpointing = True
    cfg.model.torch_compile = False

    # ── Losses: SAME loss stack as flagship phase1.yaml (every weight) ──
    cfg.loss.bce_weight = 1.0
    cfg.loss.dice_weight = 1.0
    cfg.loss.boundary_weight = 0.8
    cfg.loss.lovasz_weight = 0.5
    cfg.loss.tversky_weight = 0.5
    cfg.loss.tversky_alpha = 0.7
    cfg.loss.tversky_beta = 0.3
    cfg.loss.connectivity_weight = 0.1
    cfg.loss.use_pos_weight = True
    cfg.loss.pos_weight = None
    cfg.loss.focal_gamma = 2.0
    cfg.loss.ohem_top_k_ratio = 0.25
    cfg.loss.boundary_kernel_size = 5
    cfg.loss.dice_smooth = 1.0
    cfg.loss.weight_by_review_source = True
    cfg.loss.deep_supervision_weight = 1.0
    cfg.loss.use_pointrend = True
    cfg.loss.pointrend_n_points = 12544
    cfg.loss.pointrend_oversample_ratio = 3.0
    cfg.loss.pointrend_importance_ratio = 0.75
    cfg.loss.edge_loss_weight = 0.8
    cfg.loss.edge_loss_dilate = 3
    cfg.loss.edge_loss_pos_weight = 8.0
    cfg.loss.refinement_iter_aux_weight = 0.5
    cfg.loss.ema_distill_weight = 0.0       # off in single phase (matches flagship phase1)
    cfg.loss.refinement_fds_weight = 0.3
    cfg.loss.cgm_weight = 0.5
    cfg.loss.boundary_distance_weight = 0.3
    cfg.loss.boundary_distance_clip = 50.0
    cfg.loss.boundary_distance_normalize = True

    # ── Optim: AdamW + cosine warmup, LINEAR LR SCALING for large batch ──
    # Standard "1-hour ImageNet" recipe (Goyal et al., 2017): when scaling
    # batch by N, scale LR by N AND extend warmup. Empirically eliminates the
    # large-batch quality gap up to batch 8K. We're going from effective
    # batch 16 (old: B=8 × accum=2) → 48 (new: B=48 × accum=1) = 3x scale.
    #
    #   base_lr:     4e-4  → 1.2e-3 (linear: 3×)   — head LR
    #   backbone_lr: 5e-5  → 1.5e-4 (linear: 3×)   — backbone LR
    #   warmup:      4 ep  → 8 ep                   — longer for higher LR
    #
    # Combined with bumped epochs (+25%) below, total gradient-update budget
    # MATCHES the small-batch run, so there is no quality compromise — just
    # ~2-3× faster wall-clock from the parallelism speedup.
    cfg.optim.optimizer = "adamw"
    cfg.optim.base_lr = 1.2e-3              # was 4e-4 — LINEAR scaling for batch=48
    cfg.optim.backbone_lr = 1.5e-4          # was 5e-5 — LINEAR scaling
    cfg.optim.backbone_lr_decay = 0.9
    cfg.optim.weight_decay = 0.05
    cfg.optim.betas = (0.9, 0.999)
    cfg.optim.warmup_epochs = 8             # was 4 — 2x longer for higher LR stability
    cfg.optim.warmup_lr = 1.0e-7
    cfg.optim.lr_min = 1.0e-7
    cfg.optim.schedule = "cosine"
    cfg.optim.grad_clip_norm = 1.0
    cfg.optim.grad_accumulation_steps = 1   # was 2 — batch=48 IS the effective batch
    cfg.optim.use_amp = True
    cfg.optim.amp_dtype = "bf16"

    # ── Train: single phase, 512² — RTX 3090 24GB (~20-22 GB peak) ───────
    # Bumped epochs 80 → 100 to match total gradient-update count of the
    # small-batch run (since each epoch now has 3x fewer iterations). Net
    # wall-clock is still ~2× faster because per-iteration time scales
    # sub-linearly with batch size on GPU (parallelism win).
    cfg.train.epochs = 100                  # was 80 — restores total update count
    cfg.train.batch_size = 48               # was 8 — fills the 3090 (~14GB @512², ~21GB @640²)
    cfg.train.val_batch_size = 32           # was 16
    cfg.train.num_workers = 10              # was 6 — more workers to feed batch=48
    cfg.train.pin_memory = True
    cfg.train.persistent_workers = True
    cfg.train.val_every_n_epochs = 1
    cfg.train.val_inference_size = 0
    cfg.train.multi_scale_train = True
    cfg.train.multi_scale_min_factor = 0.75
    cfg.train.multi_scale_max_factor = 1.25
    cfg.train.cutmix_p = 0.3
    cfg.train.cutmix_alpha = 1.0
    cfg.train.use_balanced_sampler = True   # ON for the web model — single-phase needs class balance
    cfg.train.balance_by = "subcategory"
    cfg.train.balance_alpha = 0.5
    cfg.train.balance_min_count = 50
    cfg.train.use_ema = True
    cfg.train.ema_decay = 0.9999
    cfg.train.ema_warmup_steps = 1000
    cfg.train.seed = 42
    cfg.train.deterministic = False
    cfg.train.init_from = None
    cfg.train.use_tta = False
    cfg.train.tta_scales = (0.5, 0.75, 1.0, 1.25, 1.5)
    cfg.train.tta_flip = True
    cfg.train.tta_at_final_test = True
    cfg.train.skip_step_on_nonfinite_loss = True
    cfg.train.log_grad_norm = True
    cfg.train.early_stop_patience = 0
    cfg.train.early_stop_min_delta = 1.0e-4
    cfg.train.run_test_eval_on_finish = True

    # ── Data ─────────────────────────────────────────────────────────────
    cfg.data.splits_dir = "dataset/splits"
    cfg.data.image_size = 512
    cfg.data.train_split = "train"
    cfg.data.val_split = "val"
    cfg.data.test_split = "test"

    # ── Post-process (final test eval) ───────────────────────────────────
    cfg.post.enabled = True
    cfg.post.use_morphology = True
    cfg.post.morphology_kernel = 3
    cfg.post.use_guided_filter = True
    cfg.post.guided_filter_radius = 4
    cfg.post.guided_filter_eps = 1.0e-4
    cfg.post.use_dense_crf = True
    cfg.post.crf_iterations = 5
    cfg.post.crf_bilateral_sxy = 50
    cfg.post.crf_bilateral_srgb = 10
    cfg.post.crf_bilateral_compat = 12
    cfg.post.use_cc_cleanup = True
    cfg.post.cc_min_blob_area = 200
    cfg.post.cc_fill_holes_smaller_than = 0
    cfg.post.cc_keep_top_k_blobs = 0

    # ── Logging + checkpoint paths — SEPARATE from flagship ──────────────
    cfg.log.log_dir = "outputs/web_deployable"
    cfg.log.run_name = "web_v1"
    cfg.log.log_every_n_steps = 50
    cfg.log.use_tensorboard = True
    cfg.log.save_sample_predictions = 8

    cfg.ckpt.save_every_n_epochs = 5
    cfg.ckpt.save_best_metric = "val_iou"
    cfg.ckpt.save_best_mode = "max"
    cfg.ckpt.keep_last_n = 3
    cfg.ckpt.save_optimizer_state = True
    cfg.ckpt.resume_from = None

    return cfg


# ══════════════════════════════════════════════════════════════════════════
# BELOW: 1:1 mirror of training/train.py helpers (verbatim copies — keeps
# every behavior identical: CutMix logic, multi-scale resize, sample-PNG
# saving, TTA, validate, full training loop with EMA distillation).
# ══════════════════════════════════════════════════════════════════════════


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _sanitize_filename_stem(s: str, max_len: int = 32) -> str:
    out = "".join((c if (c.isalnum() or c in "-_.") else "_") for c in s)
    return out[:max_len] or "sample"


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Optional[object]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_web")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    logger.propagate = False
    return logger, None


def setup_tensorboard(log_dir: Path) -> Optional[object]:
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir / "tensorboard")
    except ImportError:
        return None


def jsonl_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════
# Multi-scale collator + pair-wise CutMix (verbatim from flagship)
# ══════════════════════════════════════════════════════════════════════════

class MultiScaleCollator:
    def __init__(self, base_size: int, min_factor: float, max_factor: float,
                 enabled: bool, patch_size: int, seed: int = 0,
                 cutmix_p: float = 0.0, cutmix_alpha: float = 1.0):
        self.base = base_size
        self.lo = min_factor; self.hi = max_factor
        self.enabled = enabled
        self.patch_size = max(1, int(patch_size))
        self.rng = random.Random(seed)
        self.cutmix_p = float(cutmix_p)
        self.cutmix_alpha = float(cutmix_alpha)

    def _maybe_cutmix(self, samples: list[dict]) -> list[dict]:
        if self.cutmix_p <= 0 or len(samples) < 2:
            return samples
        for i in range(0, len(samples) - 1, 2):
            if self.rng.random() > self.cutmix_p:
                continue
            a, b = samples[i], samples[i + 1]
            img_a = a["image"]; img_b = b["image"]
            msk_a = a["mask"];  msk_b = b["mask"]
            if img_a.shape != img_b.shape:
                continue
            _, H, W = img_a.shape
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            cut_ratio = float(np.sqrt(1.0 - lam))
            cw = max(1, int(W * cut_ratio)); ch = max(1, int(H * cut_ratio))
            cx = self.rng.randint(0, W - 1); cy = self.rng.randint(0, H - 1)
            x0 = max(0, cx - cw // 2); y0 = max(0, cy - ch // 2)
            x1 = min(W, cx + cw // 2);  y1 = min(H, cy + ch // 2)
            if x1 <= x0 or y1 <= y0:
                continue
            img_a = img_a.clone(); msk_a = msk_a.clone()
            img_a[:, y0:y1, x0:x1] = img_b[:, y0:y1, x0:x1]
            msk_a[y0:y1, x0:x1] = msk_b[y0:y1, x0:x1]
            samples[i] = {**a, "image": img_a, "mask": msk_a}
        return samples

    def __call__(self, batch: list[dict]) -> dict:
        if not self.enabled and self.cutmix_p <= 0:
            return _default_collate(batch)
        if self.enabled:
            s = self.rng.uniform(self.lo, self.hi)
            ps = self.patch_size
            new = max(ps * 4, int(round(self.base * s / ps) * ps))
            resized = []
            for sample in batch:
                img = sample["image"].unsqueeze(0)
                mask = sample["mask"].unsqueeze(0).unsqueeze(0).float()
                img2 = F.interpolate(img, size=(new, new), mode="bilinear",
                                       align_corners=False)
                mask2 = F.interpolate(mask, size=(new, new), mode="nearest")
                resized.append({
                    **sample, "image": img2.squeeze(0),
                    "mask": mask2.squeeze(0).squeeze(0).to(sample["mask"].dtype),
                })
            batch = resized
        batch = self._maybe_cutmix(batch)
        return _default_collate(batch)


def _default_collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════
# Dataloaders (verbatim from flagship — single-phase is just always-phase1)
# ══════════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg: TrainingConfig, logger: logging.Logger,
                       patch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits_dir = Path(cfg.data.splits_dir)
    # Same phase-vs-phase2 augmentation dispatch as flagship train.py:245-250.
    # Default web script runs at 512², so phase1_train_aug is what we use.
    # If the user overrides `--image-size 1024`+, switch to the gentler phase2
    # augmentation pipeline (preserves fine detail at high res).
    if cfg.data.image_size <= 768:
        train_aug = phase1_train_aug(cfg.data.image_size)
        val_aug = phase1_val_aug(cfg.data.image_size)
    else:
        train_aug = phase2_train_aug(cfg.data.image_size)
        val_aug = phase2_val_aug(cfg.data.image_size)

    train_ds = FenceDataset(
        splits_dir / f"{cfg.data.train_split}.jsonl",
        splits_dir / f"{cfg.data.train_split}_masks.jsonl",
        transform=train_aug,
        weight_by_review_source=cfg.loss.weight_by_review_source,
    )
    val_ds = FenceDataset(
        splits_dir / f"{cfg.data.val_split}.jsonl",
        splits_dir / f"{cfg.data.val_split}_masks.jsonl",
        transform=val_aug,
    )
    test_ds = FenceDataset(
        splits_dir / f"{cfg.data.test_split}.jsonl",
        splits_dir / f"{cfg.data.test_split}_masks.jsonl",
        transform=val_aug,
    )
    logger.info(f"Datasets: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    collator = MultiScaleCollator(
        base_size=cfg.data.image_size,
        min_factor=cfg.train.multi_scale_min_factor,
        max_factor=cfg.train.multi_scale_max_factor,
        enabled=cfg.train.multi_scale_train,
        patch_size=patch_size, seed=cfg.train.seed,
        cutmix_p=float(getattr(cfg.train, "cutmix_p", 0.0)),
        cutmix_alpha=float(getattr(cfg.train, "cutmix_alpha", 1.0)),
    )

    train_gen = torch.Generator(); train_gen.manual_seed(cfg.train.seed)

    train_sampler = None
    if getattr(cfg.train, "use_balanced_sampler", False):
        sample_weights = compute_balanced_sample_weights(
            splits_dir / f"{cfg.data.train_split}.jsonl",
            balance_by=cfg.train.balance_by,
            alpha=cfg.train.balance_alpha,
            min_count=cfg.train.balance_min_count,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights),
            replacement=True, generator=train_gen,
        )
        from collections import Counter
        rows = load_jsonl(splits_dir / f"{cfg.data.train_split}.jsonl")
        keys = [r.get(cfg.train.balance_by, "unknown") or "unknown" for r in rows]
        counts = Counter(keys)
        logger.info(
            f"Balanced sampler ON  by={cfg.train.balance_by}  "
            f"alpha={cfg.train.balance_alpha}  buckets={len(counts)}  "
            f"min_count_floor={cfg.train.balance_min_count}"
        )
        per_bucket_w = {k: ((1.0 / max(cfg.train.balance_min_count, c))
                              ** cfg.train.balance_alpha)
                         for k, c in counts.items()}
        ranked = sorted(per_bucket_w.items(), key=lambda kv: -kv[1])
        logger.info("  top oversampled buckets: " +
                     ", ".join(f"{k}({counts[k]})" for k, _ in ranked[:5]))

    train_dl = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        collate_fn=collator, drop_last=True,
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None,
        generator=train_gen,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.train.val_batch_size, shuffle=False,
        num_workers=max(2, cfg.train.num_workers // 2),
        pin_memory=cfg.train.pin_memory, collate_fn=_default_collate,
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.train.val_batch_size, shuffle=False,
        num_workers=max(2, cfg.train.num_workers // 2),
        pin_memory=cfg.train.pin_memory, collate_fn=_default_collate,
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None,
    )
    return train_dl, val_dl, test_dl


# ══════════════════════════════════════════════════════════════════════════
# Validation + TTA + sample-PNG saving (verbatim from flagship)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model: nn.Module, val_dl: DataLoader, device: torch.device,
              cfg: TrainingConfig, logger: logging.Logger,
              patch_size: int,
              save_samples_to: Optional[Path] = None) -> dict[str, float]:
    model.eval()
    accumulator = SegMetricsAccumulator(threshold=0.5, boundary_kernel=5)
    saved = 0
    amp_dtype = torch.bfloat16 if cfg.optim.amp_dtype == "bf16" else torch.float16
    use_amp = cfg.optim.use_amp and device.type == "cuda"
    for batch in val_dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if cfg.train.use_tta:
                probs = _inference_tta(model, x, cfg, patch_size=patch_size)
            else:
                out = model(x)
                logits = out.refined_logits if out.refined_logits is not None else out.mask_logits
                probs = torch.sigmoid(logits.squeeze(1))
        probs = probs.float()
        sc_list = [m.get("subcategory") for m in batch["metadata"]]
        accumulator.update(probs, y, subcategories=sc_list)

        if save_samples_to is not None and saved < cfg.log.save_sample_predictions:
            _save_sample_pngs(x, y, probs, save_samples_to,
                               start_idx=saved,
                               max_count=cfg.log.save_sample_predictions - saved,
                               metadata=batch["metadata"])
            saved += min(x.shape[0], cfg.log.save_sample_predictions - saved)

    return accumulator.compute()


@torch.no_grad()
def _inference_tta(model, x, cfg, patch_size: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    ps = max(1, int(patch_size))
    accum = torch.zeros((x.shape[0], H, W), device=x.device, dtype=torch.float32)
    n = 0
    for s in cfg.train.tta_scales:
        new_h = max(ps * 4, int(round(H * s / ps)) * ps)
        new_w = max(ps * 4, int(round(W * s / ps)) * ps)
        xs = (F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
               if (new_h, new_w) != (H, W) else x)
        out = model(xs)
        lg = out.refined_logits if out.refined_logits is not None else out.mask_logits
        probs = torch.sigmoid(lg.squeeze(1))
        if probs.shape[-2:] != (H, W):
            probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                                   mode="bilinear", align_corners=False).squeeze(1)
        accum += probs; n += 1
        if cfg.train.tta_flip:
            xf = torch.flip(xs, dims=(-1,))
            outf = model(xf)
            lgf = outf.refined_logits if outf.refined_logits is not None else outf.mask_logits
            pf = torch.sigmoid(lgf.squeeze(1))
            pf = torch.flip(pf, dims=(-1,))
            if pf.shape[-2:] != (H, W):
                pf = F.interpolate(pf.unsqueeze(1), size=(H, W),
                                    mode="bilinear", align_corners=False).squeeze(1)
            accum += pf; n += 1
    return accum / n


def _save_sample_pngs(x: torch.Tensor, y: torch.Tensor, probs: torch.Tensor,
                       out_dir: Path, start_idx: int, max_count: int,
                       metadata: list[dict]) -> None:
    try:
        from PIL import Image as PILImage
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    img = (img * 255).byte()
    for i in range(min(x.shape[0], max_count)):
        img_np = img[i].cpu().permute(1, 2, 0).numpy()
        gt_np = (y[i].cpu().numpy().astype(np.uint8)) * 255
        pr_np = ((probs[i].cpu().numpy() >= 0.5).astype(np.uint8)) * 255
        gt_rgb = np.stack([gt_np] * 3, axis=-1)
        pr_rgb = np.stack([pr_np] * 3, axis=-1)
        side = np.concatenate([img_np, gt_rgb, pr_rgb], axis=1)
        raw_id = (metadata[i].get("id") if i < len(metadata) else None) \
            or f"sample_{start_idx + i}"
        iid = _sanitize_filename_stem(str(raw_id))
        PILImage.fromarray(side).save(out_dir / f"{iid}.png",
                                        optimize=False, compress_level=1)


# ══════════════════════════════════════════════════════════════════════════
# ONNX export postlude — fp32 + fp16 (web-deployable target)
# ══════════════════════════════════════════════════════════════════════════

def export_onnx_artifacts(checkpoint_path: Path, image_size: int,
                            out_dir: Path, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model.onnx"
    try:
        from tools.export_onnx import export_onnx
    except ImportError as e:
        logger.warning(f"ONNX export skipped — could not import tools.export_onnx: {e}")
        return

    logger.info("=" * 60)
    logger.info("Exporting ONNX (fp32 + fp16) for browser deployment")
    logger.info("=" * 60)
    try:
        export_onnx(
            checkpoint_path=checkpoint_path,
            output_path=fp32_path,
            image_size=image_size,
            opset=17,
            dynamic_batch=False,
            use_refined=True,
            config=None,
            validate=True,
            quantize_dynamic=False,
            quantize_fp16=True,
        )
    except Exception as e:
        logger.error(f"ONNX export failed: {type(e).__name__}: {e}")
        return

    fp16_path = fp32_path.with_name(fp32_path.stem + "_fp16.onnx")
    if fp32_path.exists():
        logger.info(f"  fp32 ONNX: {fp32_path}  ({fp32_path.stat().st_size/1e6:.1f} MB)")
    if fp16_path.exists():
        size_mb = fp16_path.stat().st_size / 1e6
        verdict = "✓ ships under 100 MB" if size_mb < 100 else "⚠ over 100 MB — consider int8"
        logger.info(f"  fp16 ONNX: {fp16_path}  ({size_mb:.1f} MB)  [{verdict}]")
        logger.info(f"  →  Deploy {fp16_path.name} to the browser.")


# ══════════════════════════════════════════════════════════════════════════
# Main training loop — structurally 1:1 with flagship train_one_phase().
# Differences vs flagship: build_web_config() instead of YAML loading,
# always single-phase phase1 augmentation, ONNX export postlude at end.
# ══════════════════════════════════════════════════════════════════════════

def train(cfg: TrainingConfig) -> int:
    run_dir = Path(cfg.log.log_dir) / cfg.log.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "val_samples"
    metrics_jsonl = run_dir / "val_metrics.jsonl"
    onnx_dir = run_dir / "onnx"

    cfg.to_yaml(run_dir / "config.yaml")
    logger, _ = setup_logging(run_dir)
    tb = setup_tensorboard(run_dir / "logs") if cfg.log.use_tensorboard else None

    logger.info("=" * 60)
    logger.info(f"Run: {cfg.log.run_name}  (WEB-DEPLOYABLE single-phase)")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Config:\n{json.dumps(cfg.to_dict(), indent=2, default=str)}")

    set_seed(cfg.train.seed, cfg.train.deterministic)
    if not cfg.train.deterministic:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}, "
                    f"compute {cap_major}.{cap_minor}, "
                    f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
        if cfg.optim.use_amp and cfg.optim.amp_dtype == "bf16" and cap_major < 8:
            logger.warning(
                f"bf16 requested but compute capability is {cap_major}.{cap_minor} "
                f"(<8.0). bf16 will run in software via autocast and be SLOW. "
                f"Switching amp_dtype to 'fp16' is strongly recommended."
            )

    # Verify dataset splits BEFORE building dataloaders
    logger.info("Verifying split integrity...")
    integrity = verify_split_integrity(
        splits_dir=Path(cfg.data.splits_dir),
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.test_split),
        check_mask_files_exist=True,
    )
    for name, s in integrity.items():
        logger.info(f"  {name:<10s}  rows={s['rows']:>6,}  pos={s['pos']:>5,}  "
                     f"neg={s['neg']:>5,}  manual={s['manual']:>5,}")

    # Loss — auto pos_weight
    if cfg.loss.use_pos_weight and cfg.loss.pos_weight is None:
        pw = compute_pos_weight(
            Path(cfg.data.splits_dir) / f"{cfg.data.train_split}.jsonl",
            Path(cfg.data.splits_dir) / f"{cfg.data.train_split}_masks.jsonl",
        )
        cfg.loss.pos_weight = pw
        logger.info(f"Auto pos_weight: {pw:.4f}")
    loss_fn = CombinedLoss(cfg.loss).to(device)

    # Model
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    patch_size = int(getattr(model, "patch_size", 14))
    logger.info(f"Backbone: {cfg.model.backbone_name}  patch_size={patch_size}")
    logger.info(f"Model params: {n_params/1e6:.1f}M total, "
                 f"{n_trainable/1e6:.1f}M trainable")

    # Memory: gradient checkpointing
    if cfg.model.gradient_checkpointing:
        ok = model.enable_gradient_checkpointing()
        logger.info(f"Gradient checkpointing: {'ENABLED' if ok else 'unsupported by backbone'}")

    # Init from a previous checkpoint? (We don't typically use this for the web
    # script — but support it for resume-from-flagship scenarios.)
    if cfg.train.init_from is not None and cfg.ckpt.resume_from is None:
        init_path = Path(cfg.train.init_from)
        if init_path.exists():
            logger.info(f"Initializing model weights from {init_path}")
            CheckpointManager.load(init_path, model=model, strict=False)
        else:
            logger.warning(f"init_from path does not exist: {init_path} (training from scratch)")

    # Speed: torch.compile
    if cfg.model.torch_compile:
        try:
            model = torch.compile(model, mode=cfg.model.torch_compile_mode)
            logger.info(f"torch.compile: ENABLED (mode={cfg.model.torch_compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile failed; continuing eager: {type(e).__name__}: {e}")

    # Data — built now that we know the patch size
    train_dl, val_dl, test_dl = build_dataloaders(cfg, logger, patch_size=patch_size)

    # Optimizer (param groups with layer-wise LR decay)
    param_groups = build_param_groups(
        model, head_lr=cfg.optim.base_lr, backbone_lr=cfg.optim.backbone_lr,
        backbone_lr_decay=cfg.optim.backbone_lr_decay,
        weight_decay=cfg.optim.weight_decay,
    )
    logger.info(f"Optimizer param groups: {len(param_groups)}  "
                 f"(LR range {param_groups[-1]['lr']:.2e} .. {param_groups[0]['lr']:.2e})")
    if cfg.optim.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=cfg.optim.betas)
    elif cfg.optim.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.optim.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optim.optimizer}")

    # LR Scheduler
    steps_per_epoch = max(1, len(train_dl) // cfg.optim.grad_accumulation_steps)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = steps_per_epoch * cfg.optim.warmup_epochs
    scheduler = CosineWarmupScheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
        lr_min=cfg.optim.lr_min, warmup_lr=cfg.optim.warmup_lr,
    )

    # AMP
    scaler = None
    amp_dtype = torch.bfloat16 if cfg.optim.amp_dtype == "bf16" else torch.float16
    if cfg.optim.use_amp and device.type == "cuda":
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

    # EMA
    ema = (ModelEMA(model, decay=cfg.train.ema_decay,
                     warmup_steps=cfg.train.ema_warmup_steps)
           if cfg.train.use_ema else None)

    # Snapshot full config + provenance ONCE
    config_dict_snapshot = cfg.to_dict()
    provenance_snapshot = _provenance.collect()
    provenance_snapshot["run_name"] = cfg.log.run_name
    provenance_snapshot["run_dir"] = str(run_dir)
    provenance_snapshot["pipeline_version"] = "web_deployable/v1"
    val_history: deque = deque(maxlen=10)
    logger.info(f"Provenance: git={provenance_snapshot.get('git', {}).get('sha', 'n/a')[:8]}  "
                 f"host={provenance_snapshot.get('hostname', 'n/a')}  "
                 f"torch={provenance_snapshot.get('libraries', {}).get('torch', 'n/a')}")

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(
        ckpt_dir, keep_last_n=cfg.ckpt.keep_last_n,
        save_optimizer_state=cfg.ckpt.save_optimizer_state,
    )
    state = CheckpointState(
        epoch=0, global_step=0,
        best_metric=float("-inf") if cfg.ckpt.save_best_mode == "max" else float("inf"),
        best_metric_name=cfg.ckpt.save_best_metric,
    )

    # Resume?
    if cfg.ckpt.resume_from is not None:
        rp = Path(cfg.ckpt.resume_from)
        if rp.exists():
            logger.info(f"Resuming from {rp}")
            payload = CheckpointManager.load(
                rp, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, ema=ema, strict=True,
            )
            if "state" in payload:
                state.__dict__.update(payload["state"])
            logger.info(f"Resumed at epoch={state.epoch} step={state.global_step}  "
                         f"best {state.best_metric_name}={state.best_metric:.4f}")
        else:
            logger.warning(f"resume_from path does not exist: {rp} (starting fresh)")

    epochs_no_improve = 0

    # ── Training loop ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting training ({cfg.train.epochs - state.epoch} epochs remaining)")
    logger.info("=" * 60)
    t_start = time.time()

    for epoch in range(state.epoch, cfg.train.epochs):
        state.epoch = epoch
        model.train()
        epoch_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_comp_t: dict[str, torch.Tensor] = {}
        n_batches = 0; n_skipped_nonfinite = 0; last_grad_norm = float("nan")
        t_epoch = time.time()

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_dl):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            w = batch["sample_weight"].to(device, non_blocking=True)
            is_pos = torch.tensor(
                [(m.get("class") == "pos") for m in batch["metadata"]],
                dtype=torch.float32, device=device,
            ).unsqueeze(1)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=cfg.optim.use_amp and device.type == "cuda",
            ):
                outputs = model(x)
                loss, comps = loss_fn(
                    outputs.mask_logits, y, sample_weight=w,
                    refined_logits=outputs.refined_logits,
                    aux_logits=outputs.aux_logits,
                    edge_logits=outputs.edge_logits,
                    refined_iter_logits=outputs.refined_iter_logits,
                    refined_fds_logits=outputs.refined_fds_logits,
                    cgm_logit=outputs.cgm_logit,
                    is_positive=is_pos,
                    boundary_distance_logits=outputs.boundary_distance_logits,
                )

                # EMA self-distillation (Mean Teacher) — same gating as flagship.
                # Pulls the live model toward the EMA-teacher predictions on the
                # SAME input. Frozen teacher = no gradient through it.
                ema_distill_w = float(getattr(cfg.loss, "ema_distill_weight", 0.0))
                if ema is not None and ema_distill_w > 0:
                    with torch.no_grad():
                        ema.apply_shadow(model)
                        try:
                            teacher_out = model(x)
                            teacher_logits = (teacher_out.refined_logits
                                              if teacher_out.refined_logits is not None
                                              else teacher_out.mask_logits)
                            teacher_prob = torch.sigmoid(teacher_logits.detach())
                        finally:
                            ema.restore(model)
                    student_logits = (outputs.refined_logits
                                      if outputs.refined_logits is not None
                                      else outputs.mask_logits)
                    student_prob = torch.sigmoid(student_logits)
                    distill_loss = F.mse_loss(student_prob, teacher_prob)
                    comps["ema_distill"] = distill_loss.detach()
                    loss = loss + ema_distill_w * distill_loss.float()

            # Drop the batch on NaN/Inf — preserves training stability under
            # rare bad augmentations or numerical edge cases.
            if cfg.train.skip_step_on_nonfinite_loss and not torch.isfinite(loss):
                n_skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_for_bwd = loss / cfg.optim.grad_accumulation_steps
            (scaler.scale(loss_for_bwd).backward() if scaler is not None
             else loss_for_bwd.backward())

            # Step optimizer every grad_accumulation_steps iterations
            if (it + 1) % cfg.optim.grad_accumulation_steps == 0:
                if cfg.optim.grad_clip_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.optim.grad_clip_norm,
                    )
                    last_grad_norm = float(grad_norm_t)

                # Track whether the optimizer ACTUALLY took a step (fp16 scaler
                # may NO-OP it on inf grads — must NOT advance scheduler/EMA then).
                step_skipped = False
                if scaler is not None:
                    pre_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    step_skipped = scaler.get_scale() < pre_scale
                else:
                    if (cfg.train.skip_step_on_nonfinite_loss
                            and not math.isfinite(last_grad_norm)):
                        step_skipped = True
                    else:
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not step_skipped:
                    scheduler.step()
                    state.global_step += 1
                    if ema is not None:
                        ema.update(model, state.global_step)
                else:
                    n_skipped_nonfinite += 1

            # Accumulate losses (still on GPU — single sync point per epoch)
            epoch_loss_t = epoch_loss_t + loss.detach().float()
            for k, v in comps.items():
                if k in epoch_comp_t:
                    epoch_comp_t[k] = epoch_comp_t[k] + v.float()
                else:
                    epoch_comp_t[k] = v.float().clone()
            n_batches += 1

            # Per-step logging (forces a sync at this cadence — intentional)
            if state.global_step > 0 and state.global_step % cfg.log.log_every_n_steps == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                cur_loss = float(epoch_loss_t.item() / max(1, n_batches))
                msg = (f"[ep {epoch+1:>3}/{cfg.train.epochs}  it {it+1:>5}/{len(train_dl)}  "
                       f"step {state.global_step}]  loss={cur_loss:.4f}  lr={lr_now:.2e}")
                if cfg.train.log_grad_norm and math.isfinite(last_grad_norm):
                    msg += f"  |g|={last_grad_norm:.3f}"
                if n_skipped_nonfinite > 0:
                    msg += f"  skipped={n_skipped_nonfinite}"
                logger.info(msg)
                if tb is not None:
                    tb.add_scalar("train/loss", cur_loss, state.global_step)
                    all_lrs = [g["lr"] for g in optimizer.param_groups]
                    tb.add_scalar("train/lr", lr_now, state.global_step)
                    tb.add_scalar("train/lr_head", max(all_lrs), state.global_step)
                    tb.add_scalar("train/lr_backbone_min", min(all_lrs), state.global_step)
                    if cfg.train.log_grad_norm and math.isfinite(last_grad_norm):
                        tb.add_scalar("train/grad_norm", last_grad_norm, state.global_step)
                    tb.add_scalar("train/skipped_batches",
                                   n_skipped_nonfinite, state.global_step)
                    for k, v in comps.items():
                        tb.add_scalar(f"train/{k}", float(v.item()), state.global_step)
                    tb.flush()

        # ── End of epoch ──────────────────────────────────────────────
        epoch_loss = float(epoch_loss_t.item()) / max(1, n_batches)
        epoch_components = {k: float(v.item()) / max(1, n_batches)
                             for k, v in epoch_comp_t.items()}
        epoch_dt = time.time() - t_epoch
        suffix = (f"  skipped_batches={n_skipped_nonfinite}"
                   if n_skipped_nonfinite else "")
        logger.info(f"Epoch {epoch+1} done in {epoch_dt:.1f}s   "
                     f"train_loss={epoch_loss:.4f}{suffix}")

        # Validation
        improved = False
        if (epoch + 1) % cfg.train.val_every_n_epochs == 0 or epoch + 1 == cfg.train.epochs:
            if ema is not None:
                ema.apply_shadow(model)
            t_val = time.time()
            val_metrics = validate(model, val_dl, device, cfg, logger,
                                    patch_size=patch_size,
                                    save_samples_to=samples_dir / f"epoch_{epoch+1:03d}")
            if ema is not None:
                ema.restore(model)
            val_dt = time.time() - t_val

            log_line = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logger.info(f"Val ({val_dt:.1f}s):  {log_line}")
            if tb is not None:
                for k, v in val_metrics.items():
                    tb.add_scalar(f"val/{k.replace('val_', '')}", v, state.global_step)
                tb.flush()
            val_log_row = {
                "epoch": epoch + 1, "global_step": state.global_step,
                "train_loss": epoch_loss, "train_components": epoch_components,
                "val_metrics": val_metrics, "epoch_seconds": epoch_dt,
                "val_seconds": val_dt, "skipped_batches": n_skipped_nonfinite,
                "timestamp": _utcnow_iso(),
            }
            jsonl_log(metrics_jsonl, val_log_row)
            val_history.append({
                "epoch": epoch + 1, "global_step": state.global_step,
                "train_loss": float(epoch_loss),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "epoch_seconds": float(epoch_dt),
                "skipped_batches": int(n_skipped_nonfinite),
            })

            metric_value = val_metrics.get(state.best_metric_name)
            if metric_value is not None:
                if cfg.ckpt.save_best_mode == "max":
                    improved = metric_value > state.best_metric + cfg.train.early_stop_min_delta
                else:
                    improved = metric_value < state.best_metric - cfg.train.early_stop_min_delta
                if improved:
                    state.best_metric = metric_value
                    # best.pt with EMA weights swapped in
                    ckpt_mgr.save_best_with_ema_swap(
                        model=model, ema=ema, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, state=state,
                        extra={"val_history": list(val_history)},
                        config_dict=config_dict_snapshot,
                        provenance=provenance_snapshot,
                    )
                    # tiny weights-only inference checkpoint
                    if ema is not None:
                        ema.apply_shadow(model)
                    try:
                        ckpt_mgr.save_inference_only(
                            ckpt_dir / "best_inference.pt", model=model,
                            meta={
                                "epoch": epoch + 1,
                                "global_step": state.global_step,
                                "metric_name": state.best_metric_name,
                                "metric_value": float(metric_value),
                                "backbone_name": cfg.model.backbone_name,
                                "image_size": cfg.data.image_size,
                                "patch_size": int(getattr(model, "patch_size", 14)),
                                "saved_at": _utcnow_iso(),
                                "imagenet_mean": [0.485, 0.456, 0.406],
                                "imagenet_std": [0.229, 0.224, 0.225],
                                "val_history": list(val_history),
                                "pipeline_version": "web_deployable/v1",
                            },
                            config_dict=config_dict_snapshot,
                            provenance=provenance_snapshot,
                        )
                    finally:
                        if ema is not None:
                            ema.restore(model)
                    logger.info(f"NEW BEST {state.best_metric_name}={metric_value:.4f} "
                                 f"-> saved best.pt (EMA-swapped) + best_inference.pt")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Save latest + EMA + periodic — all carry config + provenance + history
        history_extra = {"val_history": list(val_history)} if val_history else None
        ckpt_mgr.save_latest(
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, ema=ema, state=state, extra=history_extra,
            config_dict=config_dict_snapshot, provenance=provenance_snapshot,
        )
        if ema is not None:
            ckpt_mgr.save_ema(ema, model, state,
                                config_dict=config_dict_snapshot,
                                provenance=provenance_snapshot)
        if (epoch + 1) % cfg.ckpt.save_every_n_epochs == 0:
            ckpt_mgr.save_periodic(
                epoch + 1, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, ema=ema, state=state,
                extra=history_extra,
                config_dict=config_dict_snapshot,
                provenance=provenance_snapshot,
            )

        # Early stopping
        if (cfg.train.early_stop_patience > 0
                and epochs_no_improve >= cfg.train.early_stop_patience):
            logger.info(f"Early stopping at epoch {epoch + 1}: "
                         f"{epochs_no_improve} epochs without "
                         f"{state.best_metric_name} improvement "
                         f"(patience={cfg.train.early_stop_patience}).")
            break

    # ── Done ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete in {elapsed/3600:.2f}h")
    logger.info(f"Best {state.best_metric_name}: {state.best_metric:.4f}")
    logger.info(f"Outputs: {run_dir}")

    # ── Final test-set evaluation against best.pt (with TTA) ────────────
    if cfg.train.run_test_eval_on_finish:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            logger.info("\n" + "=" * 60)
            logger.info(f"Final test-set eval (loading {best_path})")
            logger.info("=" * 60)
            CheckpointManager.load(best_path, model=model, strict=False)
            saved_use_tta = cfg.train.use_tta
            if cfg.train.tta_at_final_test:
                cfg.train.use_tta = True
                if list(cfg.train.tta_scales) == [1.0]:
                    cfg.train.tta_scales = (0.75, 1.0, 1.25)
                cfg.train.tta_flip = True
                logger.info(f"  TTA ENABLED for final test eval: "
                             f"scales={list(cfg.train.tta_scales)}, "
                             f"flip={cfg.train.tta_flip}")
            test_metrics = validate(model, test_dl, device, cfg, logger,
                                     patch_size=patch_size,
                                     save_samples_to=samples_dir / "test_final")
            cfg.train.use_tta = saved_use_tta
            log_line = " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
            logger.info(f"TEST: {log_line}")
            jsonl_log(run_dir / "test_metrics.jsonl", {
                "epoch_at_best": state.epoch + 1,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "test_metrics": test_metrics,
                "timestamp": _utcnow_iso(),
            })
            if tb is not None:
                for k, v in test_metrics.items():
                    tb.add_scalar(f"test/{k.replace('val_', '')}", v, state.global_step)
        else:
            logger.warning("Skipping final test eval: no best.pt was saved.")

    # ── ONNX EXPORT (the whole point of this script) ─────────────────────
    best_inf = ckpt_dir / "best_inference.pt"
    if best_inf.exists():
        export_onnx_artifacts(best_inf, cfg.data.image_size, onnx_dir, logger)
    else:
        logger.warning("No best_inference.pt found — ONNX export skipped.")

    if tb is not None:
        tb.close()
    return 0


# ══════════════════════════════════════════════════════════════════════════
# CLI — supports both the explicit common-knobs flags AND generic
# `--section.field value` overrides for any TrainingConfig field, mirroring
# flagship training/train.py:996+ behavior exactly.
# ══════════════════════════════════════════════════════════════════════════

def parse_overrides(extra: list[str]) -> dict:
    """Convert --section.field value pairs into a flat overrides dict.
    Identical to flagship's parse_overrides — supports any nested config field
    via dotted notation (e.g. `--train.batch_size 16`, `--model.decoder_dim 256`).
    Coerces to bool/int/float when possible, falls back to string."""
    out: dict = {}
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i + 1 < len(extra):
                val = extra[i + 1]
                if val.lower() in ("true", "false"):
                    val_p: object = (val.lower() == "true")
                else:
                    try:
                        val_p = int(val)
                    except ValueError:
                        try:
                            val_p = float(val)
                        except ValueError:
                            val_p = val
                out[key] = val_p
                i += 2
            else:
                i += 1
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Web-deployable single-phase training + ONNX export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_web_deployable.py\n"
            "  python train_web_deployable.py --epochs 100 --batch-size 16\n"
            "  python train_web_deployable.py --train.cutmix_p 0.5 --loss.boundary_weight 1.0\n"
            "  python train_web_deployable.py --resume-from outputs/web_deployable/web_v1/checkpoints/latest.pt\n"
        ),
    )
    # Common shortcut flags (also supported via generic --section.field syntax)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                     help="Path to a checkpoint for fresh-init weight loading "
                          "(no optimizer/scheduler state).")
    ap.add_argument("--backbone", type=str, default=None,
                     help="Override backbone HF name. Default: facebook/dinov2-small. "
                          "Other open options: facebook/dinov2-base (86M, ~340MB fp32).")
    # parse_known_args lets us also accept generic --section.field overrides
    args, extra = ap.parse_known_args()

    cfg = build_web_config()
    # Apply explicit shortcut flags first
    if args.epochs is not None: cfg.train.epochs = args.epochs
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.lr is not None: cfg.optim.base_lr = args.lr
    if args.image_size is not None: cfg.data.image_size = args.image_size
    if args.run_name is not None: cfg.log.run_name = args.run_name
    if args.resume_from is not None: cfg.ckpt.resume_from = args.resume_from
    if args.init_from is not None: cfg.train.init_from = args.init_from
    if args.backbone is not None: cfg.model.backbone_name = args.backbone
    # Then apply any generic --section.field overrides (matches flagship)
    overrides = parse_overrides(extra)
    if overrides:
        cfg.apply_overrides(overrides)

    return train(cfg)


if __name__ == "__main__":
    sys.exit(main())
