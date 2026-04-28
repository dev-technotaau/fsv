"""training/train.py — Main training entry point.

Handles ONE phase per invocation. Drives:
    - Config loading (YAML + CLI overrides)
    - Dataset + DataLoader build (uses tools.dataset)
    - Model build (DINOv2 + Mask2Former-style decoder + optional refinement)
    - Loss build (BCE + Dice + Boundary + ...)
    - Optimizer + LR scheduler with layer-wise decay
    - AMP + grad accumulation + grad clipping
    - EMA
    - Checkpointing (latest, best, EMA, periodic)
    - Resume from any checkpoint
    - TensorBoard + console + JSONL logging
    - Multi-scale training
    - TTA at validation
    - Sample prediction PNGs each val epoch

Usage:
    python -m training.train --config configs/phase1.yaml
    python -m training.train --config configs/phase1.yaml --resume-from outputs/training_v2/phase1/checkpoints/latest.pt
    python -m training.train --config configs/phase2.yaml --init-from outputs/training_v2/phase1/checkpoints/best.pt
    python -m training.train --config configs/phase1.yaml --train.batch_size 4   # CLI override
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Local imports
from training.config import TrainingConfig
from training.checkpoint import CheckpointManager, CheckpointState
from training.ema import ModelEMA
from training.losses import CombinedLoss
from training.lr_scheduler import CosineWarmupScheduler, build_param_groups
from training.metrics import SegMetricsAccumulator
from training.model import build_model
from training import provenance as _provenance

# Dataset module from existing tools/
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _sanitize_filename_stem(s: str, max_len: int = 32) -> str:
    """Make any string safe to use as a filename stem (no path separators,
    no shell metachars). Keeps alnum + dash/underscore/dot."""
    out = "".join((c if (c.isalnum() or c in "-_.") else "_") for c in s)
    return out[:max_len] or "sample"


# ══════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> tuple[logging.Logger, Optional[object]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger, None


def setup_tensorboard(log_dir: Path) -> Optional[object]:
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir / "tensorboard")
        return writer
    except ImportError:
        return None


def jsonl_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════
# Multi-scale training collator (random scale per batch)
# ══════════════════════════════════════════════════════════════════════

class MultiScaleCollator:
    """Wraps a default collator and:
      1. Optionally resizes the WHOLE batch to a random scale within
         [min_factor, max_factor] of the configured size (multi-scale aug).
      2. Optionally applies CutMix: pairs samples in the batch and pastes a
         random rectangular region of one onto the other, with the same cut
         applied to the masks. Strong regularizer for segmentation.

    `patch_size` snaps the new H,W to a multiple of the backbone's patch stride
    so the ViT doesn't need to auto-pad on every batch.
    """
    def __init__(self, base_size: int, min_factor: float, max_factor: float,
                 enabled: bool, patch_size: int, seed: int = 0,
                 cutmix_p: float = 0.0,
                 cutmix_alpha: float = 1.0):
        self.base = base_size
        self.lo = min_factor
        self.hi = max_factor
        self.enabled = enabled
        self.patch_size = max(1, int(patch_size))
        self.rng = random.Random(seed)
        self.cutmix_p = float(cutmix_p)
        self.cutmix_alpha = float(cutmix_alpha)

    def _maybe_cutmix(self, samples: list[dict]) -> list[dict]:
        """In-place pair-wise CutMix on a list of dict samples.
        Pairs (0,1), (2,3), ...; if odd number, last sample is left alone.
        For each pair: cut a random box from sample B, paste over sample A.
        Image and mask both get the same cut/paste.
        """
        if self.cutmix_p <= 0 or len(samples) < 2:
            return samples
        # Pair adjacent samples for mixing
        for i in range(0, len(samples) - 1, 2):
            if self.rng.random() > self.cutmix_p:
                continue
            a, b = samples[i], samples[i + 1]
            img_a = a["image"]                     # (3, H, W) tensor
            img_b = b["image"]
            msk_a = a["mask"]                       # (H, W) tensor
            msk_b = b["mask"]
            if img_a.shape != img_b.shape:
                continue   # multi-scale already normalized but be defensive
            _, H, W = img_a.shape
            # Sample lambda from Beta(alpha, alpha); the cut region is
            # ~sqrt(1-lam) * image_size on each side. Standard CutMix.
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            cut_ratio = float(np.sqrt(1.0 - lam))
            cw = max(1, int(W * cut_ratio))
            ch = max(1, int(H * cut_ratio))
            cx = self.rng.randint(0, W - 1)
            cy = self.rng.randint(0, H - 1)
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            x1 = min(W, cx + cw // 2)
            y1 = min(H, cy + ch // 2)
            if x1 <= x0 or y1 <= y0:
                continue
            # Paste B's region onto A (clones to avoid aliasing across workers)
            img_a = img_a.clone()
            msk_a = msk_a.clone()
            img_a[:, y0:y1, x0:x1] = img_b[:, y0:y1, x0:x1]
            msk_a[y0:y1, x0:x1] = msk_b[y0:y1, x0:x1]
            samples[i] = {**a, "image": img_a, "mask": msk_a}
        return samples

    def __call__(self, batch: list[dict]) -> dict:
        if not self.enabled and self.cutmix_p <= 0:
            return _default_collate(batch)

        # 1. Multi-scale resize (whole batch to one scale)
        if self.enabled:
            s = self.rng.uniform(self.lo, self.hi)
            ps = self.patch_size
            new = max(ps * 4, int(round(self.base * s / ps) * ps))
            resized = []
            for sample in batch:
                img = sample["image"].unsqueeze(0)              # (1, 3, H, W)
                mask = sample["mask"].unsqueeze(0).unsqueeze(0).float()
                img2 = F.interpolate(img, size=(new, new), mode="bilinear",
                                       align_corners=False)
                mask2 = F.interpolate(mask, size=(new, new), mode="nearest")
                resized.append({
                    **sample,
                    "image": img2.squeeze(0),
                    "mask": mask2.squeeze(0).squeeze(0).to(sample["mask"].dtype),
                })
            batch = resized

        # 2. CutMix (pair-wise within the batch)
        batch = self._maybe_cutmix(batch)

        return _default_collate(batch)


def _default_collate(batch: list[dict]) -> dict:
    """Custom collator that handles our dict samples."""
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    weights = torch.stack([b["sample_weight"] for b in batch], dim=0)
    metas = [b["metadata"] for b in batch]
    return {"image": imgs, "mask": masks, "sample_weight": weights, "metadata": metas}


# ══════════════════════════════════════════════════════════════════════
# Build datasets / dataloaders
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg: TrainingConfig, logger: logging.Logger,
                       patch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits_dir = Path(cfg.data.splits_dir)

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
        patch_size=patch_size,
        seed=cfg.train.seed,
        cutmix_p=float(getattr(cfg.train, "cutmix_p", 0.0)),
        cutmix_alpha=float(getattr(cfg.train, "cutmix_alpha", 1.0)),
    )

    # A generator seeds the random sampler so train shuffle order is
    # deterministic across runs with the same `train.seed`. Without this the
    # Python random state may differ between machines/processes.
    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.train.seed)

    # Optional class-balanced sampling: oversample rare/hard categories so the
    # model sees them in proportion to (1/freq)^alpha. Mutually exclusive with
    # `shuffle=True` (the sampler IS the shuffle).
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
        # Quick distribution log — show the top oversampled buckets so it's
        # obvious whether the balance knob is doing what you wanted.
        from collections import Counter
        rows = load_jsonl(splits_dir / f"{cfg.data.train_split}.jsonl")
        keys = [r.get(cfg.train.balance_by, "unknown") or "unknown" for r in rows]
        counts = Counter(keys)
        logger.info(
            f"Balanced sampler ON  by={cfg.train.balance_by}  "
            f"alpha={cfg.train.balance_alpha}  buckets={len(counts)}  "
            f"min_count_floor={cfg.train.balance_min_count}"
        )
        # Log top 5 weight uplifts vs uniform sampling
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


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model: nn.Module, val_dl: DataLoader, device: torch.device,
              cfg: TrainingConfig, logger: logging.Logger,
              patch_size: int,
              save_samples_to: Optional[Path] = None) -> dict[str, float]:
    model.eval()
    accumulator = SegMetricsAccumulator(threshold=0.5, boundary_kernel=5)
    saved = 0
    # Use AMP at val too — same dtype as training. ~2x faster val without
    # affecting metric accuracy (sigmoid outputs are upcast to fp32 anyway).
    amp_dtype = torch.bfloat16 if cfg.optim.amp_dtype == "bf16" else torch.float16
    use_amp = cfg.optim.use_amp and device.type == "cuda"
    for batch in val_dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                  enabled=use_amp):
            if cfg.train.use_tta:
                probs = _inference_tta(model, x, cfg, patch_size=patch_size)
            else:
                out = model(x)
                logits = out.refined_logits if out.refined_logits is not None else out.mask_logits
                probs = torch.sigmoid(logits.squeeze(1))
        # Upcast probs to fp32 so the threshold/comparison is stable
        probs = probs.float()
        sc_list = [m.get("subcategory") for m in batch["metadata"]]
        accumulator.update(probs, y, subcategories=sc_list)

        # Save a few sample predictions as PNG
        if save_samples_to is not None and saved < cfg.log.save_sample_predictions:
            _save_sample_pngs(x, y, probs, save_samples_to,
                               start_idx=saved, max_count=cfg.log.save_sample_predictions - saved,
                               metadata=batch["metadata"])
            saved += min(x.shape[0], cfg.log.save_sample_predictions - saved)

    metrics = accumulator.compute()
    return metrics


@torch.no_grad()
def _inference_tta(model, x, cfg, patch_size: int) -> torch.Tensor:
    """Test-time augmentation: avg over scales + horizontal flip.
    Snaps every TTA scale to the backbone's patch stride."""
    H, W = x.shape[-2:]
    ps = max(1, int(patch_size))
    accum = torch.zeros((x.shape[0], H, W), device=x.device, dtype=torch.float32)
    n = 0
    for s in cfg.train.tta_scales:
        new_h = max(ps * 4, int(round(H * s / ps)) * ps)
        new_w = max(ps * 4, int(round(W * s / ps)) * ps)
        if (new_h, new_w) != (H, W):
            xs = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            xs = x
        out = model(xs)
        lg = out.refined_logits if out.refined_logits is not None else out.mask_logits
        probs = torch.sigmoid(lg.squeeze(1))
        if probs.shape[-2:] != (H, W):
            probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                                   mode="bilinear", align_corners=False).squeeze(1)
        accum += probs
        n += 1
        if cfg.train.tta_flip:
            xf = torch.flip(xs, dims=(-1,))
            outf = model(xf)
            lgf = outf.refined_logits if outf.refined_logits is not None else outf.mask_logits
            pf = torch.sigmoid(lgf.squeeze(1))
            pf = torch.flip(pf, dims=(-1,))
            if pf.shape[-2:] != (H, W):
                pf = F.interpolate(pf.unsqueeze(1), size=(H, W),
                                    mode="bilinear", align_corners=False).squeeze(1)
            accum += pf
            n += 1
    return accum / n


def _save_sample_pngs(x: torch.Tensor, y: torch.Tensor, probs: torch.Tensor,
                       out_dir: Path, start_idx: int, max_count: int,
                       metadata: list[dict]) -> None:
    """Save a side-by-side PNG: [image | gt mask | predicted mask] per sample."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # De-normalize image (assume ImageNet stats from dataset.py)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    img = (img * 255).byte()
    for i in range(min(x.shape[0], max_count)):
        img_np = img[i].cpu().permute(1, 2, 0).numpy()
        gt_np = (y[i].cpu().numpy().astype(np.uint8)) * 255
        pr_np = ((probs[i].cpu().numpy() >= 0.5).astype(np.uint8)) * 255
        H, W = img_np.shape[:2]
        # Convert masks to RGB for stacking
        gt_rgb = np.stack([gt_np] * 3, axis=-1)
        pr_rgb = np.stack([pr_np] * 3, axis=-1)
        side = np.concatenate([img_np, gt_rgb, pr_rgb], axis=1)
        raw_id = (metadata[i].get("id") if i < len(metadata) else None) \
            or f"sample_{start_idx + i}"
        iid = _sanitize_filename_stem(str(raw_id))
        PILImage.fromarray(side).save(out_dir / f"{iid}.png",
                                        optimize=False, compress_level=1)


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════

def train_one_phase(cfg: TrainingConfig) -> int:
    # Output dir
    run_dir = Path(cfg.log.log_dir) / cfg.log.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "val_samples"
    metrics_jsonl = run_dir / "val_metrics.jsonl"

    # Save resolved config
    cfg.to_yaml(run_dir / "config.yaml")

    logger, _ = setup_logging(run_dir)
    tb = setup_tensorboard(run_dir / "logs") if cfg.log.use_tensorboard else None
    logger.info("=" * 60)
    logger.info(f"Run: {cfg.log.run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Config:\n{json.dumps(cfg.to_dict(), indent=2, default=str)}")

    # Reproducibility
    set_seed(cfg.train.seed, cfg.train.deterministic)

    # Free perf knob: TF32 matmul on Ampere+ (~2x speedup, negligible accuracy
    # loss for training). Default in PyTorch >=2.0 is "highest" (= no TF32);
    # "high" enables TF32 for matmul, keeps cuDNN at fp32. Strictly opt-in.
    if not cfg.train.deterministic:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}, "
                    f"compute {cap_major}.{cap_minor}, "
                    f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
        # bf16 requires compute capability >= 8.0 (Ampere).
        if cfg.optim.use_amp and cfg.optim.amp_dtype == "bf16" and cap_major < 8:
            logger.warning(
                f"bf16 requested but compute capability is {cap_major}.{cap_minor} "
                f"(<8.0). bf16 will run in software via autocast and be SLOW. "
                f"Switching amp_dtype to 'fp16' is strongly recommended."
            )

    # Verify dataset splits BEFORE building dataloaders.
    logger.info("Verifying split integrity...")
    integrity = verify_split_integrity(
        splits_dir=Path(cfg.data.splits_dir),
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.test_split),
        check_mask_files_exist=True,
    )
    for name, s in integrity.items():
        logger.info(f"  {name:<10s}  rows={s['rows']:>6,}  pos={s['pos']:>5,}  "
                     f"neg={s['neg']:>5,}  manual={s['manual']:>5,}")

    # Loss — set pos_weight if requested
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

    # Initialize from a previous checkpoint? (Phase 2 from Phase 1 best)
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

    # Data — build now that we know the patch size
    train_dl, val_dl, test_dl = build_dataloaders(cfg, logger, patch_size=patch_size)

    # Optimizer (param groups with layer-wise LR decay)
    param_groups = build_param_groups(
        model,
        head_lr=cfg.optim.base_lr,
        backbone_lr=cfg.optim.backbone_lr,
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
        # bf16 doesn't need a grad scaler; fp16 does
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

    # EMA
    ema = ModelEMA(model, decay=cfg.train.ema_decay,
                    warmup_steps=cfg.train.ema_warmup_steps) \
        if cfg.train.use_ema else None

    # Snapshot full config + provenance ONCE so every checkpoint is
    # self-describing without bloating individual write paths.
    config_dict_snapshot = cfg.to_dict()
    provenance_snapshot = _provenance.collect()
    provenance_snapshot["run_name"] = cfg.log.run_name
    provenance_snapshot["run_dir"] = str(run_dir)
    provenance_snapshot["pipeline_version"] = "training/v2"

    # Rolling history of recent val metrics — bundled into every checkpoint
    # so reviewers can see "did training converge cleanly?" without opening
    # TensorBoard. Keep last 10 epochs (small, ~2-3 KB total).
    from collections import deque
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

    # Early-stopping bookkeeping
    epochs_no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting training ({cfg.train.epochs - state.epoch} epochs remaining)")
    logger.info("=" * 60)
    t_start = time.time()

    for epoch in range(state.epoch, cfg.train.epochs):
        state.epoch = epoch
        model.train()
        # Accumulate losses as TENSORS to avoid per-step GPU<->CPU sync
        epoch_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_comp_t: dict[str, torch.Tensor] = {}
        n_batches = 0
        n_skipped_nonfinite = 0
        last_grad_norm = float("nan")
        t_epoch = time.time()

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_dl):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            w = batch["sample_weight"].to(device, non_blocking=True)
            # Per-image fence/non-fence label for the UNet3+ CGM head.
            # Built from the dataset's `class` field (see tools/dataset.py).
            # 'pos' = contains fence, anything else (neg / unknown) = no fence.
            is_pos = torch.tensor(
                [(m.get("class") == "pos") for m in batch["metadata"]],
                dtype=torch.float32, device=device,
            ).unsqueeze(1)                          # (B, 1)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
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
                )

                # EMA self-distillation (Mean Teacher): pull live model toward
                # EMA-teacher predictions on the same input. Helps generalization
                # on hard cases. Frozen teacher = no gradient through it.
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
                    # Student logits to compare against teacher prob
                    student_logits = (outputs.refined_logits
                                      if outputs.refined_logits is not None
                                      else outputs.mask_logits)
                    student_prob = torch.sigmoid(student_logits)
                    distill_loss = F.mse_loss(student_prob, teacher_prob)
                    comps["ema_distill"] = distill_loss.detach()
                    loss = loss + ema_distill_w * distill_loss.float()

            # Drop the batch entirely if loss is NaN/Inf — preserves training
            # stability under occasional bad augmentations or numerical edge
            # cases. Without this, ONE bad batch poisons the optimizer.
            if cfg.train.skip_step_on_nonfinite_loss and not torch.isfinite(loss):
                n_skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_for_bwd = loss / cfg.optim.grad_accumulation_steps
            if scaler is not None:
                scaler.scale(loss_for_bwd).backward()
            else:
                loss_for_bwd.backward()

            # Step optimizer every grad_accumulation_steps iterations
            if (it + 1) % cfg.optim.grad_accumulation_steps == 0:
                if cfg.optim.grad_clip_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.optim.grad_clip_norm,
                    )
                    last_grad_norm = float(grad_norm_t)

                # Track whether the optimizer actually took a step. With fp16
                # AMP, scaler.step() is a NO-OP if grads contain inf — we must
                # NOT advance the scheduler/EMA/global_step in that case.
                step_skipped = False
                if scaler is not None:
                    pre_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # If scale dropped, scaler skipped the optimizer step
                    step_skipped = scaler.get_scale() < pre_scale
                else:
                    # bf16/fp32: also skip if grad norm is non-finite
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

            # Accumulate losses for logging (still on GPU)
            epoch_loss_t = epoch_loss_t + loss.detach().float()
            for k, v in comps.items():
                if k in epoch_comp_t:
                    epoch_comp_t[k] = epoch_comp_t[k] + v.float()
                else:
                    epoch_comp_t[k] = v.float().clone()
            n_batches += 1

            # Log per-step (forces a sync at this cadence — that's intentional)
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
                    # Per-group LRs: head (highest), backbone-min (lowest),
                    # plus the conventional "lr" alias to the highest group.
                    all_lrs = [g["lr"] for g in optimizer.param_groups]
                    tb.add_scalar("train/lr", lr_now, state.global_step)
                    tb.add_scalar("train/lr_head", max(all_lrs), state.global_step)
                    tb.add_scalar("train/lr_backbone_min", min(all_lrs),
                                   state.global_step)
                    if cfg.train.log_grad_norm and math.isfinite(last_grad_norm):
                        tb.add_scalar("train/grad_norm", last_grad_norm, state.global_step)
                    tb.add_scalar("train/skipped_batches",
                                   n_skipped_nonfinite, state.global_step)
                    for k, v in comps.items():
                        tb.add_scalar(f"train/{k}", float(v.item()), state.global_step)
                    tb.flush()

        # ── End of epoch ─────────────────────────────────────────────
        # Single sync point per epoch for the running averages
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

            # Log val metrics
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
            # Append a SLIM version (without the heavy train_components dict)
            # to the bundled history; full row is still in val_metrics.jsonl.
            val_history.append({
                "epoch": epoch + 1, "global_step": state.global_step,
                "train_loss": float(epoch_loss),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "epoch_seconds": float(epoch_dt),
                "skipped_batches": int(n_skipped_nonfinite),
            })

            # Best-model tracking (with EMA-aware save)
            metric_value = val_metrics.get(state.best_metric_name)
            if metric_value is not None:
                if cfg.ckpt.save_best_mode == "max":
                    improved = metric_value > state.best_metric + cfg.train.early_stop_min_delta
                else:
                    improved = metric_value < state.best_metric - cfg.train.early_stop_min_delta
                if improved:
                    state.best_metric = metric_value
                    # IMPORTANT: best.pt MUST contain the EMA weights — that's
                    # the model that achieved the metric (val ran under EMA).
                    ckpt_mgr.save_best_with_ema_swap(
                        model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        state=state,
                        extra={"val_history": list(val_history)},
                        config_dict=config_dict_snapshot,
                        provenance=provenance_snapshot,
                    )
                    # Also publish a tiny weights-only snapshot for shipping
                    # (no optimizer state — typically 3-4x smaller).
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
            scaler=scaler, ema=ema, state=state,
            extra=history_extra,
            config_dict=config_dict_snapshot,
            provenance=provenance_snapshot,
        )
        if ema is not None:
            ckpt_mgr.save_ema(
                ema, model, state,
                config_dict=config_dict_snapshot,
                provenance=provenance_snapshot,
            )
        if (epoch + 1) % cfg.ckpt.save_every_n_epochs == 0:
            ckpt_mgr.save_periodic(
                epoch + 1, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                ema=ema, state=state,
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

    # ── Done ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete in {elapsed/3600:.2f}h")
    logger.info(f"Best {state.best_metric_name}: {state.best_metric:.4f}")
    logger.info(f"Outputs: {run_dir}")

    # ── Final test-set evaluation against best.pt ─────────────────────
    if cfg.train.run_test_eval_on_finish:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            logger.info("\n" + "=" * 60)
            logger.info(f"Final test-set eval (loading {best_path})")
            logger.info("=" * 60)
            CheckpointManager.load(best_path, model=model, strict=False)
            # Force TTA on for the final test eval if configured. Per-epoch
            # val keeps its own setting (TTA there is too slow).
            saved_use_tta = cfg.train.use_tta
            if cfg.train.tta_at_final_test:
                cfg.train.use_tta = True
                # Use a sensible default scale set if user left tta_scales=[1.0]
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

    if tb is not None:
        tb.close()
    return 0


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_overrides(extra: list[str]) -> dict:
    """Convert --section.field value pairs into a flat overrides dict."""
    out: dict = {}
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i + 1 < len(extra):
                val = extra[i + 1]
                # Try to coerce to int/float/bool
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
        description="Two-phase wood-fence segmentation training pipeline (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=str, required=True,
                     help="Path to YAML config file (configs/phase1.yaml or configs/phase2.yaml)")
    ap.add_argument("--resume-from", type=str, default=None,
                     help="Path to checkpoint to resume training (overrides config)")
    ap.add_argument("--init-from", type=str, default=None,
                     help="Path to checkpoint for fresh-init weight loading "
                          "(no optimizer/scheduler state); use for Phase 2 init from Phase 1 best")
    ap.add_argument("--run-name", type=str, default=None,
                     help="Override config.log.run_name")
    args, extra = ap.parse_known_args()

    cfg = TrainingConfig.from_yaml(args.config)
    overrides = parse_overrides(extra)
    if overrides:
        cfg.apply_overrides(overrides)
    if args.resume_from is not None:
        cfg.ckpt.resume_from = args.resume_from
    if args.init_from is not None:
        cfg.train.init_from = args.init_from
    if args.run_name is not None:
        cfg.log.run_name = args.run_name

    return train_one_phase(cfg)


if __name__ == "__main__":
    sys.exit(main())
