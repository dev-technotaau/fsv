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
from training.calibration import (
    _collect_val_logits,
    _collect_val_probs_by_image,
    fit_temperature,
    fit_per_subcategory_thresholds,
    lookup_threshold,
)

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
                 cutmix_alpha: float = 1.0,
                 mosaic_p: float = 0.0):
        self.base = base_size
        self.lo = min_factor
        self.hi = max_factor
        self.enabled = enabled
        self.patch_size = max(1, int(patch_size))
        # NOTE: the rng object is stored as a SEED + per-call counter instead of
        # a stateful `random.Random` because the collator gets pickled to all
        # DataLoader workers (with persistent_workers=True). A stateful RNG
        # would mean every worker shares the SAME sequence of CutMix boxes —
        # reducing the regularization effect by num_workers×. We pull fresh
        # entropy from numpy's global RNG (which IS re-seeded per worker by
        # `seed_worker`) on each call to keep per-worker diversity.
        self._base_seed = int(seed)
        self.cutmix_p = float(cutmix_p)
        self.cutmix_alpha = float(cutmix_alpha)
        self.mosaic_p = float(mosaic_p)

    def _make_rng(self) -> "random.Random":
        """Return a fresh `random.Random` instance per-call, seeded from the
        global numpy RNG (which `seed_worker` re-seeds for each DataLoader
        worker). Ensures CutMix boxes differ across workers + epochs."""
        # numpy global RNG seed → fresh Python random instance
        try:
            seed = int(np.random.randint(0, 2**31 - 1))
        except Exception:
            seed = self._base_seed
        return random.Random(seed)

    def _maybe_mosaic(self, samples: list[dict]) -> list[dict]:
        """Mosaic 4-way augmentation: for each triggered sample, replace it
        with a 4-quadrant composite drawn from current + 3 OTHER batch
        samples. Image AND mask AND CGM metadata stitched in lockstep.

        Unlike CutMix (rectangle-from-B onto A, 2-way), Mosaic exposes the
        model to 4 different scene contexts per step — particularly useful
        for fence boundaries that meet sky/grass/wall in unusual transitions.
        """
        if self.mosaic_p <= 0 or len(samples) < 4:
            return samples
        rng = self._make_rng()
        # Use a SHALLOW COPY of samples for indexing; we mutate `out` only.
        out = list(samples)
        N = len(samples)
        for i in range(N):
            if rng.random() > self.mosaic_p:
                continue
            # Pick 3 DIFFERENT samples (not i itself) to fill quadrants TR/BL/BR
            other_ids = [j for j in range(N) if j != i]
            if len(other_ids) < 3:
                continue
            j_tr, j_bl, j_br = rng.sample(other_ids, 3)
            quad_samples = [samples[i], samples[j_tr], samples[j_bl], samples[j_br]]
            # All must share the same (C, H, W) — guaranteed AFTER the
            # multi-scale resize step. Be defensive.
            shape0 = quad_samples[0]["image"].shape
            if any(s["image"].shape != shape0 for s in quad_samples):
                continue
            _, H, W = shape0
            # Sample a (cx, cy) center for the mosaic in the H x W canvas.
            # Range [H/4, 3H/4] keeps all 4 quadrants meaningfully sized.
            cy = rng.randint(H // 4, max(H // 4 + 1, 3 * H // 4))
            cx = rng.randint(W // 4, max(W // 4 + 1, 3 * W // 4))
            # Build mosaic image + mask
            img_out = quad_samples[0]["image"].clone()
            msk_out = quad_samples[0]["mask"].clone()
            # Quadrant layout (cv2 / numpy convention; y grows downward):
            #   TL: y < cy AND x < cx   (sample i)
            #   TR: y < cy AND x >= cx  (j_tr)
            #   BL: y >= cy AND x < cx  (j_bl)
            #   BR: y >= cy AND x >= cx (j_br)
            # TL is already img_out (just keep). Paste the other three.
            img_tr, msk_tr = quad_samples[1]["image"], quad_samples[1]["mask"]
            img_bl, msk_bl = quad_samples[2]["image"], quad_samples[2]["mask"]
            img_br, msk_br = quad_samples[3]["image"], quad_samples[3]["mask"]
            img_out[:, :cy, cx:] = img_tr[:, :cy, cx:]
            msk_out[:cy, cx:] = msk_tr[:cy, cx:]
            img_out[:, cy:, :cx] = img_bl[:, cy:, :cx]
            msk_out[cy:, :cx] = msk_bl[cy:, :cx]
            img_out[:, cy:, cx:] = img_br[:, cy:, cx:]
            msk_out[cy:, cx:] = msk_br[cy:, cx:]
            # CGM target: if ANY quadrant had positive pixels, the result is
            # pos. Read from the actual mosaic mask (source of truth) so we
            # don't drift from CutMix's same-property fix.
            mixed_meta = dict(quad_samples[0].get("metadata", {}))
            mixed_meta["class"] = "pos" if int(msk_out.any().item()) == 1 \
                                          else mixed_meta.get("class")
            mixed_meta["mosaic_quadrants_from"] = [
                quad_samples[k].get("metadata", {}).get("id") for k in range(4)
            ]
            # Sample weight: average of the 4 weights (each quadrant
            # contributes ~25% of pixels — close enough; slight asymmetry
            # from the (cx, cy) center cancels in expectation).
            w_avg = sum(float(s["sample_weight"].item())
                         for s in quad_samples) / 4.0
            out[i] = {
                **samples[i],
                "image": img_out,
                "mask": msk_out,
                "metadata": mixed_meta,
                "sample_weight": torch.tensor(w_avg, dtype=torch.float32),
            }
        return out

    def _maybe_cutmix(self, samples: list[dict]) -> list[dict]:
        """In-place pair-wise CutMix on a list of dict samples.
        Pairs (0,1), (2,3), ...; if odd number, last sample is left alone.
        For each pair: cut a random box from sample B, paste over sample A.
        Image AND mask AND CGM-target metadata get the same cut/paste.
        """
        if self.cutmix_p <= 0 or len(samples) < 2:
            return samples
        rng = self._make_rng()
        # Pair adjacent samples for mixing
        for i in range(0, len(samples) - 1, 2):
            if rng.random() > self.cutmix_p:
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
            cx = rng.randint(0, W - 1)
            cy = rng.randint(0, H - 1)
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
            # CGM mislabel fix: the per-image fence/non-fence label is read
            # from `metadata.class` downstream. If A was "neg" and B pasted
            # fence pixels into A, the resulting image now CONTAINS fence —
            # so the CGM target must become "pos". Recompute from the mixed
            # mask (any positive pixel → pos). Also fan the sample_weight
            # toward B proportional to the cut-area so review-source weighting
            # follows the mixed image content.
            mixed_meta = dict(a.get("metadata", {}))
            if int(msk_a.any().item()) == 1:
                mixed_meta["class"] = "pos"
            else:
                mixed_meta["class"] = mixed_meta.get("class")
            mixed_meta["cutmix_lam"] = float(lam)
            # Blend per-sample loss weight by cut area: 1-lam is the fraction
            # of A's pixels replaced by B.
            wa = float(a["sample_weight"].item())
            wb = float(b["sample_weight"].item())
            blended_w = lam * wa + (1.0 - lam) * wb
            samples[i] = {
                **a,
                "image": img_a,
                "mask": msk_a,
                "metadata": mixed_meta,
                "sample_weight": torch.tensor(blended_w, dtype=torch.float32),
            }
        return samples

    def __call__(self, batch: list[dict]) -> dict:
        if not self.enabled and self.cutmix_p <= 0 and self.mosaic_p <= 0:
            return _default_collate(batch)

        # 1. Multi-scale resize (whole batch to one scale)
        if self.enabled:
            rng = self._make_rng()
            s = rng.uniform(self.lo, self.hi)
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

        # 2. Mosaic (4-way) BEFORE CutMix so CutMix can further mix the
        # already-mosaic'd samples. Order matters: mosaic produces a single
        # composite per sample; CutMix then tweaks pair-wise on top.
        batch = self._maybe_mosaic(batch)

        # 3. CutMix (pair-wise within the batch)
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

    use_ra = bool(getattr(cfg.train, "use_randaugment", False))
    ra_n = int(getattr(cfg.train, "randaugment_n", 2))
    ra_m = int(getattr(cfg.train, "randaugment_m", 10))
    if cfg.data.image_size <= 768:
        train_aug = phase1_train_aug(
            cfg.data.image_size,
            use_randaugment=use_ra, randaugment_n=ra_n, randaugment_m=ra_m,
        )
        val_aug = phase1_val_aug(cfg.data.image_size)
    else:
        train_aug = phase2_train_aug(
            cfg.data.image_size,
            use_randaugment=use_ra, randaugment_n=ra_n,
            # Phase 2 uses a smaller magnitude (6 vs 10) since FT is gentler
            randaugment_m=min(ra_m, 6),
        )
        val_aug = phase2_val_aug(cfg.data.image_size)
    if use_ra:
        logger.info(
            f"RandAugment ON  n={ra_n}  m_phase={'10/phase1' if cfg.data.image_size <= 768 else f'{min(ra_m,6)}/phase2'}"
        )

    train_ds = FenceDataset(
        splits_dir / f"{cfg.data.train_split}.jsonl",
        splits_dir / f"{cfg.data.train_split}_masks.jsonl",
        transform=train_aug,
        weight_by_review_source=cfg.loss.weight_by_review_source,
        min_fence_pixels_for_pos=int(getattr(
            cfg.data, "min_fence_pixels_for_pos", 0
        )),
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
        mosaic_p=float(getattr(cfg.train, "mosaic_p", 0.0)),
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
        # CRITICAL: pass the dataset's actual rows (post-filter), NOT the raw
        # JSONL. The dataset drops rows via `min_fence_pixels_for_pos`, so
        # len(train_ds) < len(jsonl). Using raw JSONL row count would make the
        # sampler emit OOB indices that the dataset retries 5x and falls back
        # to a deterministic idx — corrupting both the balance and the
        # per-bucket distribution for rare classes.
        rows = list(train_ds.img_rows)
        sample_weights = compute_balanced_sample_weights(
            rows=rows,
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
              save_samples_to: Optional[Path] = None,
              apply_post_process: bool = False,
              temperature: Optional[float] = None,
              per_subcat_thresholds: Optional[dict[str, float]] = None
              ) -> dict[str, float]:
    """Validation / test loop.

    Args:
        apply_post_process: if True and `cfg.post.enabled`, run the
            (CRF/morphology/CC) cascade on each sigmoid probability map and
            score the BINARIZED-AND-CLEANED mask. Use this for the final
            test eval so reported numbers match the deployed pipeline. For
            per-epoch val we keep this OFF (CRF is slow, ~1s/image).
        temperature: optional scalar T to divide logits by BEFORE sigmoid.
            If None, falls back to `cfg.post.temperature` (default 1.0 = no
            scaling). Set after fitting on val via `fit_temperature`.
        per_subcat_thresholds: optional dict {subcategory: threshold} from
            `fit_per_subcategory_thresholds`. When provided, threshold is
            looked up per-sample by `metadata.subcategory`. Falls back to
            `cfg.post.binarize_threshold` for unknown subcategories or
            when this arg is None.
    """
    model.eval()
    # Binarize threshold: defaults to 0.5, configurable via cfg.post.binarize_threshold
    # so the threshold the model is scored against matches the threshold the
    # browser ONNX client will use. Same threshold used by the post-process
    # cascade below.
    binarize_thr = float(getattr(cfg.post, "binarize_threshold", 0.5))
    # Temperature: explicit arg > cfg.post.temperature > 1.0
    if temperature is None:
        temperature = float(getattr(cfg.post, "temperature", 1.0))
    temperature = max(1e-6, float(temperature))
    # Whether per-sample thresholds will be applied — if so, the accumulator's
    # default threshold becomes a moot fallback for unbucketed images only.
    use_per_subcat = (per_subcat_thresholds is not None
                       and isinstance(per_subcat_thresholds, dict)
                       and len(per_subcat_thresholds) > 0)
    accumulator = SegMetricsAccumulator(threshold=binarize_thr, boundary_kernel=5)
    saved = 0
    # Use AMP at val too — same dtype as training. ~2x faster val without
    # affecting metric accuracy (sigmoid outputs are upcast to fp32 anyway).
    amp_dtype = torch.bfloat16 if cfg.optim.amp_dtype == "bf16" else torch.float16
    use_amp = cfg.optim.use_amp and device.type == "cuda"

    # Optional: a non-zero `val_inference_size` resizes the input to that size
    # BEFORE the model forward (useful for evaluating phase-1 ckpts at higher
    # res without rebuilding the val_aug, or to A/B test resolution). 0 = use
    # whatever the val dataloader produced (the default; matches train size).
    val_size = int(getattr(cfg.train, "val_inference_size", 0) or 0)

    # Wire the post-process cascade only if asked AND enabled in config —
    # lazily import to keep optional cv2/pydensecrf deps off the train path.
    post_cfg = cfg.post if apply_post_process and getattr(cfg.post, "enabled", False) else None
    post_fn = None
    if post_cfg is not None:
        try:
            from training.post_process import post_process as post_fn
            from training.post_process import availability_report
            avail = availability_report()
            logger.info(f"Post-process at test eval — backends: {avail}")
        except Exception as e:
            logger.warning(f"Post-process unavailable, falling back to raw threshold: {e}")
            post_fn = None
            post_cfg = None

    for batch in val_dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        if val_size > 0 and (x.shape[-1] != val_size or x.shape[-2] != val_size):
            ps = max(1, int(patch_size))
            target = max(ps * 4, int(round(val_size / ps)) * ps)
            x = F.interpolate(x, size=(target, target), mode="bilinear",
                                align_corners=False)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                  enabled=use_amp):
            if cfg.train.use_tta:
                probs = _inference_tta(model, x, cfg, patch_size=patch_size,
                                        temperature=temperature)
            else:
                out = model(x)
                logits = out.refined_logits if out.refined_logits is not None else out.mask_logits
                # Cast to fp32 BEFORE sigmoid for bf16 saturation hygiene,
                # then divide by temperature for calibrated probabilities.
                probs = torch.sigmoid(logits.float().squeeze(1) / temperature)
        # Upcast probs to fp32 so the threshold/comparison is stable
        probs = probs.float()
        # Resize predictions back to GT resolution if val_size changed input
        if probs.shape[-2:] != y.shape[-2:]:
            probs = F.interpolate(probs.unsqueeze(1), size=y.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False).squeeze(1)

        # Apply per-sample threshold (per-subcategory) by pre-binarizing the
        # probability map: pixels >= threshold(subcategory) become 1.0, else
        # 0.0. The accumulator is then scored at threshold=0.5 against this
        # already-binarized probability — equivalent to scoring at the
        # per-subcategory threshold against the original prob.
        if use_per_subcat:
            sample_thrs = []
            for m in batch["metadata"]:
                sample_thrs.append(lookup_threshold(
                    per_subcat_thresholds,
                    m.get("subcategory"),
                    default=binarize_thr,
                ))
            thr_vec = torch.tensor(sample_thrs, dtype=probs.dtype,
                                     device=probs.device).view(-1, 1, 1)
            probs = (probs >= thr_vec).float()

        # Post-process cascade (final test eval only): CRF / morphology / CC.
        # Operates per-image on CPU numpy; we recover a binary mask and then
        # cast back to a float "prob" of {0.0, 1.0} so the same accumulator
        # path computes IoU / precision / recall on the POST-PROCESSED mask.
        if post_fn is not None and post_cfg is not None:
            # x_for_post is the ORIGINAL (pre-resize) RGB image for CRF guidance.
            # The training pipeline normalizes images; un-normalize for CRF.
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
            x_unnorm = (batch["image"].to(device) * std.view(1, 3, 1, 1)
                        + mean.view(1, 3, 1, 1)).clamp(0, 1)
            cleaned = []
            B = probs.shape[0]
            for i in range(B):
                p_np = probs[i].detach().cpu().numpy()
                img_np = (x_unnorm[i].permute(1, 2, 0).detach().cpu().numpy()
                          * 255.0).astype("uint8")
                # post_process accepts (H, W) float in [0,1] + (H, W, 3) uint8
                try:
                    m_post = post_fn(p_np, img_np, post_cfg)            # (H, W) uint8 {0, 1}
                except Exception as e:
                    logger.warning(f"post_process failed on item {i}, skipping cascade: {e}")
                    m_post = (p_np >= binarize_thr).astype("uint8")
                cleaned.append(torch.from_numpy(m_post.astype("float32")))
            probs = torch.stack(cleaned, dim=0).to(device=probs.device)
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
def _inference_tta(model, x, cfg, patch_size: int,
                    temperature: float = 1.0) -> torch.Tensor:
    """Test-time augmentation: avg over scales + horizontal flip + optional
    photometric perturbations (brightness, gamma).

    Geometric TTA (scales + flip) tests "what if the fence were a different
    size / mirrored". Photometric TTA tests "what if the fence were
    photographed under different lighting" — captures lighting robustness
    that geometric TTA can't simulate. The two axes are orthogonal and
    additive.

    Snaps every TTA scale to the backbone's patch stride. `temperature`
    divides logits before sigmoid (1.0 = no scaling).
    """
    H, W = x.shape[-2:]
    ps = max(1, int(patch_size))
    T = max(1e-6, float(temperature))
    accum = torch.zeros((x.shape[0], H, W), device=x.device, dtype=torch.float32)
    n = 0

    # Photometric variants applied to the INPUT in normalized space. The
    # input is already ImageNet-normalized (mean/std subtracted) — additive
    # brightness in normalized space ≈ delta on the [0,1] pixel space.
    photo_enabled = bool(getattr(cfg.train, "tta_photometric", False))
    brightness_deltas: list[float] = [0.0]
    if photo_enabled:
        brightness_deltas = [0.0] + list(getattr(
            cfg.train, "tta_photometric_brightness", (-0.10, 0.10)
        ))
    # Gamma is applied to the OUTPUT probability (post-sigmoid) — equivalent
    # to a small monotonic re-curve, modeling tonemapping differences.
    gamma_values: list[float] = [1.0]
    if photo_enabled:
        gamma_values = [1.0] + list(getattr(
            cfg.train, "tta_photometric_gamma", (0.9, 1.1)
        ))

    def _forward_and_accum(x_in: torch.Tensor, flip: bool) -> None:
        nonlocal accum, n
        if flip:
            x_in = torch.flip(x_in, dims=(-1,))
        out = model(x_in)
        lg = out.refined_logits if out.refined_logits is not None else out.mask_logits
        # fp32 sigmoid for bf16 saturation hygiene; divide logits by T for
        # calibrated probabilities (T=1 is the no-op default).
        probs = torch.sigmoid(lg.float().squeeze(1) / T)
        if flip:
            probs = torch.flip(probs, dims=(-1,))
        if probs.shape[-2:] != (H, W):
            probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                                    mode="bilinear",
                                    align_corners=False).squeeze(1)
        # Photometric gamma TTA: re-curve the probability output.
        for g in gamma_values:
            if g == 1.0:
                p_g = probs
            else:
                # Clamp to avoid 0^gamma issues at the tail
                p_g = probs.clamp(1e-6, 1.0).pow(float(g))
            accum += p_g
            n += 1

    for s in cfg.train.tta_scales:
        new_h = max(ps * 4, int(round(H * s / ps)) * ps)
        new_w = max(ps * 4, int(round(W * s / ps)) * ps)
        if (new_h, new_w) != (H, W):
            xs = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            xs = x
        # Brightness TTA: scaled image plus an additive offset (in normalized
        # space). Offset is the same per-channel — preserves color balance.
        for delta in brightness_deltas:
            xs_b = xs if delta == 0.0 else xs + float(delta)
            _forward_and_accum(xs_b, flip=False)
            if cfg.train.tta_flip:
                _forward_and_accum(xs_b, flip=True)

    return accum / max(1, n)


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
            # CRITICAL: pass `logger` so missing/unexpected key reports hit the
            # run log file (not just stdout where they can fly past unread).
            # `restore_rng=False` keeps phase 2's own seed (43) instead of
            # silently inheriting phase 1's final RNG state. The abort ratio
            # is conservative — if MORE than 5% of state_dict keys mismatch
            # between phase1 and phase2 (architectural drift), we'd rather
            # fail loud than silently random-init half the model and burn
            # $500 on a misconfigured run.
            CheckpointManager.load(
                init_path, model=model, strict=False,
                logger=logger,
                restore_rng=False,
                missing_key_abort_ratio=0.05,
            )
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
    elif cfg.optim.optimizer == "adamw8bit":
        # 8-bit Adam (bitsandbytes) — saves ~6 GB of optimizer state on a 1B-param
        # model with no measurable convergence difference vs fp32 Adam. Used widely
        # for fine-tuning at scale. Falls back to fp32 AdamW if bnb is not installed.
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, betas=cfg.optim.betas)
            logger.info("Using bitsandbytes AdamW8bit optimizer (8-bit moment states)")
        except ImportError:
            logger.warning("bitsandbytes not installed; falling back to fp32 AdamW. "
                           "Run `pip install bitsandbytes` to enable 8-bit Adam.")
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
                # Saved state.epoch is the epoch that was JUST COMPLETED.
                # Advance by 1 so the training loop skips it and starts at the
                # next epoch. Without this, resume re-runs the completed epoch.
                state.epoch = int(state.epoch) + 1
            logger.info(f"Resumed at epoch={state.epoch} step={state.global_step}  "
                         f"best {state.best_metric_name}={state.best_metric:.4f}")
        else:
            logger.warning(f"resume_from path does not exist: {rp} (starting fresh)")

    # Early-stopping bookkeeping
    epochs_no_improve = 0

    # Cross-epoch totals — saved into final summary so post-mortem can see
    # whether the run had stability issues (lots of skipped batches / OOMs).
    total_skipped_nonfinite = 0
    total_skipped_oom = 0
    best_epoch = 0
    # Dataset filter info: log into provenance once so checkpoints know how
    # the data was preprocessed (e.g., 103 SAM2 noise samples filtered).
    try:
        ds_filter_n = (len(getattr(train_dl.dataset, "img_rows", []))
                        if hasattr(train_dl, "dataset") else 0)
        provenance_snapshot["dataset_summary"] = {
            "train_split": cfg.data.train_split,
            "train_samples_after_filter": int(ds_filter_n),
            "min_fence_pixels_for_pos": int(getattr(
                cfg.data, "min_fence_pixels_for_pos", 0
            )),
        }
    except Exception:
        pass

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
        n_skipped_oom = 0
        last_grad_norm = float("nan")
        t_epoch = time.time()

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_dl):
          # OOM resilience: wrap the entire microbatch in a try/except so a
          # single bad batch (e.g. unusually large after upscale CutMix, or a
          # transient memory spike) doesn't waste hours of epoch progress.
          # We clear cached allocator state + zero grads + skip the step.
          # Re-raises non-OOM RuntimeErrors so we still see real bugs.
          try:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            w = batch["sample_weight"].to(device, non_blocking=True)
            # Per-image fence/non-fence label for the UNet3+ CGM head.
            # CutMix can paste fence pixels into a "neg" sample (or vice
            # versa), so we DERIVE is_pos directly from the ACTUAL pixels
            # currently in the mask. This is the source of truth — metadata
            # may be stale post-mix. (The collator also rewrites metadata.class
            # for downstream tools, but we don't depend on that here.)
            #   - any positive pixel in the (B, H, W) mask → pos image
            #   - else → neg
            # This kills the CGM mislabel bug: previously a neg-class sample
            # with fence pixels pasted in would supervise the CGM gate with
            # the LIE "no fence in this image", defeating its negative-image
            # suppression at inference time.
            is_pos = (y.flatten(1).any(dim=1).float()
                      .unsqueeze(1).to(device))     # (B, 1)

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
                    boundary_distance_logits=outputs.boundary_distance_logits,
                )

                # EMA self-distillation (Mean Teacher): pull live model toward
                # EMA-teacher predictions on the same input. Helps generalization
                # on hard cases. Frozen teacher = no gradient through it.
                #
                # Two perf/correctness fixes vs the original implementation:
                #   1. Teacher forward runs with `model.eval()` so dropout /
                #      DropPath / train-mode BN / extended gradient
                #      checkpointing are all disabled — otherwise the teacher
                #      prediction is noisy / stochastic and you're distilling
                #      garbage. (Critical for phase 2 where ema_distill_w=0.3.)
                #   2. Only run distillation on the FIRST microbatch of each
                #      grad-accumulation group. The teacher signal varies
                #      slowly between optimizer steps, so doing one teacher
                #      pass per optim step gives ~the same signal at half the
                #      cost (matters at phase 2 with grad_accum=2).
                ema_distill_w = float(getattr(cfg.loss, "ema_distill_weight", 0.0))
                distill_now = (
                    ema is not None
                    and ema_distill_w > 0
                    and (it % cfg.optim.grad_accumulation_steps == 0)
                )
                if distill_now:
                    was_training = model.training
                    was_extra_ckpt = bool(getattr(model, "_extra_checkpointing", False))
                    with torch.no_grad():
                        ema.apply_shadow(model)
                        try:
                            # Disable train-mode side effects on the teacher
                            model.eval()
                            # Also turn off our extended-checkpointing flag —
                            # there's nothing to recompute under no_grad and
                            # the wrapper would just waste activation memory.
                            if was_extra_ckpt:
                                model._extra_checkpointing = False
                            teacher_out = model(x)
                            teacher_logits = (teacher_out.refined_logits
                                              if teacher_out.refined_logits is not None
                                              else teacher_out.mask_logits)
                            teacher_prob = torch.sigmoid(teacher_logits.float().detach())
                        finally:
                            if was_extra_ckpt:
                                model._extra_checkpointing = True
                            if was_training:
                                model.train()
                            ema.restore(model)
                    # Student logits to compare against teacher prob
                    student_logits = (outputs.refined_logits
                                      if outputs.refined_logits is not None
                                      else outputs.mask_logits)
                    student_prob = torch.sigmoid(student_logits.float())
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
                # Running-mean across the epoch — smooths trends.
                cur_loss_mean = float(epoch_loss_t.item() / max(1, n_batches))
                # Per-step loss — preserves spikes the running mean would
                # otherwise hide (e.g. an early-divergence flash).
                cur_loss_step = float(loss.detach().item())
                msg = (f"[ep {epoch+1:>3}/{cfg.train.epochs}  it {it+1:>5}/{len(train_dl)}  "
                       f"step {state.global_step}]  loss={cur_loss_step:.4f}"
                       f" (avg {cur_loss_mean:.4f})  lr={lr_now:.2e}")
                if cfg.train.log_grad_norm and math.isfinite(last_grad_norm):
                    msg += f"  |g|={last_grad_norm:.3f}"
                if n_skipped_nonfinite > 0:
                    msg += f"  skipped={n_skipped_nonfinite}"
                if n_skipped_oom > 0:
                    msg += f"  oom_skips={n_skipped_oom}"
                logger.info(msg)
                if tb is not None:
                    # Per-step loss (catches spikes); epoch running-mean alias.
                    tb.add_scalar("train/loss", cur_loss_step, state.global_step)
                    tb.add_scalar("train/loss_epoch_mean", cur_loss_mean, state.global_step)
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
                    tb.add_scalar("train/skipped_oom",
                                   n_skipped_oom, state.global_step)
                    for k, v in comps.items():
                        tb.add_scalar(f"train/{k}", float(v.item()), state.global_step)
                    tb.flush()
          except torch.cuda.OutOfMemoryError as oom:
            # OOM during this microbatch (forward / backward / step). Clear
            # the allocator cache, zero any partial gradients, and skip to
            # the next batch. Losing one batch is far cheaper than crashing
            # the epoch — especially in phase 2 at 1024² where activation
            # spikes (CutMix / multi-scale upscale / unusual aspect ratios)
            # can briefly push us over the VRAM ceiling.
            n_skipped_oom += 1
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                f"CUDA OOM at epoch={epoch+1} iter={it+1}: skipping batch "
                f"(total OOM skips this epoch: {n_skipped_oom})  "
                f"detail={type(oom).__name__}: {str(oom)[:160]}"
            )
            continue

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
        # Roll up cross-epoch counters for the final summary
        total_skipped_nonfinite += n_skipped_nonfinite
        total_skipped_oom += n_skipped_oom

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
                    best_epoch = epoch + 1
                    # IMPORTANT: best.pt MUST contain the EMA weights — that's
                    # the model that achieved the metric (val ran under EMA).
                    ckpt_mgr.save_best_with_ema_swap(
                        model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        state=state,
                        extra={"val_history": list(val_history),
                                "best_epoch": best_epoch},
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
            # best.pt is post-EMA-swap (already the "best inference" weights);
            # use the strict-aware loader path but skip the abort check here
            # (we trained these weights ourselves, so any drift would be a
            # bug elsewhere we'd want to see — not a phase-mismatch).
            CheckpointManager.load(
                best_path, model=model, strict=False,
                logger=logger,
                restore_rng=False,   # don't perturb downstream RNG
            )

            # ── Post-training calibration (temperature scaling) ───────
            # Fit a single scalar T on val_dl to correct over-confident
            # sigmoid output. T then applied to ALL test predictions below.
            # 1.0 = no scaling (skip the fit).
            fitted_T = float(getattr(cfg.post, "temperature", 1.0))
            if bool(getattr(cfg.post, "fit_temperature_on_finish", False)):
                logger.info("\n" + "-" * 60)
                logger.info("Calibration: fitting temperature on val set")
                logger.info("-" * 60)
                try:
                    val_logits, val_targets, _ = _collect_val_logits(
                        model, val_dl, device,
                        amp_dtype=(torch.bfloat16
                                   if cfg.optim.amp_dtype == "bf16"
                                   else torch.float16),
                        use_amp=(cfg.optim.use_amp and device.type == "cuda"),
                    )
                    fitted_T = fit_temperature(val_logits, val_targets,
                                                 logger=logger)
                    cfg.post.temperature = fitted_T
                except Exception as e:
                    logger.warning(
                        f"Temperature fit failed ({type(e).__name__}: {e}); "
                        f"using T={fitted_T} from config"
                    )

            # ── Per-subcategory threshold tuning ──────────────────────
            # Sweep thresholds per subcategory on val_dl (with the fitted T
            # already applied so the sweep operates on calibrated probs).
            # Result is a dict {subcat: threshold} stored in checkpoint meta
            # and applied at test-eval time.
            subcat_thresholds: dict[str, float] = {}
            if bool(getattr(cfg.post, "per_subcategory_thresholds", False)):
                logger.info("\n" + "-" * 60)
                logger.info("Calibration: per-subcategory threshold sweep on val")
                logger.info("-" * 60)
                try:
                    per_image = _collect_val_probs_by_image(
                        model, val_dl, device,
                        amp_dtype=(torch.bfloat16
                                   if cfg.optim.amp_dtype == "bf16"
                                   else torch.float16),
                        use_amp=(cfg.optim.use_amp and device.type == "cuda"),
                        temperature=fitted_T,
                    )
                    subcat_thresholds = fit_per_subcategory_thresholds(
                        per_image,
                        sweep_min=float(getattr(cfg.post,
                                                  "threshold_sweep_min", 0.20)),
                        sweep_max=float(getattr(cfg.post,
                                                  "threshold_sweep_max", 0.80)),
                        sweep_step=float(getattr(cfg.post,
                                                   "threshold_sweep_step", 0.025)),
                        min_count=int(getattr(cfg.post,
                                                "threshold_sweep_min_count", 20)),
                        fallback_threshold=float(getattr(cfg.post,
                                                           "binarize_threshold", 0.5)),
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(
                        f"Per-subcategory threshold sweep failed "
                        f"({type(e).__name__}: {e}); using global threshold"
                    )
            # Force TTA on for the final test eval if configured. Per-epoch
            # val keeps its own setting (TTA there is too slow). All mutated
            # fields are restored AFTER the test eval to avoid contaminating
            # any subsequent code (e.g. a follow-up resume).
            saved_use_tta = cfg.train.use_tta
            saved_tta_scales = tuple(cfg.train.tta_scales)
            saved_tta_flip = cfg.train.tta_flip
            try:
                if cfg.train.tta_at_final_test:
                    cfg.train.use_tta = True
                    # Use a sensible default scale set if user left tta_scales=[1.0]
                    if list(cfg.train.tta_scales) == [1.0]:
                        cfg.train.tta_scales = (0.75, 1.0, 1.25)
                    cfg.train.tta_flip = True
                    logger.info(f"  TTA ENABLED for final test eval: "
                                 f"scales={list(cfg.train.tta_scales)}, "
                                 f"flip={cfg.train.tta_flip}")
                # Apply post-process cascade if configured — this is THE thing
                # that was previously dead config. Numbers reported here will
                # now match the deployed-with-post-process model.
                # Also pass fitted temperature + per-subcategory thresholds.
                test_metrics = validate(
                    model, test_dl, device, cfg, logger,
                    patch_size=patch_size,
                    save_samples_to=samples_dir / "test_final",
                    apply_post_process=bool(getattr(cfg.post, "enabled", False)),
                    temperature=fitted_T,
                    per_subcat_thresholds=(subcat_thresholds
                                            if subcat_thresholds else None),
                )
            finally:
                # Restore ALL mutated fields, even if validate() raised, so
                # config state stays clean for any downstream code.
                cfg.train.use_tta = saved_use_tta
                cfg.train.tta_scales = saved_tta_scales
                cfg.train.tta_flip = saved_tta_flip
            log_line = " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
            logger.info(f"TEST: {log_line}")
            jsonl_log(run_dir / "test_metrics.jsonl", {
                "epoch_at_best": state.epoch + 1,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "test_metrics": test_metrics,
                "post_processed": bool(getattr(cfg.post, "enabled", False)),
                "calibration": {
                    "temperature": float(fitted_T),
                    "per_subcat_thresholds": dict(subcat_thresholds) if subcat_thresholds else None,
                },
                "timestamp": _utcnow_iso(),
            })
            # ── Build the canonical FINAL SUMMARY ────────────────────
            # This dict is the source of truth for "what did this run produce".
            # Propagated INTO every checkpoint (best.pt, latest.pt, ema.pt,
            # best_inference.pt, all epoch_*.pt) AND written standalone as
            # final_summary.json so post-mortem doesn't require torch.load.
            calibration_block = {
                "temperature": float(fitted_T),
                "per_subcat_thresholds": (
                    dict(subcat_thresholds) if subcat_thresholds else None
                ),
                "binarize_threshold": float(
                    getattr(cfg.post, "binarize_threshold", 0.5)
                ),
                "fit_temperature_on_finish": bool(getattr(
                    cfg.post, "fit_temperature_on_finish", False
                )),
                "per_subcategory_thresholds_enabled": bool(getattr(
                    cfg.post, "per_subcategory_thresholds", False
                )),
            }
            elapsed_now = time.time() - t_start
            final_summary = {
                "completed_at": _utcnow_iso(),
                "total_runtime_seconds": float(elapsed_now),
                "total_runtime_hours": float(elapsed_now / 3600.0),
                "epochs_completed": int(state.epoch + 1),
                "epochs_planned": int(cfg.train.epochs),
                "total_optim_steps": int(state.global_step),
                "best_epoch": int(best_epoch),
                "best_metric_name": str(state.best_metric_name),
                "best_metric_value": float(state.best_metric),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "calibration": calibration_block,
                "post_processed": bool(getattr(cfg.post, "enabled", False)),
                "tta_at_final_test": bool(getattr(cfg.train, "tta_at_final_test", False)),
                "tta_scales_used": list(saved_tta_scales),
                "tta_flip_used": bool(saved_tta_flip),
                "stability_summary": {
                    "total_skipped_nonfinite_batches": int(total_skipped_nonfinite),
                    "total_skipped_oom_batches": int(total_skipped_oom),
                    "early_stopped": bool(epochs_no_improve >= cfg.train.early_stop_patience > 0),
                },
                "deployment_inputs": {
                    "imagenet_mean": [0.485, 0.456, 0.406],
                    "imagenet_std": [0.229, 0.224, 0.225],
                    "image_size": int(cfg.data.image_size),
                    "patch_size": int(getattr(model, "patch_size", 14)),
                    "backbone_name": str(cfg.model.backbone_name),
                    "use_refined": bool(getattr(cfg.model, "use_refinement_head", True)),
                    "channel_order": "RGB",
                },
            }

            # ── Standalone summary file (human-readable, no torch needed) ──
            summary_path = run_dir / "final_summary.json"
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(final_summary, f, indent=2, default=str)
                logger.info(f"Final summary written: {summary_path}")
            except Exception as e:
                logger.warning(
                    f"Could not write final_summary.json: {type(e).__name__}: {e}"
                )

            # ── Propagate final_summary into EVERY existing checkpoint ──
            # Updates the `extra` dict (or `meta` for inference-only files)
            # via atomic in-place rewrite — model weights are NOT touched.
            updates_for_extra = {
                "final_summary": final_summary,
                "calibration": calibration_block,
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            }
            checkpoint_files: list[tuple[Path, bool]] = [
                (ckpt_dir / "best.pt",  False),
                (ckpt_dir / "latest.pt", False),
                (ckpt_dir / "ema.pt",   False),
                (ckpt_dir / "best_inference.pt", True),   # in_meta=True
            ]
            # Also update every periodic checkpoint that survived pruning
            for periodic in sorted(ckpt_dir.glob("epoch_*.pt")):
                checkpoint_files.append((periodic, False))

            for ckpt_path, in_meta in checkpoint_files:
                try:
                    ok = CheckpointManager.update_extra(
                        ckpt_path, updates_for_extra, in_meta=in_meta,
                    )
                    if ok:
                        logger.info(
                            f"  Updated {ckpt_path.name} with final_summary "
                            f"(into {'meta' if in_meta else 'extra'})"
                        )
                except Exception as e:
                    logger.warning(
                        f"  Could not update {ckpt_path.name}: "
                        f"{type(e).__name__}: {e}"
                    )

            logger.info(
                f"Saved calibration + test metrics into ALL checkpoints  "
                f"(T={fitted_T:.4f}, "
                f"per_subcat_thresholds={len(subcat_thresholds) if subcat_thresholds else 0}, "
                f"test_iou={test_metrics.get('val_iou', float('nan')):.4f})"
            )

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
