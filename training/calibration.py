"""training/calibration.py — post-training probability + threshold calibration.

Two related procedures, both run AFTER training against the validation set:

  1. fit_temperature(model, val_dl, ...) → float T
       Optimizes a single scalar `T` so that `sigmoid(logits / T)` minimizes
       binary cross-entropy on the val set. T > 1 flattens over-confident
       predictions; T < 1 sharpens under-confident ones. The optimum reduces
       Expected Calibration Error (ECE) by ~5x typically and gives a small
       (+0.2-0.5%) IoU lift via better thresholding decisions.

       Reference: Guo et al., "On Calibration of Modern Neural Networks",
       ICML 2017.

  2. fit_per_subcategory_thresholds(model, val_dl, subcategories, ...) → dict
       Sweeps thresholds 0.2 → 0.8 per subcategory on the val set and picks
       the argmax-IoU value per bucket. Different fence types (cedar / vinyl /
       chain-link / wrought-iron) have different precision/recall trade-offs;
       chain-link wires die at threshold 0.5 but live at 0.35.

Both functions return JSON-serializable Python objects so they can be
embedded directly into the checkpoint meta dict and re-loaded at inference.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════
# 1. Temperature scaling
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_val_logits(
    model: nn.Module,
    val_dl: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    use_refined: bool = True,
    max_pixels_per_image: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, list[Optional[str]]]:
    """Forward the model over val_dl and return (logits, targets, subcats).

    To keep memory bounded, we subsample at most `max_pixels_per_image` pixels
    per image (uniform random). Calibration only needs a representative sample
    of (logit, target) pairs — not every pixel.

    Returns:
        logits: (N,) fp32 tensor of pre-sigmoid logits (FENCE channel)
        targets: (N,) fp32 tensor of {0, 1} GT mask values at the same pixels
        subcats: list of length B (one subcat per batch item, NOT per pixel) —
            returned UNexpanded; only used by `fit_per_subcategory_thresholds`
            which calls a separate pixel-aligned variant below.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_subcats: list[Optional[str]] = []
    for batch in val_dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=use_amp):
            out = model(x)
            lg = (out.refined_logits
                  if (use_refined and out.refined_logits is not None)
                  else out.mask_logits)
        # Squeeze channel dim if present, cast to fp32 for stable BCE
        if lg.dim() == 4 and lg.shape[1] == 1:
            lg = lg.squeeze(1)
        lg = lg.float()                                              # (B, H, W)
        y_f = y.float()                                              # (B, H, W)
        # Resize logit to GT resolution if val_inference_size mismatched
        if lg.shape[-2:] != y_f.shape[-2:]:
            lg = F.interpolate(lg.unsqueeze(1), size=y_f.shape[-2:],
                                 mode="bilinear",
                                 align_corners=False).squeeze(1)
        B, H, W = lg.shape
        flat_lg = lg.reshape(B, -1)
        flat_tg = y_f.reshape(B, -1)
        # Per-image subsample to bound CPU memory
        N_total = flat_lg.shape[1]
        K = min(max_pixels_per_image, N_total)
        if K < N_total:
            idx = torch.randint(0, N_total, (B, K), device=lg.device)
            flat_lg = flat_lg.gather(1, idx)
            flat_tg = flat_tg.gather(1, idx)
        all_logits.append(flat_lg.flatten().cpu())
        all_targets.append(flat_tg.flatten().cpu())
        # Subcats: one per image, used by the per-subcategory path which
        # re-runs a slimmer collection that DOES track subcat alignment.
        for m in batch["metadata"]:
            all_subcats.append(m.get("subcategory"))
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return logits, targets, all_subcats


def fit_temperature(
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    max_iter: int = 200,
    init_T: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Fit a single scalar T to minimize BCE(sigmoid(logits / T), targets).

    Args:
        val_logits: (N,) fp32 pre-sigmoid logits.
        val_targets: (N,) fp32 {0, 1} GT.
        max_iter: LBFGS iterations.
        init_T: starting value.
        logger: optional logger for pre/post NLL.

    Returns:
        Optimal T ∈ [0.1, 10.0] (clamped for sanity).
    """
    # Work on fp32 CPU — LBFGS doesn't need GPU for 1 scalar parameter.
    logits = val_logits.detach().float().cpu()
    targets = val_targets.detach().float().cpu()
    T = torch.nn.Parameter(torch.tensor(float(init_T)))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter,
                                    line_search_fn="strong_wolfe")

    def _nll() -> torch.Tensor:
        # BCE with logits / T; clamp T to a safe range so the optimizer
        # doesn't pursue a degenerate solution (T -> 0 sharpens infinitely).
        T_safe = T.clamp(min=0.05, max=20.0)
        scaled = logits / T_safe
        return F.binary_cross_entropy_with_logits(scaled, targets)

    nll_before = float(_nll().item())

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _nll()
        loss.backward()
        return loss

    try:
        optimizer.step(_closure)
    except Exception as e:
        if logger is not None:
            logger.warning(f"Temperature LBFGS failed ({type(e).__name__}: {e}); "
                            f"keeping T={init_T:.3f}")
        return float(init_T)

    T_final = float(T.detach().clamp(min=0.1, max=10.0).item())
    nll_after = float(F.binary_cross_entropy_with_logits(
        logits / T_final, targets,
    ).item())
    if logger is not None:
        logger.info(
            f"Temperature scaling fit: T={T_final:.4f}  "
            f"NLL {nll_before:.4f} -> {nll_after:.4f} "
            f"(delta={nll_before - nll_after:+.4f})"
        )
    return T_final


# ══════════════════════════════════════════════════════════════════════
# 2. Per-subcategory threshold tuning
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_val_probs_by_image(
    model: nn.Module,
    val_dl: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    use_refined: bool = True,
    temperature: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor, Optional[str]]]:
    """Forward val_dl, return per-image (prob, target, subcategory) tuples.

    Probs are post-sigmoid post-temperature, fp32, kept on CPU. We keep ONE
    entry per image (not per batch) so the threshold sweep can group correctly.
    """
    model.eval()
    per_image: list[tuple[torch.Tensor, torch.Tensor, Optional[str]]] = []
    for batch in val_dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=use_amp):
            out = model(x)
            lg = (out.refined_logits
                  if (use_refined and out.refined_logits is not None)
                  else out.mask_logits)
        if lg.dim() == 4 and lg.shape[1] == 1:
            lg = lg.squeeze(1)
        lg = lg.float() / max(1e-6, float(temperature))
        y_f = y.float()
        if lg.shape[-2:] != y_f.shape[-2:]:
            lg = F.interpolate(lg.unsqueeze(1), size=y_f.shape[-2:],
                                 mode="bilinear",
                                 align_corners=False).squeeze(1)
        probs = torch.sigmoid(lg).cpu()
        tgts = y_f.cpu()
        for i, m in enumerate(batch["metadata"]):
            per_image.append((probs[i], tgts[i], m.get("subcategory")))
    return per_image


def _iou_at_threshold(prob: torch.Tensor, target: torch.Tensor,
                       thr: float, eps: float = 1e-6) -> float:
    """Hard IoU between (prob >= thr) and binary target."""
    pred = (prob >= thr)
    gt = (target > 0.5)
    inter = float((pred & gt).sum().item())
    union = float((pred | gt).sum().item())
    if union <= 0.0:
        # Degenerate: empty pred AND empty GT → IoU undefined; return 1.0
        # so all-negative images don't drag the per-bucket optimum toward 0.
        return 1.0 if inter == 0.0 else 0.0
    return inter / (union + eps)


def fit_per_subcategory_thresholds(
    per_image: list[tuple[torch.Tensor, torch.Tensor, Optional[str]]],
    sweep_min: float = 0.20,
    sweep_max: float = 0.80,
    sweep_step: float = 0.025,
    min_count: int = 20,
    fallback_threshold: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> dict[str, float]:
    """Sweep thresholds per subcategory; return {subcat: best_threshold}.

    Subcategories with fewer than `min_count` samples fall back to the
    GLOBAL best threshold (sweep over ALL images), which is in turn fallback
    to `fallback_threshold` if the global sweep is also empty.

    Returns:
        Dict with keys = subcategory strings; one extra key '__global__' is
        always present and is the optimum threshold over the entire val set.
        Inference reads `out[subcat]` and falls back to `out['__global__']`
        and then to `fallback_threshold` if both are missing.
    """
    if not per_image:
        return {"__global__": float(fallback_threshold)}

    # Build threshold grid
    thr_grid: list[float] = []
    t = float(sweep_min)
    while t <= float(sweep_max) + 1e-9:
        thr_grid.append(round(t, 4))
        t += float(sweep_step)

    # Bucket images by subcategory
    buckets: dict[str, list[int]] = {}
    for i, (_, _, sc) in enumerate(per_image):
        key = sc if (sc is not None and str(sc).strip()) else "__unknown__"
        buckets.setdefault(key, []).append(i)

    def _best_for_indices(indices: list[int]) -> tuple[float, float]:
        if not indices:
            return float(fallback_threshold), 0.0
        best_thr = float(fallback_threshold)
        best_iou = -1.0
        for thr in thr_grid:
            ious = []
            for i in indices:
                prob, tgt, _ = per_image[i]
                ious.append(_iou_at_threshold(prob, tgt, thr))
            mean_iou = sum(ious) / len(ious)
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_thr = thr
        return best_thr, best_iou

    all_indices = list(range(len(per_image)))
    global_thr, global_iou = _best_for_indices(all_indices)
    result: dict[str, float] = {"__global__": float(global_thr)}
    if logger is not None:
        logger.info(
            f"Threshold sweep GLOBAL  N={len(all_indices)}  "
            f"best_thr={global_thr:.3f}  best_iou={global_iou:.4f}"
        )

    for key, indices in sorted(buckets.items()):
        if len(indices) < int(min_count):
            if logger is not None:
                logger.info(
                    f"Threshold sweep {key}  N={len(indices)} < min_count={min_count}"
                    f"  -> falling back to __global__={global_thr:.3f}"
                )
            continue
        bucket_thr, bucket_iou = _best_for_indices(indices)
        result[key] = float(bucket_thr)
        if logger is not None:
            delta = bucket_iou - global_iou
            logger.info(
                f"Threshold sweep {key:<24}  N={len(indices):5d}  "
                f"best_thr={bucket_thr:.3f}  iou={bucket_iou:.4f}  "
                f"(vs global {global_iou:.4f}, delta={delta:+.4f})"
            )
    return result


def lookup_threshold(thresholds: dict[str, float], subcategory: Optional[str],
                       default: float = 0.5) -> float:
    """Inference helper: pick the right threshold for a sample.

    Lookup order: explicit subcategory bucket -> __global__ -> default.
    """
    if isinstance(thresholds, dict):
        if subcategory and subcategory in thresholds:
            return float(thresholds[subcategory])
        if "__global__" in thresholds:
            return float(thresholds["__global__"])
    return float(default)
