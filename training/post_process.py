"""training/post_process.py — Inference-time mask cleanup.

Pipeline (cascade — apply in this order):
    1. Morphology       (closing -> opening) cleans tiny holes/specks
    2. Guided Filter    (cv2.ximgproc.guidedFilter) refines edges using image
    3. DenseCRF         (pydensecrf) refines pixel labels using image evidence

All stages are OFF by default; configurable via PostProcessConfig.
Each stage is best-effort — if its dependency isn't installed, that stage
just gets skipped with a warning.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Stage availability checks (run once at module import)
# ══════════════════════════════════════════════════════════════════════

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_HAS_GUIDED = False
if _HAS_CV2:
    try:
        from cv2 import ximgproc
        _ = ximgproc.guidedFilter
        _HAS_GUIDED = True
    except (ImportError, AttributeError):
        _HAS_GUIDED = False

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False


# ══════════════════════════════════════════════════════════════════════
# Individual stages
# ══════════════════════════════════════════════════════════════════════

def morphology_clean(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """mask: (H, W) uint8 0/1. Closing then opening with a kxk kernel."""
    if not _HAS_CV2:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m


def guided_filter_refine(prob: np.ndarray, image_rgb: np.ndarray,
                          radius: int = 4, eps: float = 1e-4) -> np.ndarray:
    """prob: (H, W) float32 in [0,1]. image_rgb: (H, W, 3) uint8.
    Returns refined prob (same shape, [0,1])."""
    if not _HAS_GUIDED:
        return prob
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    refined = ximgproc.guidedFilter(guide=gray, src=prob.astype(np.float32),
                                      radius=radius, eps=eps)
    return np.clip(refined, 0.0, 1.0)


def dense_crf(prob: np.ndarray, image_rgb: np.ndarray, *,
              iterations: int = 5,
              gauss_sxy: int = 3, gauss_compat: int = 3,
              bilateral_sxy: int = 80, bilateral_srgb: int = 13,
              bilateral_compat: int = 10) -> np.ndarray:
    """prob: (H, W) float32 in [0,1]. image_rgb: (H, W, 3) uint8.
    Returns refined binary mask (H, W) uint8 0/1."""
    if not _HAS_CRF:
        return (prob >= 0.5).astype(np.uint8)
    H, W = prob.shape
    soft = np.stack([1.0 - prob, prob], axis=0)              # (2, H, W)
    soft = soft * 0.9 + 0.05                                  # avoid degenerate
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary_from_softmax(soft.astype(np.float32)))
    d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
    d.addPairwiseBilateral(sxy=bilateral_sxy, srgb=bilateral_srgb,
                            rgbim=image_rgb.astype(np.uint8),
                            compat=bilateral_compat)
    Q = d.inference(iterations)
    out = np.argmax(np.array(Q).reshape(2, H, W), axis=0).astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════
# Connected-component cleanup
# ══════════════════════════════════════════════════════════════════════

def connected_component_clean(mask: np.ndarray, *,
                                min_blob_area: int = 200,
                                fill_holes_smaller_than: int = 0,
                                keep_top_k_blobs: int = 0,
                                ) -> np.ndarray:
    """Clean up speckle false-positives + (optionally) tiny holes.

    `min_blob_area`: drop any foreground blob smaller than N pixels — these
        are almost certainly false positives in trees/grass that look like
        wood. Default 200 (~14x14 pixels at 1024², ~7x7 at 512²).
    `fill_holes_smaller_than`: fill BACKGROUND holes inside the fence that
        are smaller than N pixels (e.g. occluder leftovers, picket-gap
        artifacts smaller than expected). 0 = disabled.
        WARNING: only enable if you're SURE you don't want to preserve
        small interior gaps (between fence slats, etc.).
    `keep_top_k_blobs`: if > 0, keep only the K largest foreground blobs.
        Use to reject everything except the dominant fence regions.
        0 = disabled (keep all blobs that pass min_blob_area).
    """
    if not _HAS_CV2:
        return mask
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)

    # Foreground connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # stats: [label_id, x, y, w, h, area]; label 0 is background
    keep_labels = []
    for lab in range(1, n_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_blob_area:
            keep_labels.append(lab)

    if keep_top_k_blobs > 0 and len(keep_labels) > keep_top_k_blobs:
        keep_labels = sorted(
            keep_labels,
            key=lambda l: stats[l, cv2.CC_STAT_AREA],
            reverse=True,
        )[:keep_top_k_blobs]

    out = np.zeros_like(m)
    for lab in keep_labels:
        out[labels == lab] = 255

    # Fill background holes smaller than threshold
    if fill_holes_smaller_than > 0:
        inv = 255 - out
        n_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for lab in range(1, n_h):
            # Skip the giant outer-background component — only fill INTERIOR holes.
            x, y, w, h = (h_stats[lab, cv2.CC_STAT_LEFT], h_stats[lab, cv2.CC_STAT_TOP],
                           h_stats[lab, cv2.CC_STAT_WIDTH], h_stats[lab, cv2.CC_STAT_HEIGHT])
            touches_edge = (x == 0 or y == 0
                            or x + w >= out.shape[1] or y + h >= out.shape[0])
            if touches_edge:
                continue
            if h_stats[lab, cv2.CC_STAT_AREA] < fill_holes_smaller_than:
                out[h_labels == lab] = 255

    return (out > 0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Cascade
# ══════════════════════════════════════════════════════════════════════

def post_process(prob: np.ndarray, image_rgb: np.ndarray, cfg) -> np.ndarray:
    """Apply the configured cascade. Returns binary mask (H, W) uint8 0/1."""
    if not cfg.enabled:
        return (prob >= 0.5).astype(np.uint8)

    refined = prob.astype(np.float32).copy()

    # Stage A: guided filter (smooth boundaries with image guidance)
    if cfg.use_guided_filter:
        if _HAS_GUIDED:
            refined = guided_filter_refine(refined, image_rgb,
                                            radius=cfg.guided_filter_radius,
                                            eps=cfg.guided_filter_eps)
        else:
            warnings.warn("guided filter requested but cv2.ximgproc not installed",
                           RuntimeWarning)

    # Stage B: DenseCRF (final pixel labeling)
    if cfg.use_dense_crf:
        if _HAS_CRF:
            mask = dense_crf(refined, image_rgb,
                              iterations=cfg.crf_iterations,
                              gauss_sxy=cfg.crf_gauss_sxy,
                              gauss_compat=cfg.crf_gauss_compat,
                              bilateral_sxy=cfg.crf_bilateral_sxy,
                              bilateral_srgb=cfg.crf_bilateral_srgb,
                              bilateral_compat=cfg.crf_bilateral_compat)
        else:
            warnings.warn("dense_crf requested but pydensecrf not installed",
                           RuntimeWarning)
            mask = (refined >= 0.5).astype(np.uint8)
    else:
        mask = (refined >= 0.5).astype(np.uint8)

    # Stage C: morphology cleanup
    if cfg.use_morphology:
        if _HAS_CV2:
            mask = morphology_clean(mask, kernel_size=cfg.morphology_kernel)
        else:
            warnings.warn("morphology requested but cv2 not installed",
                           RuntimeWarning)
    # Stage D: connected-component cleanup (fence-domain post-process)
    use_cc = bool(getattr(cfg, "use_cc_cleanup", False))
    if use_cc:
        if _HAS_CV2:
            mask = connected_component_clean(
                mask,
                min_blob_area=int(getattr(cfg, "cc_min_blob_area", 200)),
                fill_holes_smaller_than=int(
                    getattr(cfg, "cc_fill_holes_smaller_than", 0)
                ),
                keep_top_k_blobs=int(getattr(cfg, "cc_keep_top_k_blobs", 0)),
            )
        else:
            warnings.warn("CC cleanup requested but cv2 not installed",
                           RuntimeWarning)
    return mask


def availability_report() -> dict[str, bool]:
    return {
        "cv2": _HAS_CV2,
        "guided_filter (cv2.ximgproc)": _HAS_GUIDED,
        "dense_crf (pydensecrf)": _HAS_CRF,
    }
