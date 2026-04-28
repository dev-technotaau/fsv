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
    return mask


def availability_report() -> dict[str, bool]:
    return {
        "cv2": _HAS_CV2,
        "guided_filter (cv2.ximgproc)": _HAS_GUIDED,
        "dense_crf (pydensecrf)": _HAS_CRF,
    }
