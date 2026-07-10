"""CPU finisher — exact-swatch color-lock + mask composite (model-agnostic).

Takes the renovated fence (from Qwen) + the DINOv3 mask + the target swatch hex and:
  1. re-imposes the EXACT swatch colour on the renovated wood's LUMINANCE  -> guarantees dE<=3
  2. composites the fence back over the ORIGINAL photo                      -> background pixel-identical

The generative model only has to RENOVATE; exact colour + background correctness live here.
Pure numpy + OpenCV so it runs anywhere (no torch, no GPU).
"""
from __future__ import annotations
import numpy as np
import cv2


def _hex_to_lab(hexstr: str) -> np.ndarray:
    h = hexstr.lstrip("#")
    rgb = np.array([[[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]]], np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)  # cv2 8-bit LAB


def color_lock(fence_rgb: np.ndarray, mask: np.ndarray, swatch_hex: str,
               contrast: float = 1.0, chroma_retain: float = 0.0) -> np.ndarray:
    """Keep the renovated wood's luminance (its fresh grain/tone variation) but set the
    chroma to the EXACT swatch and re-centre the luminance on the swatch. The masked
    fence's median colour then equals the swatch (dE~0); per-pixel variation is the grain.

    contrast       : scales luminance variation around the swatch mean (1.0 = keep as-is).
    chroma_retain  : 0 = pure swatch hue (exact); small >0 keeps a touch of the render's chroma.
    """
    m = mask > 0.5
    if not m.any():
        return fence_rgb
    lab = cv2.cvtColor(fence_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    sL, sa, sb = _hex_to_lab(swatch_hex)
    med = float(np.median(L[m]))
    Ln = np.clip(sL + (L - med) * contrast, 0, 255)
    lab[..., 0] = np.where(m, Ln, L)
    lab[..., 1] = np.where(m, sa * (1 - chroma_retain) + a * chroma_retain, a)
    lab[..., 2] = np.where(m, sb * (1 - chroma_retain) + b * chroma_retain, b)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def composite(original_rgb: np.ndarray, fence_rgb: np.ndarray, mask: np.ndarray,
              feather_px: float = 2.0) -> np.ndarray:
    """fence pixels from fence_rgb, everything else pixel-identical to original_rgb."""
    m = (mask > 0.5).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px)[..., None]
    return np.clip(original_rgb.astype(np.float32) * (1 - m) +
                   fence_rgb.astype(np.float32) * m, 0, 255).astype(np.uint8)


def finish(original_rgb: np.ndarray, renovated_rgb: np.ndarray, mask: np.ndarray,
           swatch_hex: str, contrast: float = 1.0, chroma_retain: float = 0.0) -> np.ndarray:
    """Full finisher: resize render to original, color-lock the fence, composite over original.
    mask is a float [0,1] map at the ORIGINAL resolution."""
    H, W = original_rgb.shape[:2]
    if renovated_rgb.shape[:2] != (H, W):
        # The render is smaller than the photo -> UPSCALE with Lanczos to keep the wood grain sharp.
        # (INTER_AREA is an averaging downscale filter; using it to upscale blurs the grain.)
        upscaling = (H * W) > (renovated_rgb.shape[0] * renovated_rgb.shape[1])
        interp = cv2.INTER_LANCZOS4 if upscaling else cv2.INTER_AREA
        renovated_rgb = cv2.resize(renovated_rgb, (W, H), interpolation=interp)
    locked = color_lock(renovated_rgb, mask, swatch_hex, contrast=contrast, chroma_retain=chroma_retain)
    return composite(original_rgb, locked, mask)


def delta_e_median(out_rgb: np.ndarray, mask: np.ndarray, swatch_hex: str) -> float:
    """CIE76 dE of the masked fence's MEDIAN colour vs the swatch — i.e. 'is the central
    colour the swatch?'. (Measures colour accuracy, NOT the intentional per-pixel luminance
    spread of the grain, which would inflate a naive per-pixel median dE.)"""
    m = mask > 0.5
    if not m.any():
        return 99.0
    def to_real(x):  # cv2 LAB 0-255 -> real L*0-100, a*/b* -128..127
        return np.stack([x[..., 0] * 100 / 255, x[..., 1] - 128, x[..., 2] - 128], -1)
    lab = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[m]
    med = np.median(lab, axis=0)                       # median LAB colour of the fence
    d = to_real(med[None]) - to_real(_hex_to_lab(swatch_hex)[None])
    return float(np.sqrt((d ** 2).sum(-1))[0])
