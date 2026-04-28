"""PNG I/O + colorized visualization for multi-class segmentation masks.

Masks are stored as 8-bit single-channel PNGs where pixel value = class_id.
A separate "viz" PNG is saved for human review — colorized per schema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from annotation.schema import Schema


def save_class_mask(class_map: np.ndarray, out_path: Path) -> None:
    """Save class_map as 8-bit PNG. Values are class IDs — this is the
    training-ready label (NOT meant to look visually obvious). pixel value N
    means 'pixel belongs to class N'. For binary this means 0/1 values which
    look nearly identical to black in image viewers — that's expected."""
    if class_map.dtype != np.uint8:
        class_map = class_map.astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_map, mode="L").save(out_path, optimize=True)


def save_class_mask_preview(class_map: np.ndarray, out_path: Path,
                            stain_class_ids: set[int] | None = None) -> None:
    """Save a HUMAN-VIEWABLE B/W preview: stain-target pixels → white,
    everything else (background AND absorbers) → black. Matches exactly how
    the stainer will use the mask:
        stain_mask = np.isin(class_map, list(stain_class_ids))

    Purely for visual inspection. NOT the training label — downstream code
    should read from masks/ (raw class IDs), not masks_preview/.

    `stain_class_ids` — IDs of classes that should appear white. Defaults to
    {1} (fence_wood). For the 3-class schema this correctly EXCLUDES class 2
    (not_target absorber) from the preview, fixing the issue where absorbed
    walls/buildings/etc. previously showed up white alongside actual fence.
    """
    if stain_class_ids is None:
        stain_class_ids = {1}
    preview = (np.isin(class_map, list(stain_class_ids)).astype(np.uint8)) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(out_path, optimize=True)


def load_class_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im, dtype=np.uint8)


def save_colorized_viz(class_map: np.ndarray, schema: Schema,
                       out_path: Path, image: Image.Image | None = None,
                       blend_alpha: float = 0.55) -> None:
    """Emit a human-viewable PNG: colored mask, optionally alpha-blended over
    the original image. Used for QA review screens."""
    H, W = class_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in schema.classes:
        rgb[class_map == c.id] = c.color

    if image is not None:
        img_rgb = np.array(image.convert("RGB").resize((W, H)))
        # Alpha blend: where class_map > 0, blend; where == 0 (background), keep image.
        mask_fg = (class_map > 0).astype(np.float32)[..., None]
        blended = img_rgb * (1.0 - blend_alpha * mask_fg) + rgb * (blend_alpha * mask_fg)
        out = blended.clip(0, 255).astype(np.uint8)
    else:
        out = rgb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path, optimize=True)


def save_confidence_heatmap(conf_map: np.ndarray, out_path: Path) -> None:
    """Grayscale PNG where brighter = higher model confidence. Used for QA."""
    arr = (conf_map.clip(0.0, 1.0) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path, optimize=True)
