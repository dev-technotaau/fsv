"""Environment pre-flight: check data, checkpoints dir writable, GPU, disk space."""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config
from .exceptions import PreflightError


MIN_FREE_GB_RECOMMENDED = 50


def preflight(cfg: Config) -> list[str]:
    """Return list of warnings. Raises PreflightError on hard failures."""
    warnings: list[str] = []

    # Data dirs
    if not cfg.data.images_dir.exists():
        raise PreflightError(f"images_dir missing: {cfg.data.images_dir}")
    if not cfg.data.masks_dir.exists():
        raise PreflightError(f"masks_dir missing: {cfg.data.masks_dir}")
    n_images = sum(1 for p in cfg.data.images_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file())
    n_masks = sum(1 for p in cfg.data.masks_dir.iterdir() if p.suffix.lower() == ".png")
    if n_images == 0:
        raise PreflightError(f"No images found in {cfg.data.images_dir}")
    if n_masks == 0:
        raise PreflightError(f"No masks found in {cfg.data.masks_dir}")
    if abs(n_images - n_masks) > 0.2 * max(n_images, n_masks):
        warnings.append(
            f"image/mask count mismatch: {n_images} images vs {n_masks} masks"
        )

    # Output dir writable
    try:
        cfg.runtime.output_dir.mkdir(parents=True, exist_ok=True)
        probe = cfg.runtime.output_dir / ".preflight_probe"
        probe.write_text("ok")
        probe.unlink()
    except Exception as e:
        raise PreflightError(f"output_dir not writable: {cfg.runtime.output_dir}: {e}")

    # Disk space
    du = shutil.disk_usage(str(cfg.runtime.output_dir))
    free_gb = du.free / (1024 ** 3)
    if free_gb < MIN_FREE_GB_RECOMMENDED:
        warnings.append(
            f"Only {free_gb:.1f} GB free on output disk; "
            f"GA artifacts (checkpoints + ONNX exports) can easily exceed {MIN_FREE_GB_RECOMMENDED} GB."
        )

    # GPU
    try:
        import torch
        if cfg.runtime.n_gpus > 0 and not torch.cuda.is_available():
            warnings.append("Config requests GPUs but torch.cuda.is_available() == False. "
                           "Will fall back to CPU (training will be very slow).")
        elif cfg.runtime.n_gpus > torch.cuda.device_count():
            warnings.append(
                f"Config requests {cfg.runtime.n_gpus} GPUs but only "
                f"{torch.cuda.device_count()} are available."
            )
    except ImportError:
        warnings.append("PyTorch not installed — GPU pre-flight skipped.")

    return warnings
