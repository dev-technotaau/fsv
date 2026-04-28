"""training/infer.py — Inference for a trained checkpoint.

Loads any checkpoint produced by `training.train` (full or weights-only),
rebuilds the model, runs forward, optionally applies post-processing
(morphology / guided filter / DenseCRF), and writes binary masks +
overlays to disk.

Quick start:

    # Single image
    python -m training.infer \
        --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt \
        --input ./photo.jpg \
        --output ./pred.png

    # Folder of images
    python -m training.infer \
        --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt \
        --input ./test_images/ \
        --output ./predictions/ \
        --post-process

    # With TTA + post-processing (slowest, highest quality)
    python -m training.infer \
        --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt \
        --input ./test_images/ \
        --output ./predictions/ \
        --tta-scales 0.75 1.0 1.25 --tta-flip --post-process

The checkpoint stores its `backbone_name` and `image_size` in `meta` if it
was saved by `save_inference_only()`. If those are missing (e.g. a hand-built
checkpoint), pass `--config` to point at the original training YAML.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.config import TrainingConfig
from training.checkpoint import CheckpointManager
from training.model import build_model
from training.post_process import post_process, availability_report

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ══════════════════════════════════════════════════════════════════════
# Predictor wrapper
# ══════════════════════════════════════════════════════════════════════

class FencePredictor:
    def __init__(self,
                 checkpoint_path: str | Path,
                 config: Optional[TrainingConfig] = None,
                 device: Optional[str] = None,
                 amp_dtype: str = "bf16") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Resolve device (auto / cuda / cpu)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Peek at the checkpoint to recover meta (backbone_name, image_size)
        payload = torch.load(self.checkpoint_path, map_location="cpu",
                              weights_only=False)
        meta = payload.get("meta") or {}
        self.meta = meta

        if config is None:
            # Build a default config; override with whatever meta provides
            config = TrainingConfig()
            if "backbone_name" in meta:
                config.model.backbone_name = meta["backbone_name"]
            if "image_size" in meta:
                config.data.image_size = int(meta["image_size"])
        self.cfg = config

        # Build model + load weights
        self.model = build_model(config.model).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(
            payload["model"], strict=False,
        )
        if missing:
            print(f"[infer] {len(missing)} missing keys (first 3): {missing[:3]}")
        if unexpected:
            print(f"[infer] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")

        self.patch_size = int(getattr(self.model, "patch_size", 14))
        # AMP setup
        self.amp_dtype = (torch.bfloat16 if amp_dtype == "bf16"
                           else torch.float16 if amp_dtype == "fp16"
                           else None)

        # Cache normalization tensors on device
        self._mean = IMAGENET_MEAN.to(self.device)
        self._std = IMAGENET_STD.to(self.device)

    # ── Per-image API ────────────────────────────────────────────────

    @torch.no_grad()
    def predict_prob(self,
                      image: Image.Image,
                      image_size: Optional[int] = None,
                      tta_scales: tuple[float, ...] = (1.0,),
                      tta_flip: bool = False) -> np.ndarray:
        """Predict the per-pixel fence PROBABILITY (float32, [0,1]) at the
        original image resolution. No post-processing, no threshold.

        Use this when ensembling multiple checkpoints — average the probs
        across checkpoints, then threshold/post-process once at the end."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        W_orig, H_orig = image.size
        target = int(image_size or self.cfg.data.image_size)
        ps = self.patch_size
        target = max(ps * 4, int(round(target / ps)) * ps)

        img_resized = image.resize((target, target), Image.BILINEAR)
        x = torch.from_numpy(np.array(img_resized, dtype=np.uint8))
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(self.device, non_blocking=True)
        x = (x - self._mean) / self._std

        H_target = W_target = target
        accum = torch.zeros((1, H_target, W_target), device=self.device,
                              dtype=torch.float32)
        n = 0
        use_amp = (self.amp_dtype is not None and self.device.type == "cuda")
        for s in tta_scales:
            new_h = max(ps * 4, int(round(H_target * s / ps)) * ps)
            new_w = max(ps * 4, int(round(W_target * s / ps)) * ps)
            xs = (F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                                 align_corners=False)
                   if (new_h, new_w) != (H_target, W_target) else x)
            with torch.amp.autocast(device_type=self.device.type,
                                      dtype=self.amp_dtype or torch.float32,
                                      enabled=use_amp):
                out = self.model(xs)
                lg = (out.refined_logits if out.refined_logits is not None
                      else out.mask_logits)
                pf = torch.sigmoid(lg.squeeze(1)).float()
            if pf.shape[-2:] != (H_target, W_target):
                pf = F.interpolate(pf.unsqueeze(1), size=(H_target, W_target),
                                    mode="bilinear", align_corners=False).squeeze(1)
            accum += pf
            n += 1
            if tta_flip:
                xf = torch.flip(xs, dims=(-1,))
                with torch.amp.autocast(device_type=self.device.type,
                                          dtype=self.amp_dtype or torch.float32,
                                          enabled=use_amp):
                    outf = self.model(xf)
                    lgf = (outf.refined_logits if outf.refined_logits is not None
                           else outf.mask_logits)
                    pf = torch.sigmoid(lgf.squeeze(1)).float()
                pf = torch.flip(pf, dims=(-1,))
                if pf.shape[-2:] != (H_target, W_target):
                    pf = F.interpolate(pf.unsqueeze(1), size=(H_target, W_target),
                                        mode="bilinear", align_corners=False).squeeze(1)
                accum += pf
                n += 1
        prob_target = (accum / n).squeeze(0).cpu().numpy()

        # Upsample prob to original image resolution (PIL mode 'F' = fp32)
        prob_orig = np.array(
            Image.fromarray(prob_target.astype(np.float32), mode="F")
                  .resize((W_orig, H_orig), Image.BILINEAR)
        ).astype(np.float32)
        return prob_orig

    @torch.no_grad()
    def predict(self,
                 image: Image.Image,
                 image_size: Optional[int] = None,
                 tta_scales: tuple[float, ...] = (1.0,),
                 tta_flip: bool = False,
                 post: Optional[object] = None) -> np.ndarray:
        """Predict the binary fence mask for one PIL image.
        Returns (H_orig, W_orig) uint8 0/1 mask."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        prob_orig = self.predict_prob(
            image, image_size=image_size,
            tta_scales=tta_scales, tta_flip=tta_flip,
        )
        # Post-processing (morphology / guided filter / dense CRF)
        if post is not None and getattr(post, "enabled", False):
            mask = post_process(prob_orig, np.asarray(image), post)
        else:
            mask = (prob_orig >= 0.5).astype(np.uint8)
        return mask


def ensemble_predict(predictors: "list[FencePredictor]",
                      image: Image.Image,
                      image_size: Optional[int] = None,
                      tta_scales: tuple[float, ...] = (1.0,),
                      tta_flip: bool = False,
                      post: Optional[object] = None) -> np.ndarray:
    """Ensemble multiple FencePredictors by averaging probability maps,
    then thresholding (and optionally post-processing) once.

    All predictors must accept the same image; outputs are merged at
    pixel-prob level (not at the binary-mask level — averaging probs
    gives a sharper ensembled boundary than averaging binary masks)."""
    if not predictors:
        raise ValueError("ensemble_predict requires >= 1 predictor")
    if image.mode != "RGB":
        image = image.convert("RGB")
    accum: Optional[np.ndarray] = None
    for pred in predictors:
        prob = pred.predict_prob(
            image, image_size=image_size,
            tta_scales=tta_scales, tta_flip=tta_flip,
        )
        accum = prob if accum is None else (accum + prob)
    prob_avg = accum / float(len(predictors))   # type: ignore[arg-type]
    if post is not None and getattr(post, "enabled", False):
        return post_process(prob_avg, np.asarray(image), post)
    return (prob_avg >= 0.5).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def _save_outputs(image: Image.Image, mask: np.ndarray,
                   out_path: Path, save_overlay: bool) -> None:
    """Write the mask (PNG, 0/255) and an optional overlay visualization."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    if save_overlay:
        overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
        rgb = np.asarray(image.convert("RGB"))
        red = np.zeros_like(rgb)
        red[..., 0] = 255
        alpha = (mask[..., None] * 0.5).astype(np.float32)
        blended = (rgb.astype(np.float32) * (1 - alpha)
                   + red.astype(np.float32) * alpha).astype(np.uint8)
        Image.fromarray(blended).save(overlay_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run inference with a trained fence-segmentation checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, nargs="+",
                     help="Path(s) to best.pt / best_inference.pt. Pass MULTIPLE "
                          "checkpoints to ensemble — probability maps are averaged "
                          "before thresholding (sharper boundary than binary-vote "
                          "ensembles).")
    ap.add_argument("--input", required=True,
                     help="Single image path OR a directory of images.")
    ap.add_argument("--output", required=True,
                     help="Output PNG path (single image) OR output directory.")
    ap.add_argument("--config", default=None,
                     help="Optional training YAML (only needed if checkpoint "
                          "lacks meta — e.g. an old full-state ckpt).")
    ap.add_argument("--image-size", type=int, default=None,
                     help="Override inference image size (default: from ckpt meta).")
    ap.add_argument("--device", default=None, help="cuda | cpu | auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--tta-scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--post-process", action="store_true",
                     help="Apply morphology + guided filter post-processing.")
    ap.add_argument("--dense-crf", action="store_true",
                     help="Add DenseCRF (requires pydensecrf installed).")
    ap.add_argument("--save-overlay", action="store_true",
                     help="Also write a side-by-side overlay PNG.")
    args = ap.parse_args()

    cfg = (TrainingConfig.from_yaml(args.config) if args.config else None)

    # Build one or more predictors. Multiple = probability ensemble.
    ckpt_paths = list(args.checkpoint)
    predictors: list[FencePredictor] = []
    for cp in ckpt_paths:
        p = FencePredictor(
            checkpoint_path=cp, config=cfg, device=args.device,
            amp_dtype=args.amp_dtype if args.amp_dtype != "fp32" else "fp32",
        )
        predictors.append(p)
    if len(predictors) == 1:
        predictor = predictors[0]
        print(f"Predictor ready  (backbone={predictor.cfg.model.backbone_name}  "
              f"image_size={args.image_size or predictor.cfg.data.image_size}  "
              f"device={predictor.device})")
        if predictor.meta:
            print(f"  ckpt meta: {predictor.meta}")
    else:
        print(f"ENSEMBLE READY: {len(predictors)} checkpoints, probs averaged")
        for i, p in enumerate(predictors):
            print(f"  [{i}] {ckpt_paths[i]}  "
                  f"image_size={p.cfg.data.image_size}")
        # Use first predictor as the "primary" for shared cfg lookups
        predictor = predictors[0]

    # Build post-process config
    post_cfg = None
    if args.post_process or args.dense_crf:
        post_cfg = predictor.cfg.post
        post_cfg.enabled = True
        if args.dense_crf:
            post_cfg.use_dense_crf = True
        avail = availability_report()
        print(f"  post-processing availability: {avail}")

    # Resolve input -> list of paths
    in_path = Path(args.input)
    if in_path.is_file():
        paths = [in_path]
        single_out = Path(args.output)
        out_dir = None
    elif in_path.is_dir():
        paths = sorted(p for ext in SUPPORTED_EXT
                        for p in in_path.rglob(f"*{ext}"))
        single_out = None
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  found {len(paths)} images in {in_path}")
    else:
        print(f"ERROR: input does not exist: {in_path}", file=sys.stderr)
        return 2

    if not paths:
        print("ERROR: no images to process", file=sys.stderr)
        return 2

    t0 = time.time()
    n_ok = n_err = 0
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im.load()
                if len(predictors) == 1:
                    mask = predictor.predict(
                        im,
                        image_size=args.image_size,
                        tta_scales=tuple(args.tta_scales),
                        tta_flip=args.tta_flip,
                        post=post_cfg,
                    )
                else:
                    mask = ensemble_predict(
                        predictors, im,
                        image_size=args.image_size,
                        tta_scales=tuple(args.tta_scales),
                        tta_flip=args.tta_flip,
                        post=post_cfg,
                    )
            out_path = (single_out if single_out is not None
                        else out_dir / f"{p.stem}_mask.png")
            _save_outputs(im if im.mode == "RGB" else im.convert("RGB"),
                           mask, out_path, save_overlay=args.save_overlay)
            n_ok += 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-6)
                eta = (len(paths) - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1}/{len(paths)}]  {rate:.1f} img/s  eta={int(eta)}s")
        except Exception as e:
            n_err += 1
            print(f"  [{p.name}] failed: {type(e).__name__}: {str(e)[:120]}")
    elapsed = time.time() - t0
    print(f"\nDone. {n_ok} ok, {n_err} errors in {elapsed:.1f}s "
          f"({n_ok / max(elapsed, 1e-6):.1f} img/s)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
