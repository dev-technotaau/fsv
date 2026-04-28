"""tools/eval_checkpoint.py — Standalone evaluation against any split.

Loads a checkpoint, runs full validation against the named split, and writes
per-image + aggregate metrics to disk. Use this for:
  - benchmarking a finished training run against test/test_hq
  - comparing multiple checkpoints head-to-head
  - reproducing a paper-style metrics table

Quick start:

    # Eval phase 1 best on test split
    python -m tools.eval_checkpoint \
        --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt \
        --split test \
        --image-size 512

    # Eval phase 2 best on test_hq with TTA + post-processing
    python -m tools.eval_checkpoint \
        --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt \
        --split test_hq \
        --image-size 1024 --batch-size 2 \
        --tta-scales 0.75 1.0 1.25 --tta-flip --post-process

Outputs:
    eval_summary.json            aggregate metrics
    eval_per_image.jsonl         per-image IoU / Dice / boundary IoU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.config import TrainingConfig, PostProcessConfig
from training.checkpoint import CheckpointManager
from training.metrics import SegMetricsAccumulator
from training.model import build_model
from training.post_process import post_process, availability_report
from tools.dataset import (
    FenceDataset, phase1_val_aug, phase2_val_aug, seed_worker,
)


def _build_loader(split: str, splits_dir: Path, image_size: int,
                   batch_size: int, num_workers: int) -> DataLoader:
    aug = (phase1_val_aug(image_size) if image_size <= 768
           else phase2_val_aug(image_size))
    ds = FenceDataset(
        splits_dir / f"{split}.jsonl",
        splits_dir / f"{split}_masks.jsonl",
        transform=aug,
    )
    print(f"  loaded {split}: {len(ds):,} samples")

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "metadata": [b["metadata"] for b in batch],
        }

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


@torch.no_grad()
def evaluate(checkpoint: Path,
              split: str,
              splits_dir: Path,
              image_size: int,
              batch_size: int,
              num_workers: int,
              device: str,
              amp_dtype: str,
              tta_scales: tuple[float, ...],
              tta_flip: bool,
              use_post: bool,
              use_dense_crf: bool,
              out_dir: Path,
              config: Optional[TrainingConfig] = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    bundled_cfg = payload.get("config") or {}
    # Resolve config: explicit arg > bundled in checkpoint > defaults+meta
    if config is None and bundled_cfg:
        try:
            config = TrainingConfig.from_dict(bundled_cfg)
            print(f"  Using bundled config from checkpoint "
                  f"(decoder_dim={config.model.decoder_dim}, "
                  f"refinement_use_depth={config.model.refinement_use_depth})")
        except Exception as e:
            print(f"  WARN: bundled config failed to parse "
                  f"({type(e).__name__}); falling back to defaults")
            config = None
    if config is None:
        config = TrainingConfig()
        if "backbone_name" in meta:
            config.model.backbone_name = meta["backbone_name"]
    print(f"checkpoint   : {checkpoint}")
    print(f"backbone     : {config.model.backbone_name}")
    print(f"split        : {split}  image_size={image_size}  bs={batch_size}")
    print(f"tta_scales   : {tta_scales}  tta_flip={tta_flip}")
    print(f"post_process : {use_post}  dense_crf={use_dense_crf}")

    dev = torch.device(device)
    model = build_model(config.model).to(dev).eval()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)}")
    patch_size = int(getattr(model, "patch_size", 14))

    loader = _build_loader(split, splits_dir, image_size, batch_size, num_workers)

    post_cfg = None
    if use_post or use_dense_crf:
        post_cfg = PostProcessConfig(enabled=True)
        if use_dense_crf:
            post_cfg.use_dense_crf = True
        avail = availability_report()
        print(f"  post-processing availability: {avail}")

    accumulator = SegMetricsAccumulator(threshold=0.5, boundary_kernel=5)
    per_image_log = (out_dir / "eval_per_image.jsonl").open("w", encoding="utf-8")

    use_amp = amp_dtype in ("bf16", "fp16") and dev.type == "cuda"
    amp_t = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    n_done = 0
    t0 = time.time()
    for batch in loader:
        x = batch["image"].to(dev, non_blocking=True)
        y = batch["mask"].to(dev, non_blocking=True)

        H, W = x.shape[-2:]
        accum = torch.zeros((x.shape[0], H, W), device=dev, dtype=torch.float32)
        n = 0
        for s in tta_scales:
            new_h = max(patch_size * 4, int(round(H * s / patch_size)) * patch_size)
            new_w = max(patch_size * 4, int(round(W * s / patch_size)) * patch_size)
            xs = (F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                                 align_corners=False)
                   if (new_h, new_w) != (H, W) else x)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_t,
                                      enabled=use_amp):
                out = model(xs)
                lg = (out.refined_logits if out.refined_logits is not None
                      else out.mask_logits)
                pf = torch.sigmoid(lg.squeeze(1)).float()
            if pf.shape[-2:] != (H, W):
                pf = F.interpolate(pf.unsqueeze(1), size=(H, W),
                                    mode="bilinear", align_corners=False).squeeze(1)
            accum += pf
            n += 1
            if tta_flip:
                xf = torch.flip(xs, dims=(-1,))
                with torch.amp.autocast(device_type=dev.type, dtype=amp_t,
                                          enabled=use_amp):
                    outf = model(xf)
                    lgf = (outf.refined_logits if outf.refined_logits is not None
                           else outf.mask_logits)
                    pf = torch.sigmoid(lgf.squeeze(1)).float()
                pf = torch.flip(pf, dims=(-1,))
                if pf.shape[-2:] != (H, W):
                    pf = F.interpolate(pf.unsqueeze(1), size=(H, W),
                                        mode="bilinear",
                                        align_corners=False).squeeze(1)
                accum += pf
                n += 1
        probs = accum / n

        # Apply post-processing per-image (if enabled). DenseCRF/guided filter
        # need the original RGB; reconstruct from normalized tensor.
        if post_cfg is not None and post_cfg.enabled:
            # De-normalize image back to uint8 RGB
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            rgb = (x * std + mean).clamp(0, 1)
            rgb_u8 = (rgb * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            probs_np = probs.cpu().numpy()
            masks_np = np.zeros_like(probs_np, dtype=np.uint8)
            for i in range(probs.shape[0]):
                masks_np[i] = post_process(probs_np[i].astype(np.float32),
                                             rgb_u8[i], post_cfg)
            probs = torch.from_numpy(masks_np.astype(np.float32)).to(dev)

        sc_list = [md.get("subcategory") for md in batch["metadata"]]
        accumulator.update(probs, y, subcategories=sc_list)

        # Per-image rows
        for i in range(probs.shape[0]):
            iou_i = accumulator.per_image_iou[-(probs.shape[0] - i)]
            dice_i = accumulator.per_image_dice[-(probs.shape[0] - i)]
            md = batch["metadata"][i]
            per_image_log.write(json.dumps({
                "id": md.get("id"),
                "iou": iou_i, "dice": dice_i,
                "class": md.get("class"),
                "subcategory": md.get("subcategory"),
                "review_source": md.get("review_source"),
            }) + "\n")

        n_done += probs.shape[0]
        if n_done % 200 < probs.shape[0]:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  [{n_done}/{len(loader.dataset)}]  {rate:.1f} img/s")

    per_image_log.close()
    metrics = accumulator.compute()
    metrics["n_images"] = len(loader.dataset)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    metrics["images_per_second"] = round(
        len(loader.dataset) / max(metrics["wall_seconds"], 1e-6), 2,
    )
    metrics["checkpoint"] = str(checkpoint)
    metrics["split"] = split
    metrics["image_size"] = image_size
    metrics["tta_scales"] = list(tta_scales)
    metrics["tta_flip"] = tta_flip
    metrics["post_process"] = bool(use_post or use_dense_crf)

    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone checkpoint evaluation.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test",
                     choices=("val", "test", "val_hq", "test_hq"))
    ap.add_argument("--splits-dir", default="dataset/splits")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--tta-scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--post-process", action="store_true")
    ap.add_argument("--dense-crf", action="store_true")
    ap.add_argument("--config", default=None,
                     help="Optional YAML if checkpoint lacks meta")
    ap.add_argument("--out-dir", default="outputs/eval",
                     help="Where to write summary + per-image JSONL")
    args = ap.parse_args()

    cfg = (TrainingConfig.from_yaml(args.config) if args.config else None)
    evaluate(
        checkpoint=Path(args.checkpoint),
        split=args.split,
        splits_dir=Path(args.splits_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp_dtype=args.amp_dtype,
        tta_scales=tuple(args.tta_scales),
        tta_flip=args.tta_flip,
        use_post=args.post_process,
        use_dense_crf=args.dense_crf,
        out_dir=Path(args.out_dir),
        config=cfg,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
