"""Combo 15: Cascade — SAM 2 (coarse) → SegFormer-B5 refiner. FULL.

Stage 1 (coarse): Run SAM 2 auto-mask with full-image box prompt to produce a
coarse fence mask. This requires SAM 2 (or SAM v1 as fallback).

Stage 2 (refiner): Train a lightweight refiner that takes [image (3ch) + coarse_mask (1ch)]
as 4-channel input and produces a refined mask. Consistency loss penalizes
deviation from the coarse prior (keeps refiner grounded).

Requires `transformers` for SegFormer. SAM optional (graceful degrade to
image-only refinement if unavailable).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import AdapterResult, ModelAdapter
from ._common import FenceDataset, file_logger, pick_device


class CascadeSam2SegFormerAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, random_split
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        input_size = int(self.params.get("input_size", 512))
        input_size = (input_size // 32) * 32

        # ---------- Stage 1: pre-compute coarse masks from SAM ----------
        from .combo_04_sam2_full_finetune import _build_sam_predictor
        _, auto_gen, info = _build_sam_predictor(
            log, device,
            points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95,
        )
        use_sam = auto_gen is not None
        log(f"Stage 1 (SAM coarse): using {'SAM' if use_sam else 'zero-mask fallback'} — {info}")

        # ---------- Stage 2: refiner SegFormer-B5 (4-channel input) ----------
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"transformers required: {e}")
        refiner_base = "nvidia/segformer-b5-finetuned-ade-640-640"
        try:
            cfg = SegformerConfig.from_pretrained(refiner_base, num_labels=2,
                                                   ignore_mismatched_sizes=True)
            model = SegformerForSemanticSegmentation.from_pretrained(
                refiner_base, config=cfg, ignore_mismatched_sizes=True,
            )
            # Expand first conv to 4 channels (RGB + coarse_mask)
            _expand_first_conv_to_4ch(model)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"SegFormer-B5 load failed: {e}")
        model.to(device).train()

        # Build paired dataset: image + coarse_mask
        ds = _CoarseMaskDataset(
            images_dir=self.data_cfg.images_dir,
            masks_dir=self.data_cfg.masks_dir,
            input_size=input_size,
            auto_gen=auto_gen,
            log=log,
            cache_dir=self.work_dir / "coarse_cache",
            sam_box_margin=int(self.params.get("sam_box_margin", 10)),
        )
        val_n = max(2, int(len(ds) * self.data_cfg.val_split))
        gen = torch.Generator().manual_seed(self.data_cfg.seed)
        train_ds, val_ds = random_split(ds, [len(ds) - val_n, val_n], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0)

        lr = float(self.params.get("refiner_lr", 5e-5))
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        consistency_w = float(self.params.get("consistency_weight", 0.3))

        for ep in range(self.fitness_cfg.proxy_epochs):
            model.train()
            ep_loss, n = 0.0, 0
            for imgs_4ch, masks in train_loader:
                imgs_4ch, masks = imgs_4ch.to(device), masks.to(device)
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    out = model(pixel_values=imgs_4ch)
                    logits = F.interpolate(out.logits, size=input_size,
                                           mode="bilinear", align_corners=False)
                    ce = F.cross_entropy(logits, masks)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    # consistency to coarse mask (channel 3 of input)
                    coarse = imgs_4ch[:, 3]
                    consistency = F.l1_loss(probs, coarse.float())
                    loss = ce + consistency_w * consistency
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                ep_loss += loss.item(); n += 1
            log(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f}")

            # early kill
            early_kill_at = int(self.fitness_cfg.proxy_epochs * self.fitness_cfg.early_kill_at_fraction)
            if ep + 1 == early_kill_at:
                iou = _eval_cascade_iou(model, val_loader, device, input_size)
                log(f"ep {ep} val_iou={iou:.4f}")
                if iou < self.fitness_cfg.early_kill_iou:
                    return AdapterResult(metrics={"iou": iou}, status="killed_early")

        # final eval
        preds_dir = self.work_dir / "preds"; gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)
        model.eval()
        with torch.no_grad():
            for bi, (imgs_4ch, masks) in enumerate(val_loader):
                imgs_4ch = imgs_4ch.to(device)
                out = model(pixel_values=imgs_4ch)
                logits = F.interpolate(out.logits, size=input_size, mode="bilinear", align_corners=False)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                masks_np = masks.cpu().numpy()
                for j in range(probs.shape[0]):
                    binary = (probs[j] > 0.5).astype("uint8") * 255
                    Image.fromarray(binary).save(preds_dir / f"val_{bi:04d}_{j}.png")
                    Image.fromarray((masks_np[j].astype("uint8") * 255)).save(
                        gt_dir / f"val_{bi:04d}_{j}.png"
                    )
        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"final metrics: {metrics}")
        ckpt = self.work_dir / "refiner.pt"
        import torch as _t
        _t.save({"model_state": model.state_dict(), "params": self.params,
                 "combo_key": self.combo_key}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)


def _expand_first_conv_to_4ch(model):
    """Expand SegFormer's first conv from 3→4 channels (RGB + coarse prior)."""
    import torch
    import torch.nn as nn
    # SegFormer first conv: model.segformer.encoder.patch_embeddings[0].proj
    first_conv = model.segformer.encoder.patch_embeddings[0].proj
    old_w = first_conv.weight.data
    in_ch_old = old_w.shape[1]
    if in_ch_old == 4:
        return
    new_conv = nn.Conv2d(4, first_conv.out_channels,
                         kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride,
                         padding=first_conv.padding, bias=first_conv.bias is not None)
    # Copy RGB weights; init coarse-mask channel as average of RGB (warm start)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_w
        new_conv.weight[:, 3:4] = old_w.mean(dim=1, keepdim=True) * 0.1
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new_conv


def _eval_cascade_iou(model, loader, device, input_size):
    import torch
    import torch.nn.functional as F
    model.eval()
    ious = []
    with torch.no_grad():
        for imgs_4ch, masks in loader:
            imgs_4ch, masks = imgs_4ch.to(device), masks.to(device)
            out = model(pixel_values=imgs_4ch)
            logits = F.interpolate(out.logits, size=input_size, mode="bilinear", align_corners=False)
            pred = torch.softmax(logits, dim=1)[:, 1] > 0.5
            inter = (pred & masks.bool()).sum(dim=(1, 2))
            union = (pred | masks.bool()).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


class _CoarseMaskDataset:
    """Yields (4-channel_tensor, binary_gt_mask). Channel 3 is the SAM coarse mask
    (or zeros if SAM unavailable). Coarse masks are cached on disk for reuse."""
    def __init__(self, images_dir, masks_dir, input_size, auto_gen, log, cache_dir: Path, sam_box_margin):
        self.base = FenceDataset(images_dir, masks_dir, input_size, augment="none")
        self.auto_gen = auto_gen
        self.log = log
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size
        self.sam_box_margin = sam_box_margin

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        import torch
        img_t, mask_t = self.base[idx]
        # Unnormalize for SAM inference
        img_pil_np = self._unnormalize(img_t)
        coarse = self._get_coarse(idx, img_pil_np)
        coarse_t = torch.from_numpy(coarse.astype("float32")).unsqueeze(0)
        return torch.cat([img_t, coarse_t], dim=0), mask_t

    def _unnormalize(self, t):
        import torch
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        arr = (t * std + mean).clamp(0, 1).numpy()
        return (arr.transpose(1, 2, 0) * 255).astype("uint8")

    def _get_coarse(self, idx, img_np):
        cache_path = self.cache_dir / f"coarse_{idx:06d}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        if self.auto_gen is None:
            coarse = np.zeros((self.input_size, self.input_size), dtype="uint8")
        else:
            try:
                masks = self.auto_gen.generate(img_np)
                from .combo_04_sam2_full_finetune import _combine_masks_by_fence_overlap
                coarse = _combine_masks_by_fence_overlap(masks, img_np).astype("uint8")
            except Exception as e:
                self.log(f"SAM coarse failed for {idx}: {e}; using zeros")
                coarse = np.zeros((self.input_size, self.input_size), dtype="uint8")
        np.save(cache_path, coarse)
        return coarse
