"""Combo 04: SAM 2 full — auto-mask inference + proper per-image prompt-encoded
mask-decoder fine-tuning.

Training loop:
  1. For each image in the batch:
     a. Derive a bounding box prompt from GT fence mask (with configurable jitter).
     b. Optionally sample point prompts inside GT (positive) and outside (negative).
     c. Run SAM's image encoder (frozen) → image_embedding.
     d. Run SAM's prompt encoder (frozen) on box + points → sparse/dense embeddings.
     e. Run SAM's mask decoder (TRAINABLE) → low-res mask logits.
     f. Upsample to input_size, compute loss vs GT, backprop to decoder only.
  2. Every N steps: EMA update on decoder params.

Inference:
  Auto-mask-generator with grid prompts, filter + merge masks by fence overlap.

Supports SAM 2 (preferred) and SAM v1 (fallback via segment_anything).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import AdapterResult, ModelAdapter
from ._common import FenceDataset, EMAHelper, file_logger, pick_device


class Sam2FullFinetuneAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Load SAM with prompt+mask+image encoders separable
        sam_ctx = _load_sam_with_encoders(
            log, device,
            points_per_side=int(self.params.get("points_per_side", 32)),
            pred_iou_thresh=float(self.params.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(self.params.get("stability_score_thresh", 0.95)),
        )
        if sam_ctx is None:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error="No SAM available (neither sam2 nor segment_anything).")
        (sam_model, image_encoder, prompt_encoder, mask_decoder,
         auto_gen, model_input_size, info) = sam_ctx
        log(f"SAM loaded: {info}  input_size={model_input_size}")

        # ---------- fine-tune decoder ----------
        dec_lr = float(self.params.get("decoder_lr", 1e-5))
        proxy_epochs = self.fitness_cfg.proxy_epochs
        if dec_lr > 0 and proxy_epochs > 0:
            best_iou, killed = _finetune_decoder(
                sam_model=sam_model,
                image_encoder=image_encoder,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                data_cfg=self.data_cfg,
                fitness_cfg=self.fitness_cfg,
                input_size=model_input_size,
                device=device,
                lr=dec_lr,
                epochs=proxy_epochs,
                log=log,
                box_jitter=int(self.params.get("sam_box_margin", 10)),
                point_samples=3,
            )
            if killed:
                return AdapterResult(metrics={"iou": best_iou}, status="killed_early")
            log(f"fine-tune best_val_iou={best_iou:.4f}")

        # ---------- inference with auto-mask + fence-aware fusion ----------
        ds = FenceDataset(self.data_cfg.images_dir, self.data_cfg.masks_dir,
                          input_size=model_input_size, augment="none")
        val_n = max(2, int(len(ds) * self.data_cfg.val_split))
        val_pairs = ds.pairs[-val_n:]

        preds_dir = self.work_dir / "preds"
        gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)

        for i, (img_p, mask_p) in enumerate(val_pairs):
            img = np.array(Image.open(img_p).convert("RGB").resize(
                (model_input_size, model_input_size), Image.BILINEAR))
            gt = np.array(Image.open(mask_p).convert("L").resize(
                (model_input_size, model_input_size), Image.NEAREST)) > 127
            if auto_gen is not None:
                try:
                    masks = auto_gen.generate(img)
                except Exception as e:
                    log(f"auto_gen failed on {img_p.name}: {e}")
                    masks = []
                combined = _combine_masks_by_fence_overlap(masks, img)
            else:
                combined = np.zeros_like(gt, dtype=bool)
            Image.fromarray((combined.astype("uint8") * 255)).save(
                preds_dir / f"val_{i:04d}.png")
            Image.fromarray((gt.astype("uint8") * 255)).save(
                gt_dir / f"val_{i:04d}.png")

        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"final metrics (auto-mask inference): {metrics}")

        # Save decoder state
        ckpt = self.work_dir / "mask_decoder.pt"
        import torch as _t
        _t.save({
            "mask_decoder_state": mask_decoder.state_dict(),
            "params": self.params,
            "combo_key": self.combo_key,
            "info": info,
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt,
                             extra={"backend": info})


# ============================================================
# SAM loading
# ============================================================

def _load_sam_with_encoders(log, device, *, points_per_side, pred_iou_thresh, stability_score_thresh):
    """Try SAM 2 first. Return tuple or None.

    Returns: (sam_model, image_encoder, prompt_encoder, mask_decoder,
              auto_mask_generator_or_None, input_size, info_str)
    """
    import torch
    # ---------- SAM 2 ----------
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        ckpt_dir = Path.home() / ".cache" / "sam2"
        ckpt_path = ckpt_dir / "sam2_hiera_large.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint missing at {ckpt_path}. "
                f"Download from github.com/facebookresearch/sam2/releases.")
        model = build_sam2("sam2_hiera_l.yaml", str(ckpt_path), device=str(device))
        # SAM 2's structure differs from v1: it has sam_image_encoder, sam_prompt_encoder, sam_mask_decoder
        # (internal API may evolve; handle common names)
        image_encoder  = getattr(model, "image_encoder",  None) or getattr(model, "sam_image_encoder",  None)
        prompt_encoder = getattr(model, "prompt_encoder", None) or getattr(model, "sam_prompt_encoder", None)
        mask_decoder   = getattr(model, "mask_decoder",   None) or getattr(model, "sam_mask_decoder",   None)
        if image_encoder is None or prompt_encoder is None or mask_decoder is None:
            raise RuntimeError("SAM2 model lacks expected encoder/decoder attrs.")
        auto = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return (model, image_encoder, prompt_encoder, mask_decoder, auto, 1024, "SAM 2 Hiera-L")
    except Exception as e:
        log(f"SAM 2 unavailable ({e}); trying SAM v1")

    # ---------- SAM v1 ----------
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        ckpt_dir = Path.home() / ".cache" / "sam"
        candidates = [
            ckpt_dir / "sam_vit_l_0b3195.pth",
            ckpt_dir / "sam_vit_h_4b8939.pth",
            ckpt_dir / "sam_vit_b_01ec64.pth",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            raise FileNotFoundError(f"Place a SAM v1 ckpt in {ckpt_dir}")
        mtype = "vit_l" if "vit_l" in ckpt.name else ("vit_h" if "vit_h" in ckpt.name else "vit_b")
        sam = sam_model_registry[mtype](checkpoint=str(ckpt)).to(device)
        image_encoder  = sam.image_encoder
        prompt_encoder = sam.prompt_encoder
        mask_decoder   = sam.mask_decoder
        auto = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return (sam, image_encoder, prompt_encoder, mask_decoder, auto, 1024, f"SAM v1 {mtype}")
    except Exception as e:
        log(f"SAM v1 unavailable ({e})")
        return None


# ============================================================
# DECODER FINE-TUNE
# ============================================================

def _finetune_decoder(
    *,
    sam_model,
    image_encoder,
    prompt_encoder,
    mask_decoder,
    data_cfg,
    fitness_cfg,
    input_size: int,
    device,
    lr: float,
    epochs: int,
    log,
    box_jitter: int,
    point_samples: int,
):
    """Train SAM's mask decoder on fence masks with per-image prompts.

    Key implementation notes:
      - image_encoder + prompt_encoder are FROZEN
      - mask_decoder is TRAINABLE
      - each sample produces its own prompts (box from GT bbox + random points)
      - loss: BCE (on per-mask logits) + Dice — the two that SAM was trained with
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    # Freeze everything except mask_decoder
    for p in sam_model.parameters():
        p.requires_grad_(False)
    for p in mask_decoder.parameters():
        p.requires_grad_(True)

    ds = _SamPromptDataset(
        data_cfg.images_dir, data_cfg.masks_dir,
        input_size=input_size, box_jitter=box_jitter, point_samples=point_samples,
    )
    if len(ds) < 10:
        log("dataset too small for SAM fine-tune; skipping")
        return 0.0, False
    val_n = max(2, int(len(ds) * data_cfg.val_split))
    gen = torch.Generator().manual_seed(data_cfg.seed)
    train_ds, val_ds = random_split(ds, [len(ds) - val_n, val_n], generator=gen)
    # Batch size 1 — SAM prompt encoding is simpler per-image
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=_collate_sam)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0,
                              collate_fn=_collate_sam)

    optim = torch.optim.AdamW(mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = EMAHelper(mask_decoder, decay=0.999)
    early_kill_at = int(epochs * fitness_cfg.early_kill_at_fraction)
    best_iou = 0.0
    pixel_mean = getattr(sam_model, "pixel_mean", torch.tensor([123.675, 116.28, 103.53])).to(device)
    pixel_std  = getattr(sam_model, "pixel_std",  torch.tensor([58.395,  57.12,  57.375])).to(device)
    pixel_mean = pixel_mean.view(1, 3, 1, 1)
    pixel_std  = pixel_std.view(1, 3, 1, 1)

    for ep in range(epochs):
        mask_decoder.train()
        image_encoder.eval(); prompt_encoder.eval()
        ep_loss, n = 0.0, 0
        for sample in train_loader:
            img = sample["image"].to(device)            # [1, 3, 1024, 1024] in [0, 255]
            gt_mask = sample["mask"].to(device)         # [1, 1024, 1024] binary
            box = sample["box"].to(device)              # [1, 4] xyxy
            points = sample["points"].to(device)        # [1, P, 2]
            point_labels = sample["point_labels"].to(device)  # [1, P]

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                # Normalize + encode image (frozen, no grad)
                with torch.no_grad():
                    img_norm = (img - pixel_mean) / pixel_std
                    image_embeddings = image_encoder(img_norm)
                    # SAM 2 may return dict or tuple; extract embedding
                    image_embeddings = _extract_image_embedding(image_embeddings)

                    # Prompt encoding (frozen, no grad)
                    sparse_embeddings, dense_embeddings = _encode_prompts(
                        prompt_encoder,
                        points=(points, point_labels) if points.shape[1] > 0 else None,
                        boxes=box,
                        masks=None,
                    )
                # Mask decoder (trainable)
                low_res_logits, iou_preds = _run_mask_decoder(
                    mask_decoder=mask_decoder,
                    image_embeddings=image_embeddings,
                    image_pe=_get_image_pe(prompt_encoder),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Upsample to image resolution
                mask_logits = F.interpolate(low_res_logits, size=input_size,
                                            mode="bilinear", align_corners=False)
                # Loss: BCE + Dice
                gt_flt = gt_mask.float().unsqueeze(1)
                bce = F.binary_cross_entropy_with_logits(mask_logits, gt_flt)
                probs = torch.sigmoid(mask_logits)
                inter = (probs * gt_flt).sum((2, 3))
                union = probs.sum((2, 3)) + gt_flt.sum((2, 3))
                dice = (1 - (2 * inter + 1) / (union + 1)).mean()
                loss = bce + dice

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ema.update(mask_decoder)
            ep_loss += loss.item(); n += 1

        log(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f}")

        # Validation — use EMA weights
        if ep + 1 == early_kill_at or ep == epochs - 1:
            ema.swap_in(mask_decoder)
            val_iou = _eval_sam_decoder(
                mask_decoder, image_encoder, prompt_encoder,
                val_loader, device, input_size, pixel_mean, pixel_std,
            )
            ema.swap_out(mask_decoder)
            log(f"ep {ep} val_iou={val_iou:.4f}")
            if ep + 1 == early_kill_at and val_iou < fitness_cfg.early_kill_iou:
                return val_iou, True
            best_iou = max(best_iou, val_iou)

    # Keep EMA weights for downstream inference
    ema.swap_in(mask_decoder)
    return best_iou, False


def _extract_image_embedding(x):
    """SAM 2 may return dict; v1 returns tensor. Normalize to tensor."""
    if isinstance(x, dict):
        for k in ("vision_features", "image_embed", "last_hidden_state"):
            if k in x:
                return x[k]
        return next(iter(x.values()))
    if isinstance(x, (list, tuple)):
        return x[-1]
    return x


def _get_image_pe(prompt_encoder):
    """Get the positional encoding for the image embedding."""
    if hasattr(prompt_encoder, "get_dense_pe"):
        return prompt_encoder.get_dense_pe()
    # SAM 2 uses different naming — try common alternates
    for attr in ("dense_pe", "image_pe"):
        if hasattr(prompt_encoder, attr):
            obj = getattr(prompt_encoder, attr)
            return obj() if callable(obj) else obj
    # Fallback: zeros — will still train but worse
    import torch
    return torch.zeros(1, 256, 64, 64)


def _encode_prompts(prompt_encoder, points, boxes, masks):
    """Call SAM's prompt encoder with graceful signature fallback."""
    try:
        sparse, dense = prompt_encoder(
            points=points, boxes=boxes, masks=masks,
        )
        return sparse, dense
    except TypeError:
        # SAM 2 alternate signature
        out = prompt_encoder(points=points, boxes=boxes, masks=masks)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out["sparse_embeddings"], out["dense_embeddings"]


def _run_mask_decoder(
    *, mask_decoder, image_embeddings, image_pe,
    sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output: bool,
):
    """Call mask decoder with sensible signature fallback across SAM versions."""
    try:
        low_res_logits, iou_preds = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_logits, iou_preds
    except TypeError:
        # SAM 2 may use extra args (repeat_image, high_res_features, etc.)
        out = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
        )
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        return out["masks"], out["iou_predictions"]


def _eval_sam_decoder(mask_decoder, image_encoder, prompt_encoder,
                     val_loader, device, input_size, pixel_mean, pixel_std) -> float:
    import torch
    import torch.nn.functional as F
    mask_decoder.eval(); image_encoder.eval(); prompt_encoder.eval()
    ious = []
    with torch.no_grad():
        for sample in val_loader:
            img = sample["image"].to(device)
            gt = sample["mask"].to(device)
            box = sample["box"].to(device)
            points = sample["points"].to(device)
            point_labels = sample["point_labels"].to(device)
            img_norm = (img - pixel_mean) / pixel_std
            image_embeddings = _extract_image_embedding(image_encoder(img_norm))
            sparse, dense = _encode_prompts(
                prompt_encoder,
                points=(points, point_labels) if points.shape[1] > 0 else None,
                boxes=box, masks=None,
            )
            low_res, _ = _run_mask_decoder(
                mask_decoder=mask_decoder, image_embeddings=image_embeddings,
                image_pe=_get_image_pe(prompt_encoder),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            mask_logits = F.interpolate(low_res, size=input_size, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(mask_logits) > 0.5).squeeze(1)
            inter = (pred & gt.bool()).sum(dim=(1, 2))
            union = (pred | gt.bool()).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


# ============================================================
# Prompt dataset
# ============================================================

class _SamPromptDataset:
    """Returns sample dict: image (uint8-like scaled to [0,255]), mask, box, points, point_labels."""
    def __init__(self, images_dir, masks_dir, input_size, box_jitter, point_samples):
        self.base = FenceDataset(images_dir, masks_dir, input_size, augment="medium")
        self.input_size = input_size
        self.box_jitter = box_jitter
        self.point_samples = point_samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import torch
        img_t, mask_t = self.base[idx]
        # Un-normalize ImageNet → 0..255 (SAM's own normalization will re-apply)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img_t * std + mean).clamp(0, 1) * 255.0   # [3, H, W] in [0, 255]

        mask_np = mask_t.numpy().astype(bool)
        box = _bbox_from_mask(mask_np, self.input_size, self.box_jitter)
        pts, lbls = _sample_points_in_mask(mask_np, n_pos=self.point_samples,
                                           n_neg=max(1, self.point_samples // 2))

        return {
            "image":        img,                                        # [3, H, W], float
            "mask":         mask_t,                                     # [H, W], int64
            "box":          torch.tensor(box, dtype=torch.float32),     # [4]
            "points":       torch.tensor(pts, dtype=torch.float32),     # [P, 2]
            "point_labels": torch.tensor(lbls, dtype=torch.int32),      # [P]
        }


def _collate_sam(batch):
    """Collate for SAM: stack per-key. batch_size=1 is easiest."""
    import torch
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            # pad points to same count
            if k == "points" or k == "point_labels":
                max_p = max(v.shape[0] for v in vals)
                padded = []
                for v in vals:
                    if v.shape[0] < max_p:
                        pad_shape = list(v.shape)
                        pad_shape[0] = max_p - v.shape[0]
                        pad_val = 0 if v.dtype in (torch.int32, torch.int64) else 0.0
                        padded.append(torch.cat([v, torch.full(pad_shape, pad_val, dtype=v.dtype)], dim=0))
                    else:
                        padded.append(v)
                out[k] = torch.stack(padded, dim=0)
            else:
                out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    # box needs extra batch dim: [B, 1, 4] for SAM's prompt_encoder? No — [B, 4] works.
    return out


def _bbox_from_mask(mask: np.ndarray, input_size: int, jitter: int) -> tuple[float, float, float, float]:
    """Return jittered bbox (x0, y0, x1, y1). Falls back to full image if mask empty."""
    if not mask.any():
        return (0.0, 0.0, float(input_size), float(input_size))
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # random jitter outward
    j = max(0, jitter)
    x0 = max(0, x0 - random.randint(0, j))
    y0 = max(0, y0 - random.randint(0, j))
    x1 = min(input_size - 1, x1 + random.randint(0, j))
    y1 = min(input_size - 1, y1 + random.randint(0, j))
    return (float(x0), float(y0), float(x1), float(y1))


def _sample_points_in_mask(mask: np.ndarray, n_pos: int, n_neg: int):
    """Sample n_pos points inside mask (label 1) and n_neg outside (label 0).
    Returns (points [P,2] xy, labels [P])."""
    ys_in, xs_in = np.where(mask)
    ys_out, xs_out = np.where(~mask)
    pts, lbls = [], []
    if len(xs_in) > 0:
        idx = np.random.choice(len(xs_in), size=min(n_pos, len(xs_in)), replace=False)
        for i in idx:
            pts.append([float(xs_in[i]), float(ys_in[i])])
            lbls.append(1)
    if len(xs_out) > 0 and n_neg > 0:
        idx = np.random.choice(len(xs_out), size=min(n_neg, len(xs_out)), replace=False)
        for i in idx:
            pts.append([float(xs_out[i]), float(ys_out[i])])
            lbls.append(0)
    if not pts:
        # fallback: one center point
        pts.append([mask.shape[1] / 2.0, mask.shape[0] / 2.0])
        lbls.append(1)
    return np.array(pts), np.array(lbls)


# ============================================================
# Auto-mask fusion
# ============================================================

def _combine_masks_by_fence_overlap(masks: list[dict], img: np.ndarray) -> np.ndarray:
    """Fuse SAM auto-mask output into a single fence mask.
    Heuristic: biggest mask as primary, union masks whose bbox overlaps primary ≥10%.
    """
    if not masks:
        return np.zeros(img.shape[:2], dtype=bool)
    masks_sorted = sorted(masks, key=lambda m: -m["area"])
    primary = masks_sorted[0]["segmentation"].astype(bool)
    best = primary.copy()
    for m in masks_sorted[1:10]:
        seg = m["segmentation"].astype(bool)
        if (seg & primary).sum() > 0.1 * seg.sum():
            best |= seg
    return best
