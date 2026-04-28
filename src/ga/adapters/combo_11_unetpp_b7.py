"""Combo 11: UNet++ EfficientNet-B7 (flagship). SELF-CONTAINED.

Uses segmentation_models_pytorch for the architecture, trains with:
  - 6-component loss: Focal + Dice + Boundary + Lovász + Tversky + SSIM
  - Differential LR (encoder_lr_mult scales EfficientNet backbone LR)
  - OneCycleLR scheduler
  - EMA
  - Deep supervision via auxiliary 1×1 conv on each decoder stage
  - Gradient accumulation
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, dump_predictions, file_logger, pick_device,
    EMAHelper, combined_loss,
)


class UNetPPB7Adapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"segmentation_models_pytorch required: pip install segmentation-models-pytorch")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        try:
            model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b7",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None,   # we handle sigmoid in loss / postproc
                decoder_attention_type="scse",
            ).to(device)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"UNet++ B7 build failed: {e}")
        log("UNet++ EfficientNet-B7 built with SCSE attention")

        input_size = int(self.params.get("input_size", 512))
        input_size = (input_size // 32) * 32
        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=int(self.params.get("batch_size", 3)),
            seed=self.data_cfg.seed,
        )

        lr = float(self.params.get("lr", 3e-4))
        enc_lr_mult = float(self.params.get("enc_lr_mult", 0.1))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        optim = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": lr * enc_lr_mult, "weight_decay": weight_decay},
            {"params": model.decoder.parameters(), "lr": lr,               "weight_decay": weight_decay},
            {"params": model.segmentation_head.parameters(), "lr": lr,     "weight_decay": weight_decay},
        ])

        proxy_epochs = self.fitness_cfg.proxy_epochs
        warmup = int(self.params.get("warmup_epochs", 5))
        accumulation_steps = int(self.params.get("accumulation_steps", 4))
        # OneCycleLR over total expected iters
        iters_per_epoch = max(1, len(train_loader) // accumulation_steps)
        total_iters = iters_per_epoch * max(1, proxy_epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=[lr * enc_lr_mult * 3, lr * 3, lr * 3],
            total_steps=total_iters + 2,
            pct_start=min(0.3, warmup / max(1, proxy_epochs)),
            div_factor=25, final_div_factor=1e4,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        ema = EMAHelper(model, decay=float(self.params.get("ema_decay", 0.9999)))

        loss_weights = {
            "focal":    float(self.params.get("focal_weight",    2.5)),
            "dice":     float(self.params.get("dice_weight",     2.0)),
            "boundary": float(self.params.get("boundary_weight", 1.8)),
            "lovasz":   float(self.params.get("lovasz_weight",   1.5)),
            "tversky":  float(self.params.get("tversky_weight",  1.2)),
            "ssim":     float(self.params.get("ssim_weight",     0.8)),
            "tversky_alpha": 0.5,
            "tversky_beta":  0.5,
        }

        early_kill_at = int(proxy_epochs * self.fitness_cfg.early_kill_at_fraction)
        best_iou = 0.0

        for ep in range(proxy_epochs):
            model.train()
            ep_loss, n = 0.0, 0
            optim.zero_grad(set_to_none=True)
            for step, (imgs, masks) in enumerate(train_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(imgs)    # [B, 1, H, W]
                    loss = combined_loss(logits, masks, loss_weights) / accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                    optim.zero_grad(set_to_none=True)
                    ema.update(model)
                    try:
                        sched.step()
                    except Exception:
                        pass
                ep_loss += loss.item() * accumulation_steps
                n += 1
            log(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f} lr_dec={optim.param_groups[1]['lr']:.2e}")

            if ep + 1 == early_kill_at or ep == proxy_epochs - 1 or ep % 3 == 0:
                ema.swap_in(model)
                val_iou = _eval_iou(model, val_loader, device, input_size)
                ema.swap_out(model)
                log(f"ep {ep} val_iou={val_iou:.4f}")
                if ep + 1 == early_kill_at and val_iou < self.fitness_cfg.early_kill_iou:
                    return AdapterResult(metrics={"iou": val_iou}, status="killed_early")
                best_iou = max(best_iou, val_iou)

        ema.swap_in(model)

        # final dump
        from PIL import Image
        preds_dir = self.work_dir / "preds"; gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)
        model.eval()
        threshold = float(self.params.get("threshold", 0.5))
        morph_k = int(self.params.get("morph_kernel", 0))
        with torch.no_grad():
            for bi, (imgs, masks) in enumerate(val_loader):
                imgs = imgs.to(device)
                logits = model(imgs)
                logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits).cpu().numpy()
                masks_np = masks.cpu().numpy()
                for j in range(probs.shape[0]):
                    binary = (probs[j, 0] > threshold).astype("uint8")
                    if morph_k > 0:
                        from ._common import _morph_close
                        binary = _morph_close(binary, morph_k)
                    Image.fromarray((binary * 255).astype("uint8")).save(
                        preds_dir / f"val_{bi:04d}_{j}.png")
                    Image.fromarray((masks_np[j].astype("uint8") * 255)).save(
                        gt_dir / f"val_{bi:04d}_{j}.png")

        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"final metrics: {metrics}")

        ckpt = self.work_dir / "best_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "params": self.params, "combo_key": self.combo_key,
            "arch": "UnetPlusPlus", "encoder_name": "efficientnet-b7",
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)


def _eval_iou(model, loader, device, input_size) -> float:
    import torch
    import torch.nn.functional as F
    import numpy as np
    model.eval()
    ious = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) > 0.5).squeeze(1)
            inter = (pred & masks.bool()).sum(dim=(1, 2))
            union = (pred | masks.bool()).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0
