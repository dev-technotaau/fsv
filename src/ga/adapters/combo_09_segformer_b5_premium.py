"""Combo 09: SegFormer-B5 PREMIUM. SELF-CONTAINED (no subprocess).

Loads SegFormer-B5 via HuggingFace transformers, trains the full model (not just
decoder) with:
  - CE + Focal + Dice + Edge-aware losses
  - Optional OHEM (hard pixel mining)
  - EMA
  - Label smoothing
  - Warmup + cosine LR
  - Gradient accumulation

Saves both:
  - HF save_pretrained/  (loadable by any SegFormer consumer, incl. combo_14 ensemble)
  - decoder.pt           (small state-dict snapshot for fast reload)
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, dump_predictions, file_logger, pick_device,
    EMAHelper, combined_loss,
)


class SegFormerB5PremiumAdapter(ModelAdapter):
    MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

    def run(self) -> AdapterResult:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"transformers missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Load pretrained; replace head for 2 classes
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                self.MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
                id2label={0: "background", 1: "fence"},
                label2id={"background": 0, "fence": 1},
            ).to(device)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"SegFormer-B5 load failed: {e}")
        log(f"SegFormer-B5 loaded from {self.MODEL_NAME}")

        input_size = int(self.params.get("input_size", 640))
        # SegFormer output is 4× downsampled; keep input multiple of 4
        input_size = (input_size // 4) * 4

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=int(self.params.get("batch_size", 2)),
            seed=self.data_cfg.seed,
        )

        lr = float(self.params.get("lr", 6e-5))
        weight_decay = float(self.params.get("weight_decay", 0.02))
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_epochs = int(self.params.get("warmup_epochs", 5))
        accumulation_steps = int(self.params.get("accumulation_steps", 4))
        ohem_ratio = float(self.params.get("ohem_ratio", 0.7))
        edge_weight = float(self.params.get("edge_weight", 2.0))
        label_smoothing = float(self.params.get("label_smoothing", 0.1))
        ema_decay = float(self.params.get("ema_decay", 0.9999))

        proxy_epochs = self.fitness_cfg.proxy_epochs
        total_epochs = max(1, proxy_epochs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-7,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        ema = EMAHelper(model, decay=ema_decay)
        base_lr = lr
        early_kill_at = int(proxy_epochs * self.fitness_cfg.early_kill_at_fraction)

        best_iou = 0.0
        for ep in range(proxy_epochs):
            # warmup
            if ep < warmup_epochs:
                for pg in optim.param_groups:
                    pg["lr"] = base_lr * (ep + 1) / warmup_epochs
            elif ep == warmup_epochs:
                for pg in optim.param_groups:
                    pg["lr"] = base_lr

            model.train()
            ep_loss, n = 0.0, 0
            optim.zero_grad(set_to_none=True)
            for step, (imgs, masks) in enumerate(train_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    out = model(pixel_values=imgs)
                    logits = out.logits   # [B, 2, H/4, W/4]
                    logits = F.interpolate(logits, size=input_size,
                                           mode="bilinear", align_corners=False)
                    # Primary CE with label smoothing
                    ce = F.cross_entropy(logits, masks, label_smoothing=label_smoothing)
                    # Composite loss: CE + Focal + Dice + Edge + OHEM
                    # Convert 2-class logits to binary for our helpers
                    bin_logits = logits[:, 1:2, :, :] - logits[:, 0:1, :, :]
                    loss_comp = combined_loss(bin_logits, masks, {
                        "focal": 0.3,
                        "dice":  0.4,
                        "boundary": edge_weight,
                        "ohem": 0.3,
                        "ohem_ratio": ohem_ratio,
                    })
                    loss = ce + loss_comp
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                    optim.zero_grad(set_to_none=True)
                    ema.update(model)
                ep_loss += loss.item() * accumulation_steps
                n += 1
            if ep >= warmup_epochs:
                sched.step()
            log(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f} lr={optim.param_groups[0]['lr']:.2e}")

            # Validate with EMA weights
            if ep + 1 == early_kill_at or ep == proxy_epochs - 1 or ep % 3 == 0:
                ema.swap_in(model)
                val_iou = _eval_iou(model, val_loader, device, input_size)
                ema.swap_out(model)
                log(f"ep {ep} val_iou={val_iou:.4f}")
                if ep + 1 == early_kill_at and val_iou < self.fitness_cfg.early_kill_iou:
                    return AdapterResult(metrics={"iou": val_iou}, status="killed_early")
                best_iou = max(best_iou, val_iou)

        # Final eval with EMA permanently swapped in
        ema.swap_in(model)

        # Dump preds + GT
        from PIL import Image
        preds_dir = self.work_dir / "preds"; gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)
        model.eval()
        with torch.no_grad():
            for bi, (imgs, masks) in enumerate(val_loader):
                imgs = imgs.to(device)
                out = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=input_size,
                                       mode="bilinear", align_corners=False)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                masks_np = masks.cpu().numpy()
                threshold = float(self.params.get("threshold", 0.5))
                morph_k = int(self.params.get("morph_kernel", 0))
                for j in range(probs.shape[0]):
                    binary = (probs[j] > threshold).astype("uint8")
                    if morph_k > 0:
                        from ._common import _morph_close
                        binary = _morph_close(binary, morph_k)
                    Image.fromarray((binary * 255).astype("uint8")).save(
                        preds_dir / f"val_{bi:04d}_{j}.png")
                    Image.fromarray((masks_np[j].astype("uint8") * 255)).save(
                        gt_dir / f"val_{bi:04d}_{j}.png")

        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"final metrics: {metrics}")

        # Save in two formats for downstream adapters (esp. ensemble/cascade)
        hf_dir = self.work_dir / "hf_save"
        hf_dir.mkdir(exist_ok=True)
        try:
            model.save_pretrained(str(hf_dir))
        except Exception as e:
            log(f"save_pretrained failed: {e}")
        ckpt = self.work_dir / "best_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "params": self.params, "combo_key": self.combo_key,
            "hf_base": self.MODEL_NAME,
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt,
                             extra={"hf_save_pretrained_dir": str(hf_dir)})


def _eval_iou(model, loader, device, input_size) -> float:
    import torch
    import torch.nn.functional as F
    import numpy as np
    model.eval()
    ious = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(pixel_values=imgs)
            logits = F.interpolate(out.logits, size=input_size,
                                   mode="bilinear", align_corners=False)
            pred = torch.softmax(logits, dim=1)[:, 1] > 0.5
            inter = (pred & masks.bool()).sum(dim=(1, 2))
            union = (pred | masks.bool()).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0
