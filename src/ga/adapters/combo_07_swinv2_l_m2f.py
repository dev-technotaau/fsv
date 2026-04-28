"""Combo 07: Swin-V2-L + Mask2Former-lite + deep supervision. FULL."""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_m2f_lite_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class SwinV2LMask2FormerAdapter(ModelAdapter):
    MODEL_NAME = "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        try:
            backbone, channels, extract = load_timm_features(self.MODEL_NAME, pretrained=True)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"timm load failed: {e}")
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        log(f"Swin-V2-L loaded, channels={channels}")

        input_size = int(self.params.get("input_size", 512))
        # Round to multiple of 32 (Swin stride)
        input_size = (input_size // 32) * 32

        in_dim = channels[-1]
        decoder = build_m2f_lite_decoder(
            in_dim=in_dim, num_queries=100, mask_feat=256, num_classes=1,
        ).to(device)

        # Deep-supervision auxiliary heads on shallower features
        import torch.nn as nn
        deep_sup_layers = int(self.params.get("deep_sup_layers", 3))
        aux_heads = nn.ModuleList([
            nn.Conv2d(c, 1, 1) for c in channels[:deep_sup_layers]
        ]).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=2, seed=self.data_cfg.seed,
        )
        lr = float(self.params.get("lr", 1e-4))
        optim = torch.optim.AdamW(
            list(decoder.parameters()) + list(aux_heads.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        loss_weights = {
            "focal": 1.0, "dice": 1.0,
            "boundary": float(self.params.get("boundary_weight", 1.5)),
            "lovasz": 0.5,
        }

        def extract_last(bb, imgs):
            return extract(bb, imgs)[-1]

        def aux_logits(feats):
            """Call during training only (feats is the last feature map).
            For deep sup we need earlier features — re-extract.
            """
            return []   # disable aux here; see note below

        # Deep supervision properly requires access to all features.
        # Our train_proxy contract only passes the final feature to decoder.
        # Bypass: write a custom training loop inline.
        best_iou, killed = _train_with_deepsup(
            backbone, decoder, aux_heads, extract, train_loader, val_loader,
            device=device, proxy_epochs=self.fitness_cfg.proxy_epochs,
            optim=optim, loss_weights=loss_weights, input_size=input_size,
            fitness_cfg=self.fitness_cfg, logger_fn=log,
            deep_sup_weights=[0.6, 0.4, 0.2][:deep_sup_layers],
        )
        if killed:
            return AdapterResult(metrics={"iou": best_iou}, status="killed_early")

        preds_dir = dump_predictions(
            backbone, decoder, extract_last, val_loader,
            device=device, input_size=input_size, out_dir=self.work_dir,
            threshold=float(self.params.get("threshold", 0.5)),
            morph_k=int(self.params.get("morph_kernel", 0)),
        )
        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=self.work_dir / "gt")
        ckpt = self.work_dir / "decoder.pt"
        import torch as _t
        _t.save({
            "decoder_state": decoder.state_dict(),
            "aux_state": aux_heads.state_dict(),
            "params": self.params, "combo_key": self.combo_key,
            "backbone": self.MODEL_NAME,
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)


def _train_with_deepsup(
    backbone, decoder, aux_heads, extract, train_loader, val_loader,
    *, device, proxy_epochs, optim, loss_weights, input_size,
    fitness_cfg, logger_fn, deep_sup_weights,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from ._common import combined_loss, _eval_iou_generic

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    early_kill_at = int(proxy_epochs * fitness_cfg.early_kill_at_fraction)
    best_iou = 0.0

    def extract_last(bb, imgs):
        return extract(bb, imgs)[-1]

    for ep in range(proxy_epochs):
        decoder.train(); aux_heads.train()
        backbone.eval()
        ep_loss, n = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feats_list = extract(backbone, imgs)
                logits = decoder(feats_list[-1])
                logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
                loss = combined_loss(logits, masks, loss_weights)
                # aux
                for w, aux, feat in zip(deep_sup_weights, aux_heads, feats_list[:len(deep_sup_weights)]):
                    aux_l = aux(feat)
                    aux_l = F.interpolate(aux_l, size=input_size, mode="bilinear", align_corners=False)
                    loss = loss + w * combined_loss(aux_l, masks, loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(aux_heads.parameters()), 1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n += 1
        logger_fn(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f}")

        if ep + 1 == early_kill_at or ep == proxy_epochs - 1:
            val_iou = _eval_iou_generic(backbone, decoder, extract_last, val_loader, device, input_size)
            logger_fn(f"ep {ep} val_iou={val_iou:.4f}")
            if ep + 1 == early_kill_at and val_iou < fitness_cfg.early_kill_iou:
                return val_iou, True
            best_iou = max(best_iou, val_iou)
    return best_iou, False
