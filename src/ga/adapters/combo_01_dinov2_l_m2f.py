"""Combo 01: DINOv2-L (frozen) + Mask2Former-lite decoder. FULL IMPLEMENTATION."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_m2f_lite_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)


class DinoV2LMask2FormerAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)
        log(f"device={device}")

        # ---------- load DINOv2-L ----------
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"Failed torch.hub load dinov2_vitl14: {e}. "
                                       f"Check internet or pre-cache ~/.cache/torch/hub.")
        backbone.to(device).eval()

        unfreeze_last = int(self.params.get("unfreeze_last", 0))
        for p in backbone.parameters():
            p.requires_grad_(False)
        if unfreeze_last > 0:
            for blk in list(backbone.blocks)[-unfreeze_last:]:  # type: ignore[attr-defined]
                for p in blk.parameters():
                    p.requires_grad_(True)

        backbone_dim = 1024
        # DINOv2 uses patch_size=14; round input_size to multiple of 14
        input_size = (int(self.params.get("input_size", 518)) // 14) * 14
        n_patches = input_size // 14

        # ---------- decoder ----------
        decoder = build_m2f_lite_decoder(
            in_dim=backbone_dim,
            num_queries=int(self.params.get("num_queries", 100)),
            mask_feat=int(self.params.get("mask_feat_dim", 256)),
            num_classes=1,
        ).to(device)

        # ---------- data + optim ----------
        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=2, seed=self.data_cfg.seed,
        )
        dec_lr = float(self.params.get("decoder_lr", 1e-4))
        params = list(decoder.parameters())
        if unfreeze_last > 0:
            params += [p for p in backbone.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=dec_lr, weight_decay=1e-4)

        loss_weights = {
            "focal": 1.0,
            "dice":  1.0,
            "boundary": float(self.params.get("boundary_weight", 1.5)),
            "lovasz": 0.5,
            "_train_backbone": unfreeze_last > 0,
        }

        def extract(bb, imgs):
            out = bb.forward_features(imgs)
            t = out["x_norm_patchtokens"]
            B, N, C = t.shape
            return t.transpose(1, 2).reshape(B, C, n_patches, n_patches)

        # ---------- train ----------
        best_iou, killed = train_proxy(
            backbone, decoder, extract, train_loader, val_loader,
            device=device, proxy_epochs=self.fitness_cfg.proxy_epochs,
            optim=optim, loss_weights=loss_weights,
            input_size=input_size, fitness_cfg=self.fitness_cfg, logger_fn=log,
        )
        if killed:
            return AdapterResult(
                metrics={"iou": best_iou, "boundary_f1": 0.0, "tv_penalty": 0.0},
                status="killed_early",
            )

        # ---------- dump preds + full eval ----------
        preds_dir = dump_predictions(
            backbone, decoder, extract, val_loader,
            device=device, input_size=input_size, out_dir=self.work_dir,
            threshold=float(self.params.get("threshold", 0.5)),
            morph_k=int(self.params.get("morph_kernel", 0)),
        )
        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=self.work_dir / "gt")
        log(f"final metrics: {metrics}")

        ckpt = self.work_dir / "decoder.pt"
        torch.save({"decoder_state": decoder.state_dict(), "params": self.params,
                    "combo_key": self.combo_key}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)
