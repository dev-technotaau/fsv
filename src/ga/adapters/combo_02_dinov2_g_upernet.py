"""Combo 02: DINOv2-G (frozen) + UPerNet. FULL IMPLEMENTATION.

DINOv2-G is ~1.1B params — needs ≥24GB VRAM. Fallback: auto-downgrade to DINOv2-L
if CUDA OOM or no GPU.
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_upernet_head, dump_predictions,
    file_logger, pick_device, train_proxy,
)


class DinoV2GUPerNetAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Try DINOv2-G; fall back to DINOv2-L on OOM / hub failure
        hub_model = "dinov2_vitg14"
        backbone_dim = 1536  # ViT-G/14 token dim
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", hub_model, trust_repo=True)
            backbone.to(device).eval()
        except Exception as e:
            log(f"DINOv2-G load failed ({e}); falling back to DINOv2-L")
            try:
                backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
                backbone.to(device).eval()
                backbone_dim = 1024
                hub_model = "dinov2_vitl14 (fallback from G)"
            except Exception as e2:
                return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                     error=f"Both DINOv2-G and DINOv2-L failed: {e2}")

        log(f"backbone={hub_model} dim={backbone_dim}")
        for p in backbone.parameters():
            p.requires_grad_(False)

        input_size = (int(self.params.get("input_size", 518)) // 14) * 14
        n_patches = input_size // 14

        # We'll synthesize a 4-scale pyramid from DINOv2's single feature map
        # via average pooling for UPerNet. Cheap but effective.
        in_channels = [backbone_dim, backbone_dim, backbone_dim, backbone_dim]
        decoder = build_upernet_head(
            in_channels=in_channels,
            hidden_channels=int(self.params.get("decoder_channels", 384)),
            num_classes=1,
        ).to(device)

        def extract(bb, imgs):
            out = bb.forward_features(imgs)
            t = out["x_norm_patchtokens"]
            B, N, C = t.shape
            x = t.transpose(1, 2).reshape(B, C, n_patches, n_patches)
            # fake pyramid: 1x, 1/2, 1/4, 1/8
            f1 = torch.nn.functional.avg_pool2d(x, 1, 1)
            f2 = torch.nn.functional.avg_pool2d(x, 2, 2)
            f3 = torch.nn.functional.avg_pool2d(x, 4, 4)
            f4 = torch.nn.functional.avg_pool2d(x, 8, 8)
            return [f1, f2, f3, f4]

        # loaders
        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=1, seed=self.data_cfg.seed,     # batch 1 for G
        )

        dec_lr = float(self.params.get("decoder_lr", 5e-5))
        optim = torch.optim.AdamW(decoder.parameters(), lr=dec_lr, weight_decay=1e-4)

        loss_weights = {
            "ohem":  1.0,
            "dice":  1.0,
            "boundary": 0.5,
            "ohem_ratio": 0.7,
        }

        best_iou, killed = train_proxy(
            backbone, decoder, extract, train_loader, val_loader,
            device=device, proxy_epochs=self.fitness_cfg.proxy_epochs,
            optim=optim, loss_weights=loss_weights,
            input_size=input_size, fitness_cfg=self.fitness_cfg, logger_fn=log,
        )
        if killed:
            return AdapterResult(metrics={"iou": best_iou}, status="killed_early")

        preds_dir = dump_predictions(
            backbone, decoder, extract, val_loader,
            device=device, input_size=input_size, out_dir=self.work_dir,
            threshold=float(self.params.get("threshold", 0.5)),
            morph_k=int(self.params.get("morph_kernel", 0)),
        )
        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=self.work_dir / "gt")
        ckpt = self.work_dir / "decoder.pt"
        torch.save({"decoder_state": decoder.state_dict(), "params": self.params,
                    "combo_key": self.combo_key, "backbone": hub_model}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)
