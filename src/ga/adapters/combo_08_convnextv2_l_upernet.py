"""Combo 08: ConvNeXt-V2-L (FCMAE pretrained) + UPerNet. FULL."""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_upernet_head, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class ConvNeXtV2LUPerNetAdapter(ModelAdapter):
    MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k"

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
        log(f"ConvNeXt-V2-L loaded, channels={channels}")

        input_size = (int(self.params.get("input_size", 512)) // 32) * 32

        decoder = build_upernet_head(
            in_channels=channels,
            hidden_channels=384, num_classes=1,
        ).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=2, seed=self.data_cfg.seed,
        )
        lr = float(self.params.get("lr", 1e-4))
        optim = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

        loss_weights = {
            "ohem": 1.0, "dice": 1.0, "boundary": 0.5,
            "tversky": 0.5,
            "tversky_alpha": float(self.params.get("tversky_alpha", 0.5)),
            "tversky_beta":  float(self.params.get("tversky_beta", 0.5)),
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
        import torch as _t
        _t.save({"decoder_state": decoder.state_dict(), "params": self.params,
                 "combo_key": self.combo_key, "backbone": self.MODEL_NAME}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)
