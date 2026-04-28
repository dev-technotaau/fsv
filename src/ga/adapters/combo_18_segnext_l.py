"""Combo 18: SegNeXt-L (MSCAN backbone). FULL with fallback.

SegNeXt's official implementation lives in mmsegmentation; `timm` also exposes
MSCAN-style backbones as `mscan_*`. We use whichever is available. Fallback:
ConvNeXt-L encoder with a UPerNet head (loses MSCAN architecture diversity
but lets the GA keep running).
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_upernet_head, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class SegNeXtLAdapter(ModelAdapter):
    CANDIDATE_MODELS = [
        "mscan_large",                              # timm — may or may not exist
        "convnext_large.fb_in22k_ft_in1k",          # fallback
    ]

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Try candidates in order
        backbone = None
        for name in self.CANDIDATE_MODELS:
            try:
                bb, channels, extract = load_timm_features(name, pretrained=True)
                backbone, src = bb, name
                break
            except Exception as e:
                log(f"{name} unavailable: {e}")
        if backbone is None:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error="Neither MSCAN nor ConvNeXt-L available in timm.")
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        log(f"SegNeXt adapter using backbone: {src}, channels={channels}")

        input_size = (int(self.params.get("input_size", 512)) // 32) * 32

        # HamNet-like head approximation via UPerNet with tunable ham_channels
        ham_ch = int(self.params.get("ham_channels", 256))
        decoder = build_upernet_head(
            in_channels=channels,
            hidden_channels=ham_ch,
            num_classes=1,
        ).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=2, seed=self.data_cfg.seed,
        )
        lr = float(self.params.get("lr", 1e-4))
        optim = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

        loss_weights = {"ohem": 1.0, "dice": 1.0, "boundary": 0.5}

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
                 "combo_key": self.combo_key, "backbone": src}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt,
                             extra={"backbone_source": src})
