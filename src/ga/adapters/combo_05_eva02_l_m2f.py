"""Combo 05: EVA-02-L (MIM pretrained) + Mask2Former-lite decoder. FULL."""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_m2f_lite_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class Eva02LMask2FormerAdapter(ModelAdapter):
    MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"

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
                                 error=f"Failed to load {self.MODEL_NAME} via timm: {e}. "
                                       f"Install timm>=0.9.12.")
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        log(f"EVA-02-L loaded; feature channels={channels}")

        # Use deepest feature for M2F-lite
        in_dim = channels[-1]
        input_size = (int(self.params.get("input_size", 448)) // 14) * 14  # EVA-02 uses 14px patches @ 448

        decoder = build_m2f_lite_decoder(
            in_dim=in_dim, num_queries=100, mask_feat=256, num_classes=1,
        ).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=1, seed=self.data_cfg.seed,
        )
        lr = float(self.params.get("lr", 1e-4))
        optim = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

        loss_weights = {
            "focal": 1.0, "dice": 1.0,
            "boundary": float(self.params.get("boundary_weight", 1.5)),
        }

        def extract_last(bb, imgs):
            feats = extract(bb, imgs)
            return feats[-1]

        best_iou, killed = train_proxy(
            backbone, decoder, extract_last, train_loader, val_loader,
            device=device, proxy_epochs=self.fitness_cfg.proxy_epochs,
            optim=optim, loss_weights=loss_weights,
            input_size=input_size, fitness_cfg=self.fitness_cfg, logger_fn=log,
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
        _t.save({"decoder_state": decoder.state_dict(), "params": self.params,
                 "combo_key": self.combo_key, "backbone": self.MODEL_NAME}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)
