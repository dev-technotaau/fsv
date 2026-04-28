"""Combo 13: BEiT-V2-L + Mask2Former-lite decoder. FULL."""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_m2f_lite_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class BeitV2LMask2FormerAdapter(ModelAdapter):
    # BEiT-V2-L classification checkpoint from timm. Uses patch_size=16.
    MODEL_NAME = "beitv2_large_patch16_224.in1k_ft_in22k_in1k"

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        try:
            # BEiT doesn't always support features_only — use the `get_intermediate_layers` path
            import timm
            backbone = timm.create_model(self.MODEL_NAME, pretrained=True, num_classes=0)
            # Wrap to yield patch features in [B, C, H, W]
            patch_size = 16
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"timm BEiT-V2-L load failed: {e}")
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        # BEiT native resolution: 224; we'll resize to a multiple of 16
        input_size = (int(self.params.get("input_size", 512)) // patch_size) * patch_size
        n_patches = input_size // patch_size
        log(f"BEiT-V2-L, input_size={input_size}, patches={n_patches}")

        backbone_dim = 1024
        decoder = build_m2f_lite_decoder(
            in_dim=backbone_dim, num_queries=100, mask_feat=256, num_classes=1,
        ).to(device)

        def extract(bb, imgs):
            # BEiT forward with intermediate tokens
            tokens = bb.forward_features(imgs)
            # tokens shape: [B, N+1, C] with CLS; drop CLS
            if tokens.dim() == 3:
                if tokens.shape[1] == n_patches * n_patches + 1:
                    tokens = tokens[:, 1:, :]
                elif tokens.shape[1] != n_patches * n_patches:
                    # BEiT may have relative pos — handle by truncation/pad
                    expected = n_patches * n_patches
                    tokens = tokens[:, :expected, :] if tokens.shape[1] > expected \
                        else torch.nn.functional.pad(tokens, (0, 0, 0, expected - tokens.shape[1]))
                B, N, C = tokens.shape
                return tokens.transpose(1, 2).reshape(B, C, n_patches, n_patches)
            return tokens  # already a feature map

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=2, seed=self.data_cfg.seed,
        )
        lr = float(self.params.get("lr", 1e-4))
        optim = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

        loss_weights = {"focal": 1.0, "dice": 1.0, "lovasz": 0.5}

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
