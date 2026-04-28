"""Combo 06: InternImage-L (DCNv3) + UPerNet. FULL with graceful fallback.

InternImage's DCNv3 CUDA extension is hard to build on Windows. If the
`intern_image` package isn't importable, we fall back to a similar-capacity
backbone (ConvNeXt-L) to keep the GA running. The fallback is logged so
you know you're comparing apples to oranges.
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_upernet_head, dump_predictions,
    file_logger, pick_device, train_proxy,
)
from ._timm_backbone import load_timm_features


class InternImageLUPerNetAdapter(ModelAdapter):
    FALLBACK_TIMM = "convnext_large.fb_in22k_ft_in1k"

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        backbone, channels, extract, source = self._try_load_internimage(log)
        if backbone is None:
            log(f"InternImage unavailable, falling back to {self.FALLBACK_TIMM}")
            try:
                backbone, channels, extract = load_timm_features(self.FALLBACK_TIMM, pretrained=True)
                source = f"{self.FALLBACK_TIMM} (InternImage fallback)"
            except Exception as e:
                return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                     error=f"Both InternImage and ConvNeXt-L fallback failed: {e}")

        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        log(f"backbone: {source}, channels={channels}")

        input_size = int(self.params.get("input_size", 512))

        decoder = build_upernet_head(
            in_channels=channels,
            hidden_channels=256, num_classes=1,
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
            "ohem_ratio": float(self.params.get("ohem_ratio", 0.7)),
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
                 "combo_key": self.combo_key, "backbone": source}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt,
                             extra={"backbone_source": source})

    def _try_load_internimage(self, log):
        """Try to instantiate InternImage-L. Returns (model, channels, extract_fn, source) or (None,*,*,None)."""
        try:
            # Preferred: `intern_image` package from OpenGVLab repo
            from intern_image.models.intern_image import InternImage  # type: ignore[import]
            model = InternImage(
                core_op="DCNv3", channels=160, depths=[5, 5, 22, 5],
                groups=[10, 20, 40, 80], mlp_ratio=4.0, drop_path_rate=0.1,
                norm_layer="LN", layer_scale=1.0, offset_scale=2.0,
                post_norm=True, with_cp=False, out_indices=(0, 1, 2, 3),
            )
            channels = [160, 320, 640, 1280]
            def extract(bb, imgs):
                return bb(imgs)
            return model, channels, extract, "InternImage-L (DCNv3)"
        except ImportError as e:
            log(f"intern_image import failed: {e}")
            return None, None, None, None
        except Exception as e:
            log(f"InternImage instantiation failed: {e}")
            return None, None, None, None
