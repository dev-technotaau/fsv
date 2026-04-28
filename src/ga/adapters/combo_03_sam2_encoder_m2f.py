"""Combo 03: SAM 2 Hiera-L image encoder (frozen) + Mask2Former decoder.

Requires `sam2` package (pip install from facebookresearch/sam2 repo).
Falls back to HuggingFace `transformers` SAM (v1) if sam2 unavailable.
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_m2f_lite_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)


class Sam2EncoderMask2FormerAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Try SAM 2 first, then fall back to SAM v1 via transformers
        backbone, backbone_dim, extract, info = _load_sam_encoder(log, device, self.work_dir)
        if backbone is None:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"No SAM encoder available: {info}")
        log(f"SAM encoder loaded: {info}")

        # Freeze encoder; optionally unlock last K blocks
        unfreeze_last = int(self.params.get("unfreeze_last", 0))
        for p in backbone.parameters():
            p.requires_grad_(False)
        if unfreeze_last > 0 and hasattr(backbone, "blocks"):
            for blk in list(backbone.blocks)[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

        input_size = int(self.params.get("input_size", 1024))
        # SAM encoders expect 1024 — enforce
        input_size = min(max(input_size, 512), 1024)

        decoder = build_m2f_lite_decoder(
            in_dim=backbone_dim,
            num_queries=int(self.params.get("num_queries", 100)),
            mask_feat=256,
            num_classes=1,
        ).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=1, seed=self.data_cfg.seed,
        )
        dec_lr = float(self.params.get("decoder_lr", 1e-4))
        params = list(decoder.parameters())
        if unfreeze_last > 0:
            params += [p for p in backbone.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=dec_lr, weight_decay=1e-4)

        loss_weights = {"focal": 1.0, "dice": 1.0, "lovasz": 0.5, "_train_backbone": unfreeze_last > 0}

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
                 "combo_key": self.combo_key, "encoder_info": info}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)


def _load_sam_encoder(log, device, work_dir: Path):
    """Returns (backbone_module, dim, extract_fn, info_str) or (None, 0, None, err)."""
    import torch
    # Try SAM 2
    try:
        from sam2.build_sam import build_sam2  # type: ignore[import]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        # Best-effort auto-download of a small checkpoint
        ckpt_dir = Path.home() / ".cache" / "sam2"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # User must pre-download config + weights from the sam2 repo. We try a
        # standard name; raise clean error if missing.
        cfg_name = "sam2_hiera_l.yaml"
        ckpt_path = ckpt_dir / "sam2_hiera_large.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {ckpt_path}. Download from "
                f"https://github.com/facebookresearch/sam2 and place there."
            )
        model = build_sam2(cfg_name, str(ckpt_path), device=str(device))
        encoder = model.image_encoder
        dim = 256  # Hiera-L final feature channels after neck
        def extract(bb, imgs):
            # Hiera outputs multi-scale; we take the final neck feature
            out = bb(imgs)
            if isinstance(out, dict) and "vision_features" in out:
                return out["vision_features"]
            if isinstance(out, (list, tuple)):
                return out[-1]
            return out
        return encoder, dim, extract, "SAM 2 Hiera-L"
    except Exception as e:
        log(f"SAM 2 unavailable ({e}); trying HF SAM v1 fallback")
    # Fallback: HuggingFace SAM v1 ViT-B
    try:
        from transformers import SamModel  # type: ignore[import]
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        encoder = model.vision_encoder
        dim = 256  # SAM vision encoder out_channels
        def extract(bb, imgs):
            return bb(imgs).last_hidden_state
        return encoder, dim, extract, "HF SAM v1 ViT-B (fallback)"
    except Exception as e:
        return None, 0, None, f"HF SAM v1 also failed: {e}"
