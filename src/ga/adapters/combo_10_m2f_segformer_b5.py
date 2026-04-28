"""Combo 10: Mask2Former decoder + SegFormer-B5 backbone. SELF-CONTAINED.

Uses HuggingFace SegFormer-B5 as a multi-scale feature extractor (forward through
the encoder, collect intermediate hidden states from each stage), feeds these
into our proper Mask2Former decoder from _common.

Per docs/issues_and_fixes/CRITICAL_ISSUES_FOUND.md, we always train from scratch
(no resume from the old corrupt checkpoint).
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, build_mask2former_decoder, dump_predictions,
    file_logger, pick_device, train_proxy,
)


class Mask2FormerSegFormerB5Adapter(ModelAdapter):
    BACKBONE_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        try:
            from transformers import SegformerModel
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"transformers missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Load only the SegFormer encoder (not the classification head)
        try:
            backbone = SegformerModel.from_pretrained(self.BACKBONE_NAME).to(device)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"SegformerModel load failed: {e}")

        # SegFormer-B5 hidden dims per stage: [64, 128, 320, 512]
        channels = list(backbone.config.hidden_sizes)
        log(f"SegFormer-B5 encoder loaded; hidden_sizes={channels}")

        # Backbone learning rate multiplier (low — pretrained weights)
        backbone_lr_mult = float(self.params.get("backbone_lr_mult", 0.01))
        # Unfreeze: full encoder is trainable but at low LR
        for p in backbone.parameters():
            p.requires_grad_(True)

        input_size = int(self.params.get("input_size", 512))
        input_size = (input_size // 32) * 32

        decoder = build_mask2former_decoder(
            in_channels=channels,
            num_queries=100,
            mask_feat=256,
            num_classes=1,
            num_layers=int(self.params.get("deep_sup_layers", 6)),
        ).to(device)

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=int(self.params.get("batch_size", 2)),
            seed=self.data_cfg.seed,
        )

        lr = float(self.params.get("lr", 5e-5))
        optim = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": lr * backbone_lr_mult, "weight_decay": 0.01},
            {"params": decoder.parameters(),  "lr": lr,                      "weight_decay": 0.01},
        ])

        # Extract multi-scale features from SegFormer encoder
        def extract(bb, imgs):
            out = bb(pixel_values=imgs, output_hidden_states=True)
            # out.hidden_states: tuple of 4 stage outputs
            if out.hidden_states is None:
                return [out.last_hidden_state]
            feats = list(out.hidden_states)
            # Some transformers versions return shape [B, N, C]; reshape
            reshaped = []
            for f in feats:
                if f.dim() == 3:
                    B, N, C = f.shape
                    hw = int(N ** 0.5)
                    if hw * hw == N:
                        f = f.transpose(1, 2).reshape(B, C, hw, hw)
                    else:
                        # fallback — skip non-square (shouldn't happen for SegFormer)
                        continue
                reshaped.append(f)
            return reshaped[:len(channels)]  # ensure matches len(in_channels)

        # Decoder outputs dict — extract fused mask as logits
        def decoder_call(dec, feats):
            return dec(feats)

        def logits_fn(out_dict):
            return out_dict["fused_mask"]

        # Deep supervision from aux outputs
        def deep_sup(feats, out_dict):
            aux = out_dict["aux_outputs"]
            return [a[1].mean(dim=1, keepdim=True) for a in aux[:-1]]

        # Class-weight handling: fence is rare → high weight
        class_weight_fence = float(self.params.get("class_weight_fence", 15.0))
        loss_weights = {
            "focal":        1.5 * class_weight_fence / 15.0,
            "dice":         2.5,
            "boundary":     2.0,
            "lovasz":       0.0,   # disabled; numerically unstable per CRITICAL_ISSUES_FOUND
            "_train_backbone": True,
        }

        best_iou, killed = train_proxy(
            backbone, decoder, extract, train_loader, val_loader,
            device=device, proxy_epochs=self.fitness_cfg.proxy_epochs,
            optim=optim, loss_weights=loss_weights,
            input_size=input_size, fitness_cfg=self.fitness_cfg, logger_fn=log,
            warmup_epochs=int(self.params.get("warmup_epochs", 5)),
            scheduler="cosine",
            accumulation_steps=int(self.params.get("accumulation_steps", 4)),
            ema_decay=float(self.params.get("ema_decay", 0.9999)),
            decoder_call=decoder_call,
            postprocess_logits_fn=logits_fn,
            deep_sup_logits_fn=deep_sup,
            deep_sup_weights=[0.4, 0.3, 0.2, 0.1],
        )
        if killed:
            return AdapterResult(metrics={"iou": best_iou}, status="killed_early")

        preds_dir = dump_predictions(
            backbone, decoder, extract, val_loader,
            device=device, input_size=input_size, out_dir=self.work_dir,
            threshold=float(self.params.get("threshold", 0.5)),
            morph_k=int(self.params.get("morph_kernel", 0)),
            decoder_call=decoder_call,
            logits_fn=logits_fn,
            tta_mode=str(self.params.get("tta_mode", "none")),
        )
        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=self.work_dir / "gt")
        log(f"final metrics: {metrics}")

        ckpt = self.work_dir / "best_model.pth"
        import torch as _t
        _t.save({
            "backbone_state": backbone.state_dict(),
            "decoder_state": decoder.state_dict(),
            "params": self.params, "combo_key": self.combo_key,
            "backbone_name": self.BACKBONE_NAME,
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)
