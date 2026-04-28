"""Combo 17: DINOv2-L (frozen) + UNet++ decoder. FULL.

Best encoder × best decoder ablation. Builds a custom multi-scale feature
pyramid from DINOv2 ViT blocks (taps at blocks [6, 12, 18, 23]), feeds into
segmentation_models_pytorch.UnetPlusPlus via a custom encoder wrapper.
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, ModelAdapter
from ._common import (
    build_dataloaders, dump_predictions, file_logger, pick_device, train_proxy,
)


class DinoV2LUNetPPAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # Build DINOv2-L as multi-scale feature source
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"DINOv2-L load failed: {e}")
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        input_size = (int(self.params.get("input_size", 518)) // 14) * 14
        n_patches = input_size // 14

        # ViT-L/14 has 24 blocks; tap at 6, 12, 18, 23 for multi-scale
        tap_layers = [6, 12, 18, 23]
        backbone_dim = 1024

        # Custom decoder — pure UNet++ nested skip style
        dec_channels = self.params.get("decoder_channels", (512, 256, 128, 64, 32))
        if isinstance(dec_channels, str):
            # YAML may load choice tuples as list[int] already
            dec_channels = tuple(dec_channels)

        attn = self.params.get("attention_type", "scse")
        decoder = _UNetPPLike(
            encoder_channels=[backbone_dim] * len(tap_layers),
            decoder_channels=tuple(dec_channels) if isinstance(dec_channels, (list, tuple)) else (512, 256, 128, 64, 32),
            attention=attn if attn in ("scse", "cbam") else None,
            num_classes=1,
        ).to(device)

        def extract(bb, imgs):
            """Tap intermediate blocks; reshape to feature pyramid with decreasing spatial res."""
            import torch
            B = imgs.shape[0]
            # Use get_intermediate_layers if available
            if hasattr(bb, "get_intermediate_layers"):
                layers = bb.get_intermediate_layers(imgs, n=tap_layers, reshape=False)
                # layers: tuple of [B, N+extra_tokens, C] or [B, N, C]
                feats = []
                for i, t in enumerate(layers):
                    # strip CLS/register tokens — DINOv2 uses 1 CLS + 4 register = 5 extra by default
                    expected = n_patches * n_patches
                    if t.shape[1] > expected:
                        t = t[:, -expected:, :]
                    t_map = t.transpose(1, 2).reshape(B, backbone_dim, n_patches, n_patches)
                    # Downsample deeper tap layers to simulate pyramid
                    stride = 2 ** i
                    if stride > 1:
                        t_map = torch.nn.functional.avg_pool2d(t_map, stride, stride)
                    feats.append(t_map)
                return feats
            # Fallback: same feature repeated at 4 scales
            out = bb.forward_features(imgs)
            t = out["x_norm_patchtokens"].transpose(1, 2).reshape(B, backbone_dim, n_patches, n_patches)
            return [
                t,
                torch.nn.functional.avg_pool2d(t, 2, 2),
                torch.nn.functional.avg_pool2d(t, 4, 4),
                torch.nn.functional.avg_pool2d(t, 8, 8),
            ]

        train_loader, val_loader = build_dataloaders(
            self.data_cfg, input_size,
            augment=str(self.params.get("augmentation", "medium")),
            batch_size=1, seed=self.data_cfg.seed,
        )
        dec_lr = float(self.params.get("decoder_lr", 1e-4))
        optim = torch.optim.AdamW(decoder.parameters(), lr=dec_lr, weight_decay=1e-4)
        loss_weights = {"focal": 1.0, "dice": 1.0, "boundary": 1.5, "lovasz": 0.5}

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
                 "combo_key": self.combo_key, "attention": attn}, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt)


class _UNetPPLike:
    """Lightweight UNet++ decoder with nested skip connections + optional SCSE/CBAM."""
    def __init__(self, encoder_channels, decoder_channels, attention, num_classes):
        import torch
        import torch.nn as nn
        class SCSE(nn.Module):
            def __init__(self, c, r=16):
                super().__init__()
                self.cse = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1),
                    nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1), nn.Sigmoid(),
                )
                self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
            def forward(self, x):
                return x * self.cse(x) + x * self.sse(x)

        def conv_block(in_c, out_c):
            blocks = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            ]
            if attention == "scse":
                blocks.append(SCSE(out_c))
            return nn.Sequential(*blocks)

        class UPP(nn.Module):
            def __init__(self):
                super().__init__()
                # Build nested skip grid [i,j] for i in encoder stages, j in nesting depth
                enc_n = len(encoder_channels)
                dec_n = len(decoder_channels)
                self.X = nn.ModuleDict()
                for i in range(enc_n):
                    in_ch = encoder_channels[i]
                    for j in range(dec_n - i):
                        if j == 0:
                            self.X[f"{i}_{j}"] = conv_block(in_ch, decoder_channels[-(j + 1)])
                        else:
                            # concat: X[i][0:j] + upsampled X[i+1][j-1]
                            concat_in = sum(decoder_channels[-(k + 1)] for k in range(j))
                            concat_in += decoder_channels[-(j + 1)] if j > 0 else 0  # prev nest
                            concat_in += decoder_channels[-j]  # upsampled from lower stage
                            self.X[f"{i}_{j}"] = conv_block(concat_in, decoder_channels[-(j + 1)])
                self.final = nn.Conv2d(decoder_channels[-1], num_classes, 1)

            def forward(self, feats):
                import torch
                import torch.nn.functional as F
                enc_n = len(feats)
                dec_n = len(decoder_channels)
                # Initialize X[i][0] from encoder features
                X = {}
                for i in range(enc_n):
                    X[(i, 0)] = self.X[f"{i}_0"](feats[i])
                # Fill nested grid
                for j in range(1, dec_n):
                    for i in range(enc_n - j):
                        below = F.interpolate(X[(i + 1, j - 1)], size=X[(i, 0)].shape[-2:],
                                              mode="bilinear", align_corners=False)
                        prior_nests = [X[(i, k)] for k in range(j)]
                        cat = torch.cat(prior_nests + [below], dim=1)
                        X[(i, j)] = self.X[f"{i}_{j}"](cat)
                return self.final(X[(0, dec_n - 1)])

        import torch
        self._mod = UPP()

    def __call__(self, feats): return self._mod(feats)
    def parameters(self, *a, **k): return self._mod.parameters(*a, **k)
    def state_dict(self): return self._mod.state_dict()
    def train(self, mode=True): self._mod.train(mode); return self
    def eval(self): self._mod.eval(); return self
    def to(self, *a, **k): self._mod.to(*a, **k); return self
