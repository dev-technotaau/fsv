"""Combo 12: MODNet-style matting (trimap-free). FULL with simplified architecture.

Real MODNet has 3 branches (semantic, detail, fusion). We implement a
simplified 2-branch version with a MobileNetV2-like encoder + matting decoder
trained with pseudo-trimap alpha targets derived from binary GT masks.

The key idea: dilate GT by `trimap_dilation` to get an "unknown" band around
edges; within that band we train alpha as a Gaussian-softened transition
instead of hard 0/1. This teaches the network to produce soft alpha on edges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import AdapterResult, ModelAdapter
from ._common import FenceDataset, file_logger, pick_device


class ModNetMattingAdapter(ModelAdapter):
    def run(self) -> AdapterResult:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, random_split, Dataset
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")

        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # timm MobileNetV2 encoder — small, fast, works with matting
        try:
            from ._timm_backbone import load_timm_features
            backbone, channels, extract = load_timm_features("mobilenetv2_100.ra_in1k", pretrained=True)
        except Exception as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"Failed to load MobileNetV2 backbone: {e}")
        backbone.to(device).train()

        input_size = int(self.params.get("input_size", 512))
        input_size = (input_size // 32) * 32

        refiner_depth = int(self.params.get("refiner_depth", 5))
        se_ratio      = float(self.params.get("se_ratio", 0.25))
        trimap_dil    = int(self.params.get("trimap_dilation", 10))

        decoder = _MattingDecoder(
            encoder_channels=channels,
            refiner_depth=refiner_depth,
            se_ratio=se_ratio,
        ).to(device)

        # Pseudo-alpha dataset
        full_ds = _PseudoAlphaDataset(
            self.data_cfg.images_dir, self.data_cfg.masks_dir,
            input_size=input_size, trimap_dilation=trimap_dil,
            augment=str(self.params.get("augmentation", "medium")),
        )
        val_n = max(2, int(len(full_ds) * self.data_cfg.val_split))
        gen = torch.Generator().manual_seed(self.data_cfg.seed)
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_n, val_n], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0)

        lr = float(self.params.get("lr", 1e-4))
        params = list(decoder.parameters()) + list(backbone.parameters())
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        best_iou = 0.0
        for ep in range(self.fitness_cfg.proxy_epochs):
            decoder.train(); backbone.train()
            ep_loss, n = 0.0, 0
            for imgs, alpha_gt in train_loader:
                imgs, alpha_gt = imgs.to(device), alpha_gt.to(device)
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    feats = extract(backbone, imgs)
                    alpha_pred = decoder(feats, imgs)
                    loss = _matting_loss(alpha_pred, alpha_gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optim); scaler.update()
                ep_loss += loss.item(); n += 1
            log(f"ep {ep} avg_loss={ep_loss / max(1, n):.4f}")

            # Early kill check
            early_kill_at = int(self.fitness_cfg.proxy_epochs * self.fitness_cfg.early_kill_at_fraction)
            if ep + 1 == early_kill_at:
                val_iou = _eval_matting_iou(backbone, decoder, extract, val_loader, device)
                log(f"ep {ep} val_iou={val_iou:.4f}")
                if val_iou < self.fitness_cfg.early_kill_iou:
                    return AdapterResult(metrics={"iou": val_iou}, status="killed_early")
                best_iou = max(best_iou, val_iou)

        # ---------- final eval + save soft PNGs ----------
        from PIL import Image
        preds_dir = self.work_dir / "preds"
        gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)
        decoder.eval(); backbone.eval()
        with torch.no_grad():
            for bi, (imgs, alpha_gt) in enumerate(val_loader):
                imgs = imgs.to(device)
                feats = extract(backbone, imgs)
                alpha_pred = decoder(feats, imgs).cpu().numpy()
                alpha_gt_np = alpha_gt.cpu().numpy()
                for j in range(alpha_pred.shape[0]):
                    a = alpha_pred[j, 0]
                    binary = (a > 0.5).astype("uint8") * 255
                    Image.fromarray(binary).save(preds_dir / f"val_{bi:04d}_{j}.png")
                    Image.fromarray((alpha_gt_np[j] > 0.5).astype("uint8") * 255
                                    ).save(gt_dir / f"val_{bi:04d}_{j}.png")
                    # Save soft alpha too for downstream alpha-blending
                    (preds_dir / f"val_{bi:04d}_{j}_alpha.png").write_bytes(
                        _pil_to_png_bytes((a * 255).astype("uint8")))

        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"final metrics: {metrics}")
        ckpt = self.work_dir / "matting.pt"
        import torch as _t
        _t.save({
            "backbone_state": backbone.state_dict(),
            "decoder_state": decoder.state_dict(),
            "params": self.params, "combo_key": self.combo_key,
        }, ckpt)
        return AdapterResult(metrics=metrics, status="ok", ckpt_path=ckpt,
                             extra={"note": "Soft alpha PNGs saved as *_alpha.png for alpha-blend rendering."})


def _matting_loss(alpha_pred, alpha_gt):
    """L1 + compositional + laplacian pyramid losses."""
    import torch.nn.functional as F
    l1 = F.l1_loss(alpha_pred, alpha_gt)
    # laplacian pyramid L1
    lap = 0.0
    p_prev, t_prev = alpha_pred, alpha_gt
    for _ in range(3):
        p_down = F.avg_pool2d(p_prev, 2)
        t_down = F.avg_pool2d(t_prev, 2)
        p_up = F.interpolate(p_down, size=p_prev.shape[-2:], mode="bilinear", align_corners=False)
        t_up = F.interpolate(t_down, size=t_prev.shape[-2:], mode="bilinear", align_corners=False)
        lap = lap + F.l1_loss(p_prev - p_up, t_prev - t_up)
        p_prev, t_prev = p_down, t_down
    return l1 + 0.5 * lap


def _eval_matting_iou(backbone, decoder, extract, loader, device):
    import torch
    import numpy as np
    backbone.eval(); decoder.eval()
    ious = []
    with torch.no_grad():
        for imgs, alpha_gt in loader:
            imgs, alpha_gt = imgs.to(device), alpha_gt.to(device)
            feats = extract(backbone, imgs)
            alpha_pred = decoder(feats, imgs)
            pred = (alpha_pred > 0.5).squeeze(1).bool()
            gt = (alpha_gt > 0.5).squeeze(1).bool() if alpha_gt.dim() == 4 else (alpha_gt > 0.5)
            inter = (pred & gt).sum(dim=(1, 2))
            union = (pred | gt).sum(dim=(1, 2))
            iou = (inter.float() / union.clamp(min=1).float()).mean().item()
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def _pil_to_png_bytes(arr):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class _MattingDecoder:
    """Wrapper module — decoder that fuses encoder features + image into alpha."""
    def __init__(self, encoder_channels, refiner_depth, se_ratio):
        import torch
        import torch.nn as nn

        class MD(nn.Module):
            def __init__(self):
                super().__init__()
                c = encoder_channels[-1]
                self.up = nn.ModuleList()
                prev = c
                for ch in reversed(encoder_channels[:-1]):
                    self.up.append(nn.Sequential(
                        nn.Conv2d(prev + ch, prev // 2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(prev // 2), nn.ReLU(inplace=True),
                    ))
                    prev = prev // 2
                self.final_up = nn.Sequential(
                    nn.Conv2d(prev, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                )
                # refiner — residual blocks that mix with input RGB for fine alpha
                self.refiner = nn.Sequential()
                in_ch = 64 + 3
                for i in range(refiner_depth):
                    self.refiner.add_module(f"r{i}", nn.Sequential(
                        nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
                        nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    ))
                    in_ch = 32
                self.out = nn.Conv2d(32, 1, 1)

            def forward(self, feats, img):
                import torch.nn.functional as F
                x = feats[-1]
                for i, layer in enumerate(self.up):
                    skip = feats[-(i + 2)]
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                    x = layer(torch.cat([x, skip], dim=1))
                x = F.interpolate(x, size=img.shape[-2:], mode="bilinear", align_corners=False)
                x = self.final_up(x)
                x = self.refiner(torch.cat([x, img], dim=1))
                return torch.sigmoid(self.out(x))

        import torch
        self._mod = MD()

    def __call__(self, *a, **k): return self._mod(*a, **k)
    def parameters(self, *a, **k): return self._mod.parameters(*a, **k)
    def state_dict(self): return self._mod.state_dict()
    def train(self, mode=True): self._mod.train(mode); return self
    def eval(self): self._mod.eval(); return self
    def to(self, *a, **k): self._mod.to(*a, **k); return self


class _PseudoAlphaDataset:
    """Dataset that generates soft-alpha targets from binary fence masks.

    Hard regions (clearly fence or clearly background) stay at 1.0 / 0.0.
    Unknown band (near GT edge) gets a Gaussian-softened transition
    so the matting network can learn sub-pixel edge behavior.
    """
    def __init__(self, images_dir, masks_dir, input_size, trimap_dilation, augment):
        self.ds = FenceDataset(images_dir, masks_dir, input_size, augment=augment)
        self.dilation = trimap_dilation

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        import torch
        img_t, mask_t = self.ds[idx]
        binary = mask_t.numpy().astype("uint8")
        try:
            from scipy.ndimage import binary_dilation, distance_transform_edt
            dil = binary_dilation(binary, iterations=self.dilation)
            ero = ~binary_dilation(~binary.astype(bool), iterations=self.dilation)
            unknown = dil & ~ero
            alpha = binary.astype("float32")
            if unknown.any():
                dt_fg = distance_transform_edt(~binary.astype(bool))
                alpha_soft = np.exp(-dt_fg / (self.dilation * 0.5))
                alpha[unknown] = alpha_soft[unknown]
        except ImportError:
            alpha = binary.astype("float32")
        return img_t, torch.from_numpy(alpha).unsqueeze(0).float()
