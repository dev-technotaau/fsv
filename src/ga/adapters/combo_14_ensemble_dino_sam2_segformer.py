"""Combo 14: Ensemble (DINOv2 + SAM 2 + SegFormer-B5) — weighted avg + CRF.

REQUIRES pre-trained checkpoints for combos 01, 03, 09 on disk. Discovers them
under the GA output tree. Runs inference with each, averages mask probabilities
with genome-controlled alphas, then applies DenseCRF (if pydensecrf available).

This adapter does NOT train gradients. `proxy_epochs` is ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import AdapterResult, ModelAdapter
from ._common import FenceDataset, file_logger, pick_device


class EnsembleAdapter(ModelAdapter):
    REQUIRED_COMBOS = ["01_dinov2_l_m2f", "03_sam2_encoder_m2f", "09_segformer_b5_premium"]

    def run(self) -> AdapterResult:
        try:
            import torch
        except ImportError as e:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"torch missing: {e}")
        log = file_logger(self.work_dir / "adapter.log")
        device = pick_device(self.gpu_id)

        # ---------- discover checkpoints ----------
        ga_root = _find_ga_output_root(self.work_dir)
        if ga_root is None:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error="Could not locate GA output root above work_dir. "
                                       "Ensemble needs pre-trained checkpoints from combos 01/03/09.")
        ckpts: dict[str, Path] = {}
        for key in self.REQUIRED_COMBOS:
            ckpt = _find_latest_ckpt(ga_root, key)
            if ckpt is None:
                return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                     error=f"No checkpoint found for required combo {key} under {ga_root}. "
                                           f"Run stage1 first; ensemble needs it.")
            ckpts[key] = ckpt
            log(f"found checkpoint: {key} → {ckpt}")

        # ---------- load predictors ----------
        predictors: list[tuple[str, callable, int]] = []
        try:
            predictors.append(("dino", _make_dino_predictor(ckpts["01_dinov2_l_m2f"], device, log), 518))
        except Exception as e:
            log(f"failed to load DINO predictor: {e}")
        try:
            predictors.append(("sam2", _make_sam2_encoder_predictor(ckpts["03_sam2_encoder_m2f"], device, log), 512))
        except Exception as e:
            log(f"failed to load SAM2-encoder predictor: {e}")
        try:
            predictors.append(("segformer", _make_segformer_predictor(ckpts["09_segformer_b5_premium"], device, log), 640))
        except Exception as e:
            log(f"failed to load SegFormer predictor: {e}")
        if len(predictors) < 2:
            return AdapterResult(metrics={"iou": 0.0}, status="crashed",
                                 error=f"Ensemble needs ≥2 loadable predictors; got {len(predictors)}.")

        # ---------- inference over val set ----------
        input_size = max(p[2] for p in predictors)
        ds = FenceDataset(self.data_cfg.images_dir, self.data_cfg.masks_dir,
                          input_size, augment="none")
        val_n = max(2, int(len(ds) * self.data_cfg.val_split))
        val_pairs = ds.pairs[-val_n:]

        alpha_d = float(self.params.get("alpha_dino", 0.4))
        alpha_s = float(self.params.get("alpha_sam2", 0.3))
        alpha_b = float(self.params.get("alpha_b5", 0.3))
        crf_iter = int(self.params.get("crf_iter", 5))
        crf_pos_sxy = float(self.params.get("crf_pos_sxy", 3.0))
        crf_bilat  = float(self.params.get("crf_bilateral", 10.0))

        # Normalize alphas over available predictors only
        w = {"dino": alpha_d, "sam2": alpha_s, "segformer": alpha_b}
        names = [p[0] for p in predictors]
        s = sum(w[n] for n in names)
        w = {n: w[n] / s for n in names}
        log(f"normalized weights: {w}")

        preds_dir = self.work_dir / "preds"
        gt_dir = self.work_dir / "gt"
        preds_dir.mkdir(exist_ok=True); gt_dir.mkdir(exist_ok=True)

        for i, (img_p, mask_p) in enumerate(val_pairs):
            img = np.array(Image.open(img_p).convert("RGB").resize(
                (input_size, input_size), Image.BILINEAR))
            gt = np.array(Image.open(mask_p).convert("L").resize(
                (input_size, input_size), Image.NEAREST)) > 127

            agg = np.zeros((input_size, input_size), dtype=np.float32)
            for name, predict_fn, size in predictors:
                prob = predict_fn(img_p)   # returns HxW float in [0,1]
                prob = _resize_prob(prob, input_size)
                agg += w[name] * prob

            if crf_iter > 0:
                agg_after = _apply_crf(img, agg, crf_iter, crf_pos_sxy, crf_bilat)
                binary = agg_after > 0.5
            else:
                binary = agg > 0.5

            Image.fromarray((binary.astype("uint8") * 255)).save(preds_dir / f"val_{i:04d}.png")
            Image.fromarray((gt.astype("uint8") * 255)).save(gt_dir / f"val_{i:04d}.png")

        metrics = self.compute_iou_and_bf1(preds_dir, masks_dir=gt_dir)
        log(f"ensemble metrics: {metrics}")
        return AdapterResult(metrics=metrics, status="ok",
                             ckpt_path=self.work_dir / "config.txt",
                             extra={"weights": w, "used_predictors": names})


def _find_ga_output_root(work_dir: Path) -> Optional[Path]:
    """Walk up until we find a dir containing 'generations' or 'fitness_cache.jsonl'."""
    cur = work_dir.resolve()
    for _ in range(10):
        if (cur / "fitness_cache.jsonl").exists() or (cur / "generations").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _find_latest_ckpt(ga_root: Path, combo_key: str) -> Optional[Path]:
    """Find the most recent best checkpoint for a given combo under ga_root/generations.

    Returns either a file path (.pth/.pt) OR a directory path (HF save_pretrained dir).
    Preference: HF dir > .pth > .pt. If multiple matches, newest by mtime.
    """
    gens_dir = ga_root / "generations"
    if not gens_dir.exists():
        return None
    candidates: list[Path] = []
    for ind in gens_dir.rglob(f"ind_*_{combo_key}_*"):
        # HF save_pretrained directory
        for d in ind.glob("hf_save"):
            if d.is_dir() and (d / "config.json").exists():
                candidates.append(d)
        # .pth / .pt files
        for pat in ("best_model*.pth", "decoder.pt", "mask_decoder.pt",
                    "refiner.pt", "matting.pt", "checkpoints/best_model*.pth"):
            for f in ind.glob(pat):
                candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resize_prob(prob: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((prob * 255).astype("uint8"))
    return np.array(img.resize((size, size), Image.BILINEAR)).astype("float32") / 255.0


def _apply_crf(img, prob, n_iter, pos_sxy, bilat_sxy):
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return prob   # graceful no-op
    H, W = prob.shape
    probs_2cls = np.stack([1 - prob, prob], axis=0)
    U = unary_from_softmax(probs_2cls)
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=pos_sxy, compat=3)
    d.addPairwiseBilateral(sxy=bilat_sxy, srgb=13, rgbim=img.astype("uint8"), compat=10)
    Q = d.inference(n_iter)
    return np.array(Q).reshape((2, H, W))[1]


# ---------- predictor factories ----------

def _make_dino_predictor(ckpt_path: Path, device, log):
    """Return a function (img_path) -> prob_map for combo_01 model."""
    import torch
    from ._common import build_m2f_lite_decoder
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = state.get("params", {})
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
    backbone.to(device).eval()
    input_size = (int(params.get("input_size", 518)) // 14) * 14
    n_patches = input_size // 14
    decoder = build_m2f_lite_decoder(
        in_dim=1024,
        num_queries=int(params.get("num_queries", 100)),
        mask_feat=int(params.get("mask_feat_dim", 256)),
        num_classes=1,
    ).to(device)
    decoder._mod.load_state_dict(state["decoder_state"])
    decoder.eval()

    def predict(img_path):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(img_path).convert("RGB").resize((input_size, input_size), Image.BILINEAR)
        t = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            out = backbone.forward_features(t)
            tokens = out["x_norm_patchtokens"].transpose(1, 2).reshape(1, 1024, n_patches, n_patches)
            logits = decoder(tokens)
            logits = torch.nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return prob
    return predict


def _make_sam2_encoder_predictor(ckpt_path: Path, device, log):
    """Return a function (img_path) -> prob_map for combo_03 model."""
    import torch
    from ._common import build_m2f_lite_decoder
    from .combo_03_sam2_encoder_m2f import _load_sam_encoder

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = state.get("params", {})
    backbone, dim, extract, info = _load_sam_encoder(log, device, Path("."))
    if backbone is None:
        raise RuntimeError(f"SAM encoder unavailable: {info}")
    backbone.to(device).eval()
    input_size = int(params.get("input_size", 512))
    decoder = build_m2f_lite_decoder(
        in_dim=dim,
        num_queries=int(params.get("num_queries", 100)),
        mask_feat=256, num_classes=1,
    ).to(device)
    decoder._mod.load_state_dict(state["decoder_state"])
    decoder.eval()

    def predict(img_path):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(img_path).convert("RGB").resize((input_size, input_size), Image.BILINEAR)
        t = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            feats = extract(backbone, t)
            logits = decoder(feats)
            logits = torch.nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return prob
    return predict


def _make_segformer_predictor(ckpt_path: Path, device, log):
    """Return a function (img_path) -> prob_map for combo_09's SegFormer-B5.

    Handles ALL supported checkpoint shapes:
      1. HF `save_pretrained/` directory (has config.json)
      2. .pth with {"model_state_dict": ...}  (combo_09 self-contained format)
      3. .pth with raw state_dict
      4. .pth with {"backbone_state": ..., "decoder_state": ...}  (combo_10 format)
    """
    import torch
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
    except ImportError as e:
        raise RuntimeError(f"transformers required for SegFormer predictor: {e}")

    base = "nvidia/segformer-b5-finetuned-ade-640-640"
    model = None

    # Case 1: HF save_pretrained directory
    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_path)).to(device).eval()
            log(f"Loaded SegFormer from HF save_pretrained dir: {ckpt_path}")
        except Exception as e:
            log(f"HF dir load failed ({e}); falling back to pth")

    # Case 1b: file next to a config.json -> treat as HF dir
    if model is None and ckpt_path.is_file() and (ckpt_path.parent / "config.json").exists():
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_path.parent)).to(device).eval()
            log(f"Loaded SegFormer from HF dir (parent of {ckpt_path.name})")
        except Exception:
            pass

    # Case 2/3/4: .pth variants
    if model is None and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SegformerForSemanticSegmentation.from_pretrained(
            base, num_labels=2, ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "fence"},
            label2id={"background": 0, "fence": 1},
        )
        loaded = False
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    missing, unexpected = model.load_state_dict(state[key], strict=False)
                    log(f"Loaded {key}; missing={len(missing)} unexpected={len(unexpected)}")
                    loaded = True
                    break
            if not loaded:
                # try whole-dict as state_dict
                try:
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    log(f"Loaded raw state_dict; missing={len(missing)} unexpected={len(unexpected)}")
                    loaded = True
                except Exception:
                    pass
            # combo_10 format: separate backbone/decoder
            if not loaded and "backbone_state" in state:
                try:
                    model.segformer.load_state_dict(state["backbone_state"], strict=False)
                    log("Loaded combo_10 backbone_state into SegFormer encoder (decoder ignored)")
                    loaded = True
                except Exception as e:
                    log(f"combo_10 backbone load failed: {e}")
        if not loaded:
            raise RuntimeError(f"Could not load SegFormer weights from {ckpt_path}")
        model = model.to(device).eval()

    if model is None:
        raise RuntimeError(f"Could not resolve SegFormer checkpoint at {ckpt_path}")

    input_size = 640

    def predict(img_path):
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(img_path).convert("RGB").resize((input_size, input_size), Image.BILINEAR)
        t = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pixel_values=t)
            logits = out.logits
            up = torch.nn.functional.interpolate(logits, size=input_size,
                                                 mode="bilinear", align_corners=False)
            prob = torch.softmax(up, dim=1)[0, 1].cpu().numpy()
        return prob
    return predict
