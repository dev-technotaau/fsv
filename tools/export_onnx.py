"""tools/export_onnx.py — Export a trained checkpoint to ONNX.

Produces a single-input single-output ONNX file ready for:
  - browser inference via onnxruntime-web
  - server inference via onnxruntime
  - inspection via Netron

Quick start:

    python -m tools.export_onnx \
        --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt \
        --output models/fence_dinov3_phase1.onnx \
        --image-size 512

Optional flags:
  --opset 17           ONNX opset (17 is the default; 14+ required for ViT)
  --dynamic-batch      export dynamic batch dim (default: fixed batch=1)
  --use-refined        export the refined-mask output instead of coarse
                        (refined head adds ~6MB ONNX size + ~10ms inference)
  --quantize-dynamic   apply dynamic int8 quantization (smaller, CPU-faster)
  --validate           run a parity check between PyTorch and ONNX outputs
                        on a random input (recommended)

Notes:
  - Image must be (1, 3, image_size, image_size), float32, ImageNet-normalized.
  - Output is (1, 1, image_size, image_size) sigmoid probabilities in [0, 1].
  - DINOv3 ViT exports work with opset >= 14 (we default to 17 for safety).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import TrainingConfig
from training.model import build_model


class _OnnxWrapper(nn.Module):
    """Wrap FenceSegmentationModel to expose a single tensor output (sigmoid
    probabilities) — keeps the ONNX graph + the JS inference code simple."""
    def __init__(self, model: nn.Module, use_refined: bool) -> None:
        super().__init__()
        self.model = model
        self.use_refined = use_refined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = (out.refined_logits
                  if (self.use_refined and out.refined_logits is not None)
                  else out.mask_logits)
        return torch.sigmoid(logits)


def _load_checkpoint(checkpoint_path: Path,
                      config: Optional[TrainingConfig]
                      ) -> tuple[nn.Module, dict, int, dict, dict]:
    """Returns: (model, meta, patch_size, bundled_config_dict, provenance)."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    bundled_cfg = payload.get("config") or {}
    prov = payload.get("provenance") or {}
    # Resolve config: explicit arg > bundled > defaults+meta
    if config is None and bundled_cfg:
        try:
            config = TrainingConfig.from_dict(bundled_cfg)
            print(f"  Using bundled config from checkpoint "
                  f"(decoder_dim={config.model.decoder_dim}, "
                  f"refinement_use_depth={config.model.refinement_use_depth})")
        except Exception as e:
            print(f"  WARN: bundled config failed to parse "
                  f"({type(e).__name__}); falling back to defaults")
            config = None
    if config is None:
        config = TrainingConfig()
        if "backbone_name" in meta:
            config.model.backbone_name = meta["backbone_name"]
        if "image_size" in meta:
            config.data.image_size = int(meta["image_size"])
    model = build_model(config.model).eval()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    patch_size = int(getattr(model, "patch_size", 14))
    return model, meta, patch_size, bundled_cfg, prov


def export_onnx(checkpoint_path: str | Path,
                  output_path: str | Path,
                  image_size: int,
                  opset: int = 17,
                  dynamic_batch: bool = False,
                  use_refined: bool = False,
                  config: Optional[TrainingConfig] = None,
                  validate: bool = True,
                  quantize_dynamic: bool = False,
                  quantize_fp16: bool = False) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint_path}")
    model, meta, patch_size, bundled_cfg, prov = _load_checkpoint(checkpoint_path, config)
    print(f"  backbone     : {(config or TrainingConfig()).model.backbone_name}")
    print(f"  patch_size   : {patch_size}")
    print(f"  use_refined  : {use_refined}")

    # Snap image_size to a valid patch stride
    snapped = max(patch_size * 4,
                   int(round(image_size / patch_size)) * patch_size)
    if snapped != image_size:
        print(f"  WARNING: image_size {image_size} -> snapped to "
              f"{snapped} (multiple of patch_size={patch_size})")
        image_size = snapped

    wrapper = _OnnxWrapper(model, use_refined=use_refined).eval()
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    # Export
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "mask_prob": {0: "batch"}}

    print(f"Exporting -> {output_path}  (opset={opset}, "
          f"dynamic_batch={dynamic_batch})")
    t0 = time.time()
    torch.onnx.export(
        wrapper, dummy, str(output_path),
        input_names=["image"],
        output_names=["mask_prob"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    size_mb = output_path.stat().st_size / (1 << 20)
    print(f"  Exported in {time.time() - t0:.1f}s ({size_mb:.1f} MB)")

    # Capture exporter library versions (for sidecar) ── before validation
    # because the validator imports onnxruntime which we'd want to record too.
    exporter_versions: dict = {"torch": torch.__version__}
    try:
        import onnx as _onnx
        exporter_versions["onnx"] = _onnx.__version__
    except ImportError:
        pass
    try:
        import onnxscript as _onnxscript
        exporter_versions["onnxscript"] = getattr(_onnxscript, "__version__", "n/a")
    except ImportError:
        pass

    # Validate parity against PyTorch FIRST so we can include parity stats
    # in the sidecar JSON we're about to write.
    parity_stats: dict = {"validated": False}
    if validate:
        try:
            import onnxruntime as ort
            exporter_versions["onnxruntime"] = ort.__version__
        except ImportError:
            print("  [skip validate] onnxruntime not installed.")
        else:
            print("Validating ONNX vs PyTorch ...")
            with torch.no_grad():
                ref = wrapper(dummy).numpy()
            sess = ort.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"],
            )
            got = sess.run(["mask_prob"], {"image": dummy.numpy()})[0]
            max_abs = float(np.max(np.abs(ref - got)))
            mean_abs = float(np.mean(np.abs(ref - got)))
            tol = 5e-3   # ViT + bilinear interpolation can drift slightly
            ok = max_abs <= tol
            print(f"  abs diff: max={max_abs:.4e}  mean={mean_abs:.4e}")
            if ok:
                print("  Parity OK (within tolerance).")
            else:
                print(f"  WARNING: max abs diff {max_abs:.4e} > {tol:.0e}. "
                      f"Likely fine for inference but worth investigating.")
            parity_stats = {
                "validated": True,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "tolerance": tol,
                "within_tolerance": ok,
                "validator_provider": "CPUExecutionProvider",
            }

    # Add export-time provenance (when/where/with-what THIS export was done)
    try:
        from training import provenance as _provenance
        export_prov = _provenance.collect()
    except Exception:
        export_prov = {}

    # Sidecar metadata file — fully self-describing for downstream deployment
    sidecar = output_path.with_suffix(".json")
    eff_cfg = config or TrainingConfig.from_dict(bundled_cfg) if bundled_cfg else (config or TrainingConfig())
    suggested_post = {
        "use_morphology": getattr(eff_cfg.post, "use_morphology", True),
        "use_guided_filter": getattr(eff_cfg.post, "use_guided_filter", True),
        "use_dense_crf": getattr(eff_cfg.post, "use_dense_crf", True),
        "use_cc_cleanup": getattr(eff_cfg.post, "use_cc_cleanup", True),
        "cc_min_blob_area": getattr(eff_cfg.post, "cc_min_blob_area", 200),
        "crf_bilateral_sxy": getattr(eff_cfg.post, "crf_bilateral_sxy", 50),
        "crf_bilateral_srgb": getattr(eff_cfg.post, "crf_bilateral_srgb", 10),
        "crf_bilateral_compat": getattr(eff_cfg.post, "crf_bilateral_compat", 12),
    }
    sidecar_payload = {
        "checkpoint_meta": meta,
        "image_size": image_size,
        "patch_size": patch_size,
        "input_name": "image",
        "input_shape": [1 if not dynamic_batch else "batch", 3, image_size, image_size],
        "input_dtype": "float32",
        "input_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "channel_order": "RGB",
        },
        "output_name": "mask_prob",
        "output_shape": [1 if not dynamic_batch else "batch", 1, image_size, image_size],
        "output_dtype": "float32",
        "output_range": [0.0, 1.0],
        "use_refined": use_refined,
        "opset": opset,
        "size_mb": round(size_mb, 2),
        "exported_with": "tools/export_onnx.py",
        # Self-describing fields (mirror what's in the .pt checkpoint)
        "training_config": bundled_cfg or (config.to_dict() if config else None),
        "training_provenance": prov,
        "export_provenance": export_prov,
        "suggested_post_process": suggested_post,
        # Exporter / runtime versions used to produce + validate this ONNX
        "exporter_versions": exporter_versions,
        # ONNX-vs-PyTorch numerical parity check
        "parity_check": parity_stats,
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, default=str))
    print(f"  Sidecar: {sidecar}")

    # fp16 conversion (smaller, GPU-faster on Ampere+ / WebGPU; sweet spot
    # between fp32 and int8 quantization for accuracy/size tradeoff).
    if quantize_fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            print("  [skip fp16] requires `pip install onnxconverter-common`.")
        else:
            qpath = output_path.with_name(output_path.stem + "_fp16.onnx")
            print(f"Converting to fp16 -> {qpath}")
            model_onnx = onnx.load(str(output_path))
            model_fp16 = float16.convert_float_to_float16(
                model_onnx,
                keep_io_types=True,        # keep I/O as fp32 for compat with most runtimes
                disable_shape_infer=False,
            )
            onnx.save(model_fp16, str(qpath))
            qsize = qpath.stat().st_size / (1 << 20)
            print(f"  fp16: {qsize:.1f} MB ({qsize / size_mb * 100:.0f}% of fp32)")

    # Dynamic int8 quantization (CPU-only, smallest)
    if quantize_dynamic:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("  [skip quantize] onnxruntime quantization not available.")
        else:
            qpath = output_path.with_name(output_path.stem + "_int8.onnx")
            print(f"Quantizing dynamically (int8) -> {qpath}")
            quantize_dynamic(
                model_input=str(output_path),
                model_output=str(qpath),
                weight_type=QuantType.QUInt8,
            )
            qsize = qpath.stat().st_size / (1 << 20)
            print(f"  int8: {qsize:.1f} MB ({qsize / size_mb * 100:.0f}% of fp32)")

    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a trained checkpoint to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True,
                     help="Path to best.pt or best_inference.pt")
    ap.add_argument("--output", required=True,
                     help="Path to write the .onnx file (auto-creates parent dir)")
    ap.add_argument("--image-size", type=int, default=None,
                     help="Inference image size (default: from ckpt meta or 512)")
    ap.add_argument("--config", default=None,
                     help="Optional training YAML if checkpoint lacks meta")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--use-refined", action="store_true",
                     help="Export refined head output (slower, slightly better)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--quantize-fp16", action="store_true",
                     help="Also export an _fp16.onnx variant (~50% size, "
                          "GPU-friendly; needs `pip install onnxconverter-common`)")
    ap.add_argument("--quantize-dynamic", action="store_true",
                     help="Also export an _int8.onnx variant (~25% size, "
                          "CPU-friendly; via onnxruntime.quantization)")
    args = ap.parse_args()

    cfg = (TrainingConfig.from_yaml(args.config) if args.config else None)
    img_size = args.image_size
    if img_size is None:
        img_size = (cfg.data.image_size if cfg
                    else 512)

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=img_size,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        use_refined=args.use_refined,
        config=cfg,
        validate=not args.no_validate,
        quantize_dynamic=args.quantize_dynamic,
        quantize_fp16=args.quantize_fp16,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
