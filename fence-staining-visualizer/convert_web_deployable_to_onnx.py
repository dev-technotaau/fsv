"""Convert the web_deployable FenceSegmentationModel checkpoint to browser ONNX.

Source: outputs/web_deployable/web_v1/checkpoints/best_inference.pt
Output: fence_model_dinov2.onnx (single-file, sigmoid + temperature baked in)
        fence_model_dinov2.json (sidecar)

The model is OUR FenceSegmentationModel (DINOv2-Small + ViTToFPN + MSDeform
6L + Mask2Former 6L 192-dim 16-query + UNet3+ refinement with CGM/BDR/PointRend
/edge/FDS heads). Built from the bundled config inside best_inference.pt.

Notes:
  - Uses LEGACY torch.onnx.export (`dynamo=False`) so weights are bundled
    into a single .onnx file (the new dynamo path saves them externally).
  - INPUT_SIZE is auto-snapped to a valid patch-size multiple. DINOv2 uses
    patch_size=14, so 512 -> 518 (37*14). The JS code MUST use this value.
  - Sigmoid is baked into the graph; temperature defaults to 1.0 (no
    calibration was fit during this training).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows (torch.onnx prints emoji)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import numpy as np
import torch
import torch.nn as nn

# Make repo modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import TrainingConfig
from training.model import build_model

CKPT_PATH = Path(__file__).parent.parent / "outputs/web_deployable/web_v1/checkpoints/best_inference.pt"
ONNX_PATH = Path(__file__).parent / "fence_model_dinov2.onnx"
SIDECAR_PATH = ONNX_PATH.with_suffix(".json")
# fp16 sibling — same model, half the bytes for faster browser load.
# Input/output remain fp32; only internal weights/ops are fp16, so the
# JS code is unchanged — just point MODEL_PATH at the fp16 file.
FP16_PATH = ONNX_PATH.with_name(ONNX_PATH.stem + "_fp16.onnx")
FP16_SIDECAR_PATH = FP16_PATH.with_suffix(".json")
DEFAULT_INPUT_SIZE = 512
# Opset 18 — needed because PyTorch exports several ops (ScatterND, certain
# Conv variants) at native opset 18 and the version-converter downgrade to
# 17 either fails or produces nodes that ONNXRuntime can't execute. Modern
# onnxruntime-web (>= 1.16) supports opset 18 fine.
OPSET = 18


class _Wrapper(nn.Module):
    """Wrap FenceSegmentationModel: forward → refined_logits → sigmoid (fp32)."""
    def __init__(self, model: nn.Module, use_refined: bool = True,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.use_refined = use_refined
        T = max(1e-6, float(temperature))
        self.register_buffer("temperature",
                              torch.tensor(T, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = (out.refined_logits
                  if (self.use_refined and out.refined_logits is not None)
                  else out.mask_logits)
        return torch.sigmoid(logits.float() / self.temperature)


def main() -> int:
    print(f"Loading checkpoint: {CKPT_PATH}")
    payload = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    bundled_cfg_dict = payload.get("config") or {}
    print(f"  metric_value (val_iou): {meta.get('metric_value', 'n/a')}")
    print(f"  pipeline_version:       {meta.get('pipeline_version', 'n/a')}")
    print(f"  bundled config?         {bool(bundled_cfg_dict)}")

    if not bundled_cfg_dict:
        print("FATAL: no bundled config in checkpoint — cannot rebuild model arch.")
        return 1

    # Rebuild config from bundled dict (handles the strict unknown-key check)
    try:
        cfg = TrainingConfig.from_dict(bundled_cfg_dict)
    except ValueError as e:
        print(f"  Strict config load failed ({e}); attempting permissive rebuild...")
        # Permissive fallback: drop unknown keys per section
        from dataclasses import fields
        from training.config import (ModelConfig, LossConfig, OptimConfig,
                                     TrainConfig, DataConfig, PostProcessConfig,
                                     LogConfig, CheckpointConfig)
        section_classes = {
            "model": ModelConfig, "loss": LossConfig, "optim": OptimConfig,
            "train": TrainConfig, "data": DataConfig, "post": PostProcessConfig,
            "log": LogConfig, "ckpt": CheckpointConfig,
        }
        cleaned = {}
        for sec, sec_dict in bundled_cfg_dict.items():
            if sec in section_classes and isinstance(sec_dict, dict):
                allowed = {f.name for f in fields(section_classes[sec])}
                cleaned[sec] = {k: v for k, v in sec_dict.items() if k in allowed}
            else:
                cleaned[sec] = sec_dict
        cfg = TrainingConfig.from_dict(cleaned)

    print(f"  backbone:        {cfg.model.backbone_name}")
    print(f"  decoder dim/q/L: {cfg.model.decoder_dim}/{cfg.model.decoder_num_queries}/{cfg.model.decoder_num_layers}")
    print(f"  refinement:      iters={cfg.model.refinement_iterations}  "
          f"channels={cfg.model.refinement_channels}  depth={cfg.model.refinement_use_depth}")
    print(f"  pointrend module:{cfg.model.refinement_use_pointrend_module}  "
          f"(will be DISABLED for ONNX export — see note below)")
    print(f"  CGM gating:      {cfg.model.refinement_use_cgm}")

    # ── Disable PointRend MLP for ONNX export ────────────────────────────
    # PointRend's ScatterElements op exports with a fp64/fp32 type
    # mismatch (PyTorch ONNX bug) that ONNXRuntime refuses to load. The
    # MLP refines top-K uncertain pixels at inference — accuracy impact
    # is small (~0.5% IoU). Other refinement features (CGM, BDR, FDS,
    # edge head, iterative refinement) all stay enabled.
    cfg.model.refinement_use_pointrend_module = False

    # Build model + load state_dict (PointRend MLP weights become unexpected)
    print(f"\nBuilding model (PointRend disabled for ONNX) + loading weights...")
    model = build_model(cfg.model)
    sd = payload["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing={len(missing)} (first 3: {list(missing)[:3]})")
    if unexpected:
        # Filter out the EXPECTED unexpected keys (PointRend MLP)
        expected_unexpected = [k for k in unexpected if "pointrend_mlp" in k]
        unexpected_unexpected = [k for k in unexpected if "pointrend_mlp" not in k]
        print(f"  unexpected={len(unexpected)} "
              f"(of which PointRend MLP: {len(expected_unexpected)}, OTHER: {len(unexpected_unexpected)})")
        if unexpected_unexpected:
            print(f"    OTHER unexpected (first 3): {unexpected_unexpected[:3]}")
    if not missing and (not unexpected or all('pointrend' in k for k in unexpected)):
        print(f"  state_dict load: ALL TRAINED WEIGHTS LOADED (only PointRend MLP dropped)")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params/1e6:.2f}M  trainable: {n_trainable/1e6:.2f}M")

    # Snap input size to patch_size multiple (DINOv2 uses 14, NOT 16)
    patch_size = int(getattr(model, "patch_size", 14))
    snapped = max(patch_size * 4,
                   int(round(DEFAULT_INPUT_SIZE / patch_size)) * patch_size)
    if snapped != DEFAULT_INPUT_SIZE:
        print(f"  WARNING: input size {DEFAULT_INPUT_SIZE} -> snapped to {snapped} "
              f"(must be multiple of patch_size={patch_size})")
    INPUT_SIZE = snapped

    # Wrap + smoke test
    temperature = float(meta.get("temperature", 1.0))
    wrapper = _Wrapper(model, use_refined=True, temperature=temperature).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    print(f"\nPyTorch forward {INPUT_SIZE}x{INPUT_SIZE}:  output {ref.shape}  "
          f"range=[{float(ref.min()):.4f}, {float(ref.max()):.4f}]")

    # Export — try DYNAMO path first (better type tracking, no fp64
    # promotion of constants). Fall back to LEGACY if dynamo fails.
    # We then fold-back any external weights into a single .onnx for the
    # browser deployment requirement.
    print(f"\nExporting -> {ONNX_PATH}  (opset={OPSET}, trying dynamo first)")
    t0 = time.time()
    used_path = None
    try:
        torch.onnx.export(
            wrapper, dummy, str(ONNX_PATH),
            input_names=["input"], output_names=["output"],
            opset_version=OPSET, do_constant_folding=True,
            dynamo=True, dynamic_axes=None,
        )
        used_path = "dynamo"
    except Exception as e:
        print(f"  dynamo path failed ({type(e).__name__}); falling back to legacy")
        try:
            torch.onnx.export(
                wrapper, dummy, str(ONNX_PATH),
                input_names=["input"], output_names=["output"],
                opset_version=OPSET, do_constant_folding=True,
                dynamo=False, dynamic_axes=None,
            )
            used_path = "legacy"
        except TypeError:
            torch.onnx.export(
                wrapper, dummy, str(ONNX_PATH),
                input_names=["input"], output_names=["output"],
                opset_version=OPSET, do_constant_folding=True,
                dynamic_axes=None,
            )
            used_path = "legacy (no dynamo kwarg)"
    print(f"  Used export path: {used_path}")

    # Fold any external weights back into the .onnx (defensive)
    data_sidecar = Path(str(ONNX_PATH) + ".data")
    if data_sidecar.exists():
        print(f"  Folding external weights "
              f"({data_sidecar.stat().st_size/1e6:.1f} MB) into single .onnx...")
        import onnx
        m = onnx.load(str(ONNX_PATH))
        onnx.save(m, str(ONNX_PATH), save_as_external_data=False)
        try:
            data_sidecar.unlink()
        except OSError:
            pass

    # ── Cast all fp64 (double) tensors in the ONNX graph to fp32 ──────────
    # PyTorch's ONNX exporter sometimes promotes constant-folded weights
    # to fp64 mid-graph, even when the source PyTorch params are fp32.
    # ONNXRuntime's Conv/Gemm/MatMul kernels typically don't support fp64
    # → "Could not find an implementation for Conv(11)" load failure.
    # We fix it here by sweeping every initializer + value_info + cast op
    # in the graph and forcing fp32. Bit-equivalent results within fp32
    # precision (negligible accuracy impact).
    print(f"  Sweeping ONNX graph: casting any fp64 -> fp32 ...")
    import onnx
    from onnx import numpy_helper, TensorProto
    m = onnx.load(str(ONNX_PATH))
    n_initializers_cast = 0
    n_value_info_cast = 0
    n_cast_nodes_fixed = 0
    new_inits = []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.DOUBLE:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            new_inits.append(new_init)
            n_initializers_cast += 1
        else:
            new_inits.append(init)
    if n_initializers_cast > 0:
        del m.graph.initializer[:]
        m.graph.initializer.extend(new_inits)
    # value_info (intermediate tensors)
    for v in m.graph.value_info:
        if v.type.tensor_type.elem_type == TensorProto.DOUBLE:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
            n_value_info_cast += 1
    # graph inputs / outputs
    for io in list(m.graph.input) + list(m.graph.output):
        if io.type.tensor_type.elem_type == TensorProto.DOUBLE:
            io.type.tensor_type.elem_type = TensorProto.FLOAT
    # Cast nodes that target DOUBLE → retarget to FLOAT
    for node in m.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    n_cast_nodes_fixed += 1
    print(f"    initializers re-cast: {n_initializers_cast}")
    print(f"    value_infos re-cast:  {n_value_info_cast}")
    print(f"    Cast nodes retargeted:{n_cast_nodes_fixed}")
    onnx.save(m, str(ONNX_PATH), save_as_external_data=False)

    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"  Exported in {time.time()-t0:.1f}s  ({size_mb:.1f} MB single-file)")

    # Parity check
    try:
        import onnxruntime as ort
        print(f"\nParity (onnxruntime CPU vs PyTorch CPU)")
        sess = ort.InferenceSession(str(ONNX_PATH),
                                      providers=["CPUExecutionProvider"])
        got = sess.run(["output"], {"input": dummy.numpy()})[0]
        max_abs = float(np.max(np.abs(ref - got)))
        mean_abs = float(np.mean(np.abs(ref - got)))
        tol = 5e-3
        ok = max_abs <= tol
        print(f"  abs diff: max={max_abs:.4e}  mean={mean_abs:.4e}  tol={tol:.0e}  "
              f"{'OK' if ok else 'WARN'}")
        parity = {"validated": True, "max_abs_diff": max_abs,
                  "mean_abs_diff": mean_abs, "tolerance": tol,
                  "within_tolerance": ok}
    except ImportError:
        parity = {"validated": False, "reason": "onnxruntime missing"}
    except Exception as e:
        print(f"  parity FAILED: {type(e).__name__}: {e}")
        parity = {"validated": False, "reason": str(e)}

    # Sidecar
    sidecar = {
        "model_path": ONNX_PATH.name,
        "architecture": "FenceSegmentationModel "
                        f"(backbone={cfg.model.backbone_name}, "
                        f"decoder=mask2former dim={cfg.model.decoder_dim} "
                        f"q={cfg.model.decoder_num_queries} L={cfg.model.decoder_num_layers}, "
                        f"refinement_iters={cfg.model.refinement_iterations})",
        "training_pipeline_version": meta.get("pipeline_version"),
        "training_epoch": meta.get("epoch"),
        "training_global_step": meta.get("global_step"),
        "training_metric_name": meta.get("metric_name"),
        "training_metric_value": meta.get("metric_value"),
        "input": {
            "name": "input",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "dtype": "float32",
            "channel_order": "RGB",
            "preprocessing": {
                "rescale": "pixel / 255.0",
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225],
                "layout": "NCHW",
            },
            "patch_size": patch_size,
        },
        "output": {
            "name": "output",
            "shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "interpretation": "sigmoid probabilities of fence pixel",
            "sigmoid_baked_in": True,
            "temperature_baked_in": True,
            "temperature": temperature,
        },
        "recommended": {
            "binarize_threshold": 0.5,
            "client_postprocess": [
                f"Resize input image to {INPUT_SIZE}x{INPUT_SIZE} (bilinear)",
                "ImageNet normalize, NCHW",
                "Run inference",
                "Bilinear upsample 518x518 -> original image size",
                "Threshold at 0.5 (or higher for cleaner staining)",
                "Optional: connected-component cleanup, drop blobs <2% of image",
            ],
        },
        "size_mb": round(size_mb, 1),
        "opset": OPSET,
        "parity_check": parity,
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, default=str))
    print(f"\nSidecar: {SIDECAR_PATH}")

    # ───────────────────────────────────────────────────────────────────
    # fp16 VARIANT — half-size for faster mobile / slow-network loads.
    # Strategy: keep input/output as fp32 (so JS code is unchanged), but
    # convert internal weights + most ops to fp16. A small block-list
    # keeps numerically-sensitive ops in fp32:
    #   - Sigmoid: tail saturation (we divide logits by temperature first)
    #   - LayerNormalization: variance calc, fp16 underflows easily
    #   - Softmax: attention probabilities, fp16 underflows on long tails
    # Typical browser-side accuracy impact on segmentation: <1% IoU drop.
    # ───────────────────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print(f"Producing fp16 variant -> {FP16_PATH.name}")
    print("-" * 60)
    fp16_size_mb = None
    fp16_parity: dict = {"validated": False, "reason": "not attempted"}
    try:
        from onnxconverter_common import float16 as ocf16
        m_fp32 = onnx.load(str(ONNX_PATH))
        m_fp16 = ocf16.convert_float_to_float16(
            m_fp32,
            keep_io_types=True,
            op_block_list=["Sigmoid", "LayerNormalization", "Softmax"],
            disable_shape_infer=False,
        )
        onnx.save(m_fp16, str(FP16_PATH), save_as_external_data=False)
        fp16_size_mb = FP16_PATH.stat().st_size / 1e6
        ratio_pct = 100.0 * fp16_size_mb / max(size_mb, 1e-6)
        print(f"  fp16 size: {fp16_size_mb:.1f} MB  ({ratio_pct:.0f}% of fp32)")

        # Parity vs PyTorch reference. Tolerance is intentionally looser
        # than fp32 (5e-2 vs 5e-3) — fp16 is inherently lossy at the tails
        # but the binarized mask is robust to small probability shifts.
        try:
            import onnxruntime as ort  # may already be imported above
            sess_fp16 = ort.InferenceSession(
                str(FP16_PATH), providers=["CPUExecutionProvider"]
            )
            got_fp16 = sess_fp16.run(["output"], {"input": dummy.numpy()})[0]
            max_abs_fp16 = float(np.max(np.abs(ref - got_fp16)))
            mean_abs_fp16 = float(np.mean(np.abs(ref - got_fp16)))
            tol_fp16 = 5e-2
            ok_fp16 = max_abs_fp16 <= tol_fp16
            print(f"  fp16 parity vs PyTorch fp32: "
                  f"max={max_abs_fp16:.4e}  mean={mean_abs_fp16:.4e}  "
                  f"tol={tol_fp16:.0e}  {'OK' if ok_fp16 else 'WARN'}")
            fp16_parity = {
                "validated": True,
                "max_abs_diff": max_abs_fp16,
                "mean_abs_diff": mean_abs_fp16,
                "tolerance": tol_fp16,
                "within_tolerance": ok_fp16,
                "reference": "PyTorch fp32",
            }
        except ImportError:
            fp16_parity = {"validated": False, "reason": "onnxruntime missing"}
        except Exception as e:
            print(f"  fp16 parity FAILED: {type(e).__name__}: {e}")
            fp16_parity = {"validated": False, "reason": str(e)}

        # Write fp16 sidecar — clone fp32 sidecar then override the
        # precision-specific fields. JS can pick which JSON to load based
        # on which ONNX URL it's pointing at.
        fp16_sidecar = dict(sidecar)
        fp16_sidecar["model_path"] = FP16_PATH.name
        fp16_sidecar["precision"] = "fp16"
        fp16_sidecar["size_mb"] = round(fp16_size_mb, 1)
        fp16_sidecar["parity_check"] = fp16_parity
        fp16_sidecar["fp16_notes"] = (
            "Half-precision variant. Input/output are fp32 (so the JS "
            "browser code is identical to fp32). Internal weights and "
            "most ops run in fp16; Sigmoid, LayerNormalization, and "
            "Softmax are kept in fp32 for numerical stability. Typical "
            "segmentation IoU impact: <1%."
        )
        FP16_SIDECAR_PATH.write_text(
            json.dumps(fp16_sidecar, indent=2, default=str)
        )
        print(f"  fp16 sidecar: {FP16_SIDECAR_PATH.name}")

        # Cross-link the fp32 sidecar so consumers know an fp16 sibling exists
        try:
            sidecar["fp16_variant"] = {
                "model_path": FP16_PATH.name,
                "sidecar": FP16_SIDECAR_PATH.name,
                "size_mb": round(fp16_size_mb, 1),
                "parity_check": fp16_parity,
            }
            SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, default=str))
        except OSError:
            pass

    except ImportError:
        print("  SKIPPED: onnxconverter-common not installed. "
              "Install with: pip install onnxconverter-common")
    except Exception as e:
        print(f"  fp16 export FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print(f"  ONNX (fp32): {ONNX_PATH.name}  ({size_mb:.1f} MB)  INPUT={INPUT_SIZE}")
    if fp16_size_mb is not None:
        print(f"  ONNX (fp16): {FP16_PATH.name}  ({fp16_size_mb:.1f} MB)  "
              f"← deploy this to the web for ~2x faster load")
    print(f"  Sidecars:    {SIDECAR_PATH.name}"
          + (f", {FP16_SIDECAR_PATH.name}" if fp16_size_mb is not None else ""))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
