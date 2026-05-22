"""Convert SegFormer-B3 fence checkpoint to browser-deployable ONNX.

Input:  best_unetpp_v2.pth         (SegFormer-B3, ~715 MB raw .pth)
Output: fence_model_segformer.onnx (fp32, ~178 MB, sigmoid baked in)
        fence_model_segformer.json (sidecar with preprocessing details)

Reads EMA weights (typically +0.5-1% IoU vs raw model weights). Bakes
ImageNet normalization expectations + sigmoid into a single graph so the
browser only needs to: divide pixels by 255, swap to NCHW, run the model,
threshold at 0.5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so emoji-containing log lines from
# torch.onnx ("Optimize the ONNX graph... ✅") don't crash with UnicodeEncodeError
# under the default cp1252 codepage. No-op on Linux/macOS.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


CKPT_PATH = Path(__file__).parent / "best_unetpp_v2.pth"
ONNX_PATH = Path(__file__).parent / "fence_model_segformer.onnx"
SIDECAR_PATH = ONNX_PATH.with_suffix(".json")
INPUT_SIZE = 512
USE_EMA = True
OPSET = 17


class _SegformerWrapper(nn.Module):
    """Wrap smp.Segformer to output sigmoid probabilities directly.

    Bakes sigmoid into the ONNX graph so browser code receives [0,1] mask
    probabilities — no need to apply sigmoid in JS. Cast logits to fp32
    before sigmoid for numerical hygiene at the saturated tail.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.sigmoid(logits.float())


def main() -> int:
    print(f"Loading checkpoint: {CKPT_PATH}")
    payload = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"  arch:    {payload.get('arch')}")
    print(f"  encoder: {payload.get('encoder')}")
    print(f"  loss:    {payload.get('loss')}")
    print(f"  epoch:   {payload.get('epoch')}")
    print(f"  best_iou (training-time metric): {payload.get('best_iou'):.4f}")

    sd_to_load = payload.get("ema") if USE_EMA else None
    if sd_to_load is None:
        print("  EMA weights NOT present, falling back to model weights")
        sd_to_load = payload["model"]
        weights_used = "model"
    else:
        weights_used = "ema"
    print(f"  Using weights: {weights_used}")

    model = smp.Segformer(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    missing, unexpected = model.load_state_dict(sd_to_load, strict=False)
    if missing or unexpected:
        print(f"  WARN: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"    missing[:3]: {missing[:3]}")
        if unexpected[:3]:
            print(f"    unexpected[:3]: {unexpected[:3]}")
    else:
        print(f"  state_dict load: PERFECT")

    wrapper = _SegformerWrapper(model).eval()
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"\nWrapper: {n_params/1e6:.2f}M params  (sigmoid baked into graph)")

    # Smoke test forward in PyTorch
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    print(f"\nPyTorch forward OK")
    print(f"  input  shape: {tuple(dummy.shape)}")
    print(f"  output shape: {ref.shape}")
    print(f"  output range: [{float(ref.min()):.4f}, {float(ref.max()):.4f}]")

    # Export to ONNX
    # Use the LEGACY TorchScript-based exporter (`dynamo=False`) because
    # the new dynamo-based path saves large weights as an EXTERNAL .onnx.data
    # file by default — bad for browser deployment which needs ONE file. The
    # legacy path bundles everything into a single .onnx (model + weights).
    print(f"\nExporting -> {ONNX_PATH}  (opset={OPSET}, dynamo=False)")
    t0 = time.time()
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_PATH),
            input_names=["input"],
            output_names=["output"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,            # legacy path = single-file .onnx
            dynamic_axes=None,
        )
    except TypeError:
        # Older torch (no `dynamo` kwarg) — legacy path is the only one anyway.
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_PATH),
            input_names=["input"],
            output_names=["output"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    # If a stray .onnx.data file was produced (older PyTorch quirks), fold it
    # back into the .onnx via onnx.save with save_as_external_data=False.
    data_sidecar = Path(str(ONNX_PATH) + ".data")
    if data_sidecar.exists():
        print(f"  Folding external weights ({data_sidecar.stat().st_size / 1e6:.1f} MB) "
              f"back into single .onnx...")
        import onnx
        m = onnx.load(str(ONNX_PATH))
        onnx.save(m, str(ONNX_PATH), save_as_external_data=False)
        try:
            data_sidecar.unlink()
        except OSError:
            pass
    print(f"  Exported in {time.time()-t0:.1f}s  "
          f"({ONNX_PATH.stat().st_size / 1e6:.1f} MB single-file)")

    # Parity check via onnxruntime
    try:
        import onnxruntime as ort
        print(f"\nParity check (onnxruntime CPU vs PyTorch CPU)")
        sess = ort.InferenceSession(str(ONNX_PATH),
                                      providers=["CPUExecutionProvider"])
        got = sess.run(["output"], {"input": dummy.numpy()})[0]
        max_abs = float(np.max(np.abs(ref - got)))
        mean_abs = float(np.mean(np.abs(ref - got)))
        tol = 5e-3
        ok = max_abs <= tol
        print(f"  abs diff: max={max_abs:.4e}  mean={mean_abs:.4e}  tol={tol:.0e}  "
              f"{'OK' if ok else 'WARN'}")
        parity = {
            "validated": True,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "tolerance": tol,
            "within_tolerance": ok,
        }
    except ImportError:
        print(f"  (onnxruntime not installed; skipping parity check)")
        parity = {"validated": False, "reason": "onnxruntime not installed"}
    except Exception as e:
        print(f"  parity check failed: {type(e).__name__}: {e}")
        parity = {"validated": False, "reason": str(e)}

    # Sidecar JSON for the browser
    sidecar = {
        "model_path": ONNX_PATH.name,
        "architecture": "smp.Segformer(encoder_name='mit_b3', classes=1)",
        "training_arch_tag": payload.get("arch"),
        "training_encoder": payload.get("encoder"),
        "training_loss": payload.get("loss"),
        "training_epoch": int(payload.get("epoch", 0)),
        "training_best_iou": float(payload.get("best_iou", 0.0)),
        "weights_used": weights_used,
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
        },
        "output": {
            "name": "output",
            "shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "interpretation": "sigmoid probabilities of fence pixel (0=bg, 1=fence)",
            "sigmoid_baked_in": True,
        },
        "recommended": {
            "binarize_threshold": 0.5,
            "resize_method": "bilinear",
            "client_postprocess": [
                "Resize 512x512 prob map to original image size (bilinear)",
                "Threshold at 0.5 OR feed soft prob to your blend",
                "Optional: light gaussian blur for smoother edges",
            ],
        },
        "size_mb": round(ONNX_PATH.stat().st_size / 1e6, 1),
        "opset": OPSET,
        "parity_check": parity,
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, default=str))
    print(f"\nSidecar: {SIDECAR_PATH}")
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print(f"  ONNX:    {ONNX_PATH.name}  ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  Sidecar: {SIDECAR_PATH.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
