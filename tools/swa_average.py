"""tools/swa_average.py — Stochastic Weight Averaging (offline).

Averages the model weights of N checkpoints into a single "averaged" model
that's typically more robust + slightly higher IoU than any individual
checkpoint. Standard practice for the final shippable model:

    1. Take the last N periodic / best checkpoints
    2. Element-wise average their weights
    3. Use the averaged weights for inference

Usage:
    # Average the last 3 periodic checkpoints from phase 2
    python -m tools.swa_average ^
        --checkpoints outputs/training_v2/phase2/checkpoints/epoch_045.pt ^
                      outputs/training_v2/phase2/checkpoints/epoch_048.pt ^
                      outputs/training_v2/phase2/checkpoints/best.pt ^
        --output outputs/training_v2/phase2/checkpoints/swa_inference.pt

    # Or average everything in a directory (epoch_*.pt)
    python -m tools.swa_average ^
        --checkpoint-dir outputs/training_v2/phase2/checkpoints ^
        --pattern "epoch_*.pt" ^
        --output outputs/training_v2/phase2/checkpoints/swa_inference.pt

Notes:
  - Only averages tensor params with same shape/dtype across all inputs
  - Skips integer counters and other non-floating tensors
  - Output is a weights-only checkpoint (loadable by FencePredictor /
    eval_checkpoint / export_onnx exactly like best_inference.pt)
  - Inherits meta from the FIRST checkpoint (backbone_name, image_size, etc.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _load_state_dict(path: Path) -> tuple[dict, dict, dict, dict]:
    """Load the model state_dict + meta + config + provenance from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError(f"{path} has no 'model' key (got: {list(payload.keys())})")
    model = payload["model"]
    meta = payload.get("meta", {})
    config = payload.get("config", {})
    prov = payload.get("provenance", {})
    return model, meta, config, prov


def average_checkpoints(paths: list[Path]) -> tuple[dict, dict, dict, dict]:
    """Element-wise average the weights of N checkpoints.
    Returns (averaged_state_dict, meta_from_first, config_from_first, prov_merged)."""
    if not paths:
        raise ValueError("No checkpoints to average")

    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    # Load the first to set up the averaging buffer
    sd0, meta, config0, prov0 = _load_state_dict(paths[0])
    avg: dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in sd0.items():
        if v.dtype.is_floating_point:
            avg[k] = v.detach().clone().float()
        else:
            # Integer / bool tensors (counters, BN num_batches_tracked) — keep
            # the value from the FIRST checkpoint as-is, can't be averaged.
            avg[k] = v.detach().clone()
            skipped.append(k)

    # Sum the rest into the buffer
    for path in paths[1:]:
        sd, _, _, _ = _load_state_dict(path)
        for k, v in sd.items():
            if k not in avg:
                continue
            if v.dtype.is_floating_point:
                if avg[k].shape != v.shape:
                    print(f"  [skip] shape mismatch {k}: "
                          f"{tuple(avg[k].shape)} vs {tuple(v.shape)}")
                    continue
                avg[k].add_(v.detach().float())

    # Divide by N to get the mean (only for floating tensors)
    n = float(len(paths))
    for k, v in list(avg.items()):
        if v.dtype.is_floating_point:
            avg[k] = (v / n).to(dtype=sd0[k].dtype)   # cast back to original dtype

    if skipped:
        print(f"  ({len(skipped)} non-float tensors kept from first ckpt: e.g. "
              f"{skipped[:3]})")

    return avg, meta, config0, prov0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stochastic Weight Averaging — average N checkpoints "
                    "into one robust inference model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoints", nargs="+", type=Path, default=None,
                     help="Explicit list of checkpoints to average.")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                     help="Directory of checkpoints to glob.")
    ap.add_argument("--pattern", default="epoch_*.pt",
                     help="Glob pattern for --checkpoint-dir (default 'epoch_*.pt').")
    ap.add_argument("--output", type=Path, required=True,
                     help="Path to write the averaged inference checkpoint.")
    ap.add_argument("--include-best", action="store_true",
                     help="If using --checkpoint-dir, also include 'best.pt' "
                          "in the average.")
    args = ap.parse_args()

    if args.checkpoints:
        paths = sorted(args.checkpoints)
    elif args.checkpoint_dir:
        paths = sorted(args.checkpoint_dir.glob(args.pattern))
        if args.include_best:
            best = args.checkpoint_dir / "best.pt"
            if best.exists():
                paths.append(best)
    else:
        print("ERROR: pass either --checkpoints or --checkpoint-dir",
              file=sys.stderr)
        return 1

    if len(paths) < 2:
        print(f"WARNING: only {len(paths)} checkpoint(s) to average — "
              f"SWA needs >= 2 to be meaningful", file=sys.stderr)

    t0 = time.time()
    avg_sd, meta, config_dict, prov = average_checkpoints(paths)

    # Add SWA-specific provenance and re-collect "now" environment info
    try:
        from training import provenance as _provenance
        swa_prov = _provenance.collect()
    except Exception:
        swa_prov = {}
    swa_prov["swa"] = {
        "source_count": len(paths),
        "sources": [p.name for p in paths],
        "averaged_at": swa_prov.get("created_at"),
        "source_provenance_first": prov,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "model": avg_sd,
        "inference_only": True,
        "meta": {**meta, "swa_source_count": len(paths),
                 "swa_sources": [p.name for p in paths]},
        "provenance": swa_prov,
    }
    if config_dict:
        payload["config"] = config_dict
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(args.output)
    sz_mb = args.output.stat().st_size / 1e6
    print(f"\nWrote {args.output}  ({sz_mb:.1f} MB) in {time.time()-t0:.1f}s")
    print("\nUse it like any other inference checkpoint:")
    print(f"  python -m training.infer --checkpoint {args.output} ...")
    print(f"  python -m tools.eval_checkpoint --checkpoint {args.output} ...")
    print(f"  python -m tools.export_onnx --checkpoint {args.output} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
