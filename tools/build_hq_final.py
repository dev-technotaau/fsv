#!/usr/bin/env python3
"""Build the HQ-tier final manifest + split files for Phase-2 fine-tuning.

After auto/manual review (export_final.py) you have:
  dataset/manifest_final.jsonl       — all 33K rows with class corrections
  dataset/splits/{train,val,test}.jsonl — fresh full-dataset splits

This script derives the HQ-tier subset (shorter edge >= --hq-threshold) and
writes:
  dataset/manifest_hq_final.jsonl
  dataset/splits/train_hq.jsonl  (subset of train.jsonl that's HQ)
  dataset/splits/val_hq.jsonl    (subset of val.jsonl that's HQ)
  dataset/splits/test_hq.jsonl   (subset of test.jsonl that's HQ)

By deriving _hq splits from the FULL split files (rather than re-stratifying
the HQ subset independently), HQ images stay on the same side of the
train/val/test boundary as their non-HQ siblings — preserves data integrity
across resolution tiers and avoids accidental train/test leakage if a
high-res copy and low-res copy of the same scene exist.

Usage:
    python tools/build_hq_final.py
    python tools/build_hq_final.py --hq-threshold 1280
    python tools/build_hq_final.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Handle read-only target (e.g., test_hq.jsonl chmod-ed 444 by previous run)
    if path.exists():
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest-final", type=Path,
                    default=Path("dataset/manifest_final.jsonl"))
    ap.add_argument("--out-manifest-hq", type=Path,
                    default=Path("dataset/manifest_hq_final.jsonl"))
    ap.add_argument("--splits-dir", type=Path,
                    default=Path("dataset/splits"))
    ap.add_argument("--hq-threshold", type=int, default=1024,
                    help="Images with min(width, height) >= this go in HQ "
                         "(default 1024, matches Phase-2 fine-tune resolution)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.manifest_final.exists():
        print(f"ERROR: {args.manifest_final} not found", file=sys.stderr)
        return 2

    # ── Load manifest_final + identify HQ subset ─────────────────────────
    print(f"Loading {args.manifest_final}")
    manifest = load_jsonl(args.manifest_final)
    print(f"  {len(manifest):,} rows")

    hq_rows = [r for r in manifest
               if min(r.get("width", 0), r.get("height", 0)) >= args.hq_threshold]
    hq_ids = {r["id"] for r in hq_rows}

    pos_count = sum(1 for r in hq_rows if r.get("class") == "pos")
    neg_count = sum(1 for r in hq_rows if r.get("class") == "neg")
    print(f"\nHQ subset (shorter edge >= {args.hq_threshold}px): "
          f"{len(hq_rows):,} rows  (pos={pos_count:,}, neg={neg_count:,})")
    print(f"  ({100*len(hq_rows)/max(len(manifest),1):.1f}% of full manifest)")

    # ── Filter each full split to HQ membership ──────────────────────────
    split_results: list[tuple[str, int, Path]] = []
    for split_name in ("train", "val", "test"):
        src = args.splits_dir / f"{split_name}.jsonl"
        if not src.exists():
            print(f"\nWARN: {src} not found — skipping {split_name}_hq derivation")
            continue
        rows = load_jsonl(src)
        hq_split = [r for r in rows if r["id"] in hq_ids]
        out = args.splits_dir / f"{split_name}_hq.jsonl"
        split_results.append((split_name, len(hq_split), out))
        print(f"  {split_name:>5s}.jsonl ({len(rows):>6,}) -> "
              f"{split_name}_hq.jsonl ({len(hq_split):>6,})")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    # ── Write HQ manifest ────────────────────────────────────────────────
    write_jsonl_atomic(hq_rows, args.out_manifest_hq)
    print(f"\nWrote {args.out_manifest_hq}  ({len(hq_rows):,} rows)")

    # ── Write HQ splits ──────────────────────────────────────────────────
    for split_name, n, out in split_results:
        # Re-load source + filter (to use atomic writes correctly on each)
        src = args.splits_dir / f"{split_name}.jsonl"
        rows = load_jsonl(src)
        hq_split = [r for r in rows if r["id"] in hq_ids]
        write_jsonl_atomic(hq_split, out)
        if split_name == "test":
            try:
                os.chmod(out, 0o444)   # mirror split_dataset.py: lock test_hq
            except OSError:
                pass
        ro_tag = "  [read-only]" if split_name == "test" else ""
        print(f"  {out}  ({n:,} rows){ro_tag}")

    print(f"\nDone. Phase-1 (512px): manifest_final.jsonl + splits/{{train,val,test}}.jsonl")
    print(f"      Phase-2 (1024px): manifest_hq_final.jsonl + splits/{{train,val,test}}_hq.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
