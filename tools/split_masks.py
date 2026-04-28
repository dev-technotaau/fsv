#!/usr/bin/env python3
"""Organize mask + preview files into per-split directories using hard links.

Reads each split JSONL, hard-links the corresponding files from
`dataset/annotations_v1/{masks,masks_preview,viz}/<id>.png` into
`dataset/splits/<split>/{masks,masks_preview,viz}/<id>.png`.

Hard links use ZERO additional disk space — they're just additional directory
entries pointing to the same file content. If hard linking isn't supported
(rare; cross-volume only), falls back to copy.

Default splits processed: train, val, test, train_hq, val_hq, test_hq.

Usage:
    python tools/split_masks.py                           # masks + preview
    python tools/split_masks.py --include-viz             # also viz (large)
    python tools/split_masks.py --splits train val test   # only full splits
    python tools/split_masks.py --copy                    # force copy (no hardlink)
    python tools/split_masks.py --dry-run                 # preview only
    python tools/split_masks.py --clean                   # nuke per-split dirs first
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> str:
    """Hard-link src -> dst; fall back to copy if linking fails.
    Returns 'link', 'copy', or 'skip' (already exists)."""
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations-root", type=Path,
                    default=Path("dataset/annotations_v1"))
    ap.add_argument("--splits-dir", type=Path,
                    default=Path("dataset/splits"))
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test",
                             "train_hq", "val_hq", "test_hq"],
                    help="Split file basenames (without .jsonl) to process.")
    ap.add_argument("--include-viz", action="store_true",
                    help="Also link viz/<id>.png files (large; usually not "
                         "needed for training).")
    ap.add_argument("--copy", action="store_true",
                    help="Force copy instead of hard link (uses extra disk).")
    ap.add_argument("--clean", action="store_true",
                    help="Delete per-split mask/preview/viz dirs before linking. "
                         "Use this if a previous split assignment changed and "
                         "you want a fresh tree (won't touch the source files).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_masks = args.annotations_root / "masks"
    src_preview = args.annotations_root / "masks_preview"
    src_viz = args.annotations_root / "viz"

    if not src_masks.exists():
        print(f"ERROR: source masks dir not found: {src_masks}", file=sys.stderr)
        return 2

    grand_total = {"link": 0, "copy": 0, "skip": 0, "missing_mask": 0,
                   "missing_preview": 0, "missing_viz": 0}

    for split_name in args.splits:
        split_file = args.splits_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            print(f"\n[skip] {split_file} not found")
            continue
        rows = load_jsonl(split_file)
        ids = [r["id"] for r in rows]
        out_root = args.splits_dir / split_name
        out_masks = out_root / "masks"
        out_preview = out_root / "masks_preview"
        out_viz = out_root / "viz" if args.include_viz else None

        print(f"\n=== {split_name}: {len(ids):,} images "
              f"-> {out_root}/ ===")

        if args.clean and not args.dry_run:
            for d in (out_masks, out_preview, out_viz):
                if d is not None and d.exists():
                    print(f"  [clean] removing {d}")
                    shutil.rmtree(d)

        if args.dry_run:
            mode = "copy" if args.copy else "hard-link"
            tag = " + viz" if args.include_viz else ""
            print(f"  [dry-run] would {mode} masks + preview{tag} "
                  f"for {len(ids):,} images")
            continue

        ops = {"link": 0, "copy": 0, "skip": 0, "missing_mask": 0,
               "missing_preview": 0, "missing_viz": 0}
        t0 = time.time()
        last_report = t0

        for i, iid in enumerate(ids, 1):
            # mask
            sm = src_masks / f"{iid}.png"
            if sm.exists():
                ops[link_or_copy(sm, out_masks / f"{iid}.png", args.copy)] += 1
            else:
                ops["missing_mask"] += 1
            # preview
            sp = src_preview / f"{iid}.png"
            if sp.exists():
                ops[link_or_copy(sp, out_preview / f"{iid}.png", args.copy)] += 1
            else:
                ops["missing_preview"] += 1
            # viz (optional)
            if args.include_viz:
                sv = src_viz / f"{iid}.png"
                if sv.exists():
                    ops[link_or_copy(sv, out_viz / f"{iid}.png", args.copy)] += 1
                else:
                    ops["missing_viz"] += 1

            now = time.time()
            if now - last_report >= 5.0 or i == len(ids):
                rate = i / max(now - t0, 1e-6)
                print(f"  [{i:>6,}/{len(ids):,}]  {rate:6.0f} img/s  "
                      f"link={ops['link']:,} copy={ops['copy']:,} "
                      f"skip={ops['skip']:,} miss_mask={ops['missing_mask']}")
                last_report = now

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s   "
              f"link={ops['link']:,} copy={ops['copy']:,} skip={ops['skip']:,} "
              f"miss_mask={ops['missing_mask']} miss_preview={ops['missing_preview']}"
              f"{' miss_viz=' + str(ops['missing_viz']) if args.include_viz else ''}")
        for k in grand_total:
            grand_total[k] += ops[k]

    print()
    print("=" * 60)
    print("Summary across all splits")
    print("=" * 60)
    for k, v in grand_total.items():
        print(f"  {k:<18s} {v:>8,}")
    print()
    print("Per-split layout:")
    for s in args.splits:
        if (args.splits_dir / f"{s}.jsonl").exists():
            print(f"  {args.splits_dir / s}/")
            print(f"      masks/<id>.png")
            print(f"      masks_preview/<id>.png")
            if args.include_viz:
                print(f"      viz/<id>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
