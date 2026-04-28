"""tools/analyze_eval.py — Slice eval_per_image.jsonl by subcategory / source.

After running tools/eval_checkpoint.py, you have:
    outputs/eval/<run>/eval_per_image.jsonl    # per-image IoU + dice + meta

This tool aggregates those rows into IoU/Dice slices by:
  - subcategory (style_cedar, occlusion, scene_context, ...)
  - review_source (manual, auto_accept_positive, auto_negative_clear, unreviewed)
  - class (pos / neg)

Use it to see WHERE the model is weakest. E.g. "the model is at 92% IoU
overall but only 71% on `occlusion_mild` subcategory" tells you exactly
which subset needs more training data or augmentation.

Usage:
    python -m tools.analyze_eval --eval-dir outputs/eval/phase2_test_hq

    # Compare two runs side-by-side
    python -m tools.analyze_eval --eval-dir outputs/eval/phase1_test \
                                  --compare outputs/eval/phase2_test_hq

    # Report worst N images per slice
    python -m tools.analyze_eval --eval-dir outputs/eval/phase2_test_hq --worst 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slice_metrics(rows: list[dict], by: str) -> dict:
    """Group rows by `by` field, compute IoU + Dice statistics per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get(by) or "(missing)"
        groups[str(key)].append(r)
    out = {}
    for key, items in sorted(groups.items()):
        ious = [float(r["iou"]) for r in items if r.get("iou") is not None]
        dices = [float(r["dice"]) for r in items if r.get("dice") is not None]
        if not ious:
            continue
        out[key] = {
            "n": len(ious),
            "iou_mean": mean(ious),
            "iou_median": median(ious),
            "iou_std": stdev(ious) if len(ious) > 1 else 0.0,
            "iou_p10": sorted(ious)[int(len(ious) * 0.10)],
            "dice_mean": mean(dices) if dices else 0.0,
        }
    return out


def _print_slice(label: str, slice_data: dict) -> None:
    print(f"\n{'='*78}")
    print(f"  Sliced by: {label}")
    print(f"{'='*78}")
    print(f"  {'group':<28s}  {'n':>6s}  {'iou_mean':>9s}  "
          f"{'iou_median':>10s}  {'iou_std':>8s}  {'iou_p10':>8s}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*8}")
    sorted_keys = sorted(slice_data.keys(),
                          key=lambda k: slice_data[k]["iou_mean"])
    for key in sorted_keys:
        s = slice_data[key]
        print(f"  {key[:28]:<28s}  {s['n']:>6,}  {s['iou_mean']:>9.4f}  "
              f"{s['iou_median']:>10.4f}  {s['iou_std']:>8.4f}  "
              f"{s['iou_p10']:>8.4f}")


def _print_worst(rows: list[dict], n: int) -> None:
    """Print the N worst-IoU images so you can inspect them."""
    print(f"\n{'='*78}")
    print(f"  {n} worst-IoU images")
    print(f"{'='*78}")
    sorted_rows = sorted(rows, key=lambda r: r.get("iou", 1.0))[:n]
    for r in sorted_rows:
        print(f"  iou={r.get('iou', 0):.3f}  dice={r.get('dice', 0):.3f}  "
              f"id={str(r.get('id'))[:36]:<36s}  "
              f"class={str(r.get('class')):<5s}  "
              f"sub={str(r.get('subcategory'))[:18]:<18s}  "
              f"src={r.get('review_source')}")


def _compare(a_rows: list[dict], b_rows: list[dict],
              a_label: str, b_label: str) -> None:
    """Compare two runs on the SAME images (matched by id)."""
    a_by_id = {r["id"]: r for r in a_rows if "id" in r}
    b_by_id = {r["id"]: r for r in b_rows if "id" in r}
    common = sorted(set(a_by_id) & set(b_by_id))
    if not common:
        print("\n  No common image IDs between the two runs — skipping compare.")
        return
    print(f"\n{'='*78}")
    print(f"  Head-to-head: {a_label}  vs  {b_label}  ({len(common)} common images)")
    print(f"{'='*78}")
    a_better = b_better = ties = 0
    a_iou_total = b_iou_total = 0.0
    biggest_wins: list[tuple[float, dict, dict]] = []
    biggest_regressions: list[tuple[float, dict, dict]] = []
    for iid in common:
        ai = a_by_id[iid]["iou"]
        bi = b_by_id[iid]["iou"]
        a_iou_total += ai
        b_iou_total += bi
        delta = bi - ai
        if delta > 0.005:
            b_better += 1
            biggest_wins.append((delta, a_by_id[iid], b_by_id[iid]))
        elif delta < -0.005:
            a_better += 1
            biggest_regressions.append((delta, a_by_id[iid], b_by_id[iid]))
        else:
            ties += 1
    print(f"  {a_label} mean IoU: {a_iou_total/len(common):.4f}")
    print(f"  {b_label} mean IoU: {b_iou_total/len(common):.4f}")
    print(f"  delta:               {(b_iou_total - a_iou_total)/len(common):+.4f}")
    print(f"  B better on: {b_better}  |  A better on: {a_better}  |  tied: {ties}")
    biggest_wins.sort(key=lambda t: -t[0])
    biggest_regressions.sort(key=lambda t: t[0])
    print(f"\n  Top 5 wins for {b_label}:")
    for d, ar, br in biggest_wins[:5]:
        print(f"    +{d:.3f}  id={str(ar['id'])[:36]:<36s} sub={ar.get('subcategory')}")
    print(f"\n  Top 5 regressions for {b_label}:")
    for d, ar, br in biggest_regressions[:5]:
        print(f"    {d:+.3f}  id={str(ar['id'])[:36]:<36s} sub={ar.get('subcategory')}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Slice eval_per_image.jsonl by subcategory + show worst cases."
    )
    ap.add_argument("--eval-dir", type=Path, required=True,
                     help="Directory containing eval_per_image.jsonl + eval_summary.json")
    ap.add_argument("--compare", type=Path, default=None,
                     help="Compare against a second eval-dir (same images preferred).")
    ap.add_argument("--worst", type=int, default=20,
                     help="Show top N worst-IoU images (default 20). Set 0 to skip.")
    ap.add_argument("--out", type=Path, default=None,
                     help="Optional: write the full sliced report to a JSON file.")
    args = ap.parse_args()

    pi_path = args.eval_dir / "eval_per_image.jsonl"
    if not pi_path.exists():
        print(f"ERROR: {pi_path} not found. Run tools.eval_checkpoint first.",
              file=sys.stderr)
        return 1
    summary_path = args.eval_dir / "eval_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"\n  Summary ({args.eval_dir}):")
        for k in ("val_iou", "val_dice", "val_boundary_iou", "val_per_image_iou",
                  "n_images", "tta_scales", "post_process"):
            if k in summary:
                print(f"    {k:<24s}  {summary[k]}")

    rows = _load_jsonl(pi_path)
    print(f"\n  Loaded {len(rows)} per-image rows from {pi_path}")

    slice_sub = _slice_metrics(rows, by="subcategory")
    slice_class = _slice_metrics(rows, by="class")
    slice_review = _slice_metrics(rows, by="review_source")

    _print_slice("class", slice_class)
    _print_slice("review_source", slice_review)
    _print_slice("subcategory", slice_sub)

    if args.worst > 0:
        _print_worst(rows, args.worst)

    if args.compare is not None:
        b_path = args.compare / "eval_per_image.jsonl"
        if b_path.exists():
            b_rows = _load_jsonl(b_path)
            _compare(rows, b_rows, str(args.eval_dir.name), str(args.compare.name))
        else:
            print(f"\n  WARNING: --compare path missing eval_per_image.jsonl: {b_path}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": summary if summary_path.exists() else None,
            "by_class": slice_class,
            "by_review_source": slice_review,
            "by_subcategory": slice_sub,
        }
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\n  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
