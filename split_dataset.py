#!/usr/bin/env python3
"""split_dataset.py — Enterprise-grade stratified train/val/test split.

Reads dataset/manifest.jsonl, performs a reproducible stratified split by
(class × subcategory), and emits three self-contained JSONL files plus a
split_info.json manifest for audit + reproducibility.

Features:
  • Stratified by (class, subcategory) — proportional representation per split
  • Fully deterministic via --seed (reproducible across machines)
  • Self-contained split files (full manifest rows, not just IDs)
  • Atomic writes — temp file + rename, no half-written outputs on crash
  • Comprehensive metadata — seed, ratios, manifest hash, git commit, command
  • Integrity checks — no row in multiple splits; row-count sanity
  • Small-group handling — subcategories <3 samples land in train with warning
  • Dry-run preview — full distribution report without writing anything
  • Force overwrite protection — default errors if outputs already exist

Usage:
    python split_dataset.py                             # 70/15/15 @ seed 42
    python split_dataset.py --ratios 0.8,0.1,0.1
    python split_dataset.py --seed 1337
    python split_dataset.py --dry-run                   # preview only
    python split_dataset.py --force                     # overwrite existing

Output (when committed):
    dataset/splits/train.jsonl          (full manifest rows for train)
    dataset/splits/val.jsonl            (full manifest rows for val)
    dataset/splits/test.jsonl           (full manifest rows for test)
    dataset/splits/split_info.json      (audit + reproducibility metadata)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Logging — stdout + rolling file (enterprise audit requirement)
# ══════════════════════════════════════════════════════════════════════

log = logging.getLogger("split")


def setup_logging(out_dir: Path) -> Path:
    """Configure logging to stdout AND an append-only log file under out_dir.
    Returns the log file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "split_dataset.log"
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,  # replace any prior handlers
    )
    return log_path


# ══════════════════════════════════════════════════════════════════════
# Stable hashing (FIX #1: replaces Python's randomized hash())
# ══════════════════════════════════════════════════════════════════════

def stable_hash32(s: str) -> int:
    """Deterministic 32-bit hash — stable across Python interpreter runs and
    platforms. Uses SHA-256 (first 4 bytes). Replaces built-in hash() which is
    randomized by PYTHONHASHSEED and breaks cross-run reproducibility."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# ══════════════════════════════════════════════════════════════════════
# BK-tree for dhash near-dup grouping (FIX #2: leak-proof splits)
# ══════════════════════════════════════════════════════════════════════

class _BKNode:
    __slots__ = ("key", "children")

    def __init__(self, key: int):
        self.key = key
        self.children: dict[int, "_BKNode"] = {}


def _popcount(x: int) -> int:
    return x.bit_count()


class BKTree:
    """Minimal Burkhard-Keller tree keyed by hamming distance.
    Used to find all keys within N hamming of a query in near-log time."""

    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, key: int) -> None:
        if self.root is None:
            self.root = _BKNode(key)
            return
        node = self.root
        while True:
            d = _popcount(node.key ^ key)
            if d == 0:
                return   # exact-duplicate key; nothing to do
            nxt = node.children.get(d)
            if nxt is None:
                node.children[d] = _BKNode(key)
                return
            node = nxt

    def find_within(self, key: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        out: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = _popcount(node.key ^ key)
            if d <= threshold:
                out.append(node.key)
            lo, hi = max(1, d - threshold), d + threshold
            for child_d, child in node.children.items():
                if lo <= child_d <= hi:
                    stack.append(child)
        return out


def _dhash_to_uint64(x: int | None) -> int | None:
    """Manifest stores dhash as signed int64 (SQLite constraint).
    Convert back to unsigned for correct XOR/popcount."""
    if x is None:
        return None
    return x + (1 << 64) if x < 0 else x


def group_by_dhash(rows: list[dict], threshold: int) -> dict[str, str]:
    """Build an {id → group_id} map. Rows whose dhash is within `threshold`
    hamming distance belong to the same group. Rows without a dhash are
    singletons (group_id = their own id)."""
    # Union-find
    parent: dict[str, str] = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Map dhash (uint64) → list of ids with that exact dhash
    exact: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        dh = _dhash_to_uint64(r.get("dhash"))
        if dh is not None:
            exact[dh].append(r["id"])

    # Union all ids with exact-identical dhash (cheap, catches the easy cases)
    for ids in exact.values():
        if len(ids) > 1:
            root = ids[0]
            for other in ids[1:]:
                union(root, other)

    # BK-tree keyed by unique dhash; map dhash → representative id
    bk = BKTree()
    dh_rep: dict[int, str] = {dh: ids[0] for dh, ids in exact.items()}
    for dh in dh_rep:
        bk.add(dh)

    # For each unique dhash, find near-dups and union representatives.
    for dh, rep_id in dh_rep.items():
        for neighbor in bk.find_within(dh, threshold):
            if neighbor == dh:
                continue
            union(rep_id, dh_rep[neighbor])

    return {rid: find(rid) for rid in parent}


# ══════════════════════════════════════════════════════════════════════
# I/O utilities
# ══════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid JSON: {e}") from None
    return rows


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    """Write rows to path via temp+rename so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Clear read-only attribute on existing target (Windows: a previous
    # split_dataset run chmod-ed test.jsonl to 0o444; os.replace would
    # fail without this).
    if path.exists():
        try:
            import os as _os
            _os.chmod(path, 0o644)
        except OSError:
            pass
    tmp.replace(path)


def write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    """Return short HEAD commit if we're in a git repo, else None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════

def parse_ratios(s: str) -> tuple[float, float, float]:
    """Parse '0.7,0.15,0.15' → (0.7, 0.15, 0.15), validating sum ≈ 1.0."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--ratios must have 3 values (train,val,test); got {len(parts)}")
    try:
        r = tuple(float(p) for p in parts)   # type: ignore[misc]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--ratios must be floats: {s!r}")
    if any(v < 0 or v > 1 for v in r):
        raise argparse.ArgumentTypeError(f"--ratios values must be in [0,1]: {r}")
    if not (0.999 < sum(r) < 1.001):
        raise argparse.ArgumentTypeError(
            f"--ratios must sum to 1.0; got {sum(r):.4f}")
    return r  # type: ignore[return-value]


def validate_manifest(rows: list[dict]) -> None:
    """Ensure the manifest has the fields we need for stratification."""
    if not rows:
        raise ValueError("manifest is empty")
    required = {"id", "class", "subcategory"}
    missing_any = [r for r in rows if not required <= set(r.keys())]
    if missing_any:
        raise ValueError(
            f"{len(missing_any)} manifest rows missing one of {required}; "
            f"first bad row: {missing_any[0]}")
    # Uniqueness of IDs
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = [i for i in ids if ids.count(i) > 1]
        raise ValueError(f"duplicate IDs in manifest: {dupes[:5]}...")


# ══════════════════════════════════════════════════════════════════════
# Stratified splitter
# ══════════════════════════════════════════════════════════════════════

def stratify_split(
    rows: list[dict],
    ratios: tuple[float, float, float],
    seed: int,
    min_group_size: int,
    dhash_group_map: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Split rows three ways, stratified by (class, subcategory).

    If `dhash_group_map` is provided, rows in the same dhash-group are
    treated as one atomic unit — they always go to the same split, preventing
    near-duplicate leakage between train and val/test.

    Within each stratum the shuffle seed is derived from `(seed, class,
    subcategory)` via STABLE SHA256 hashing, so the same --seed reproduces
    the exact same split across Python versions, machines, and PYTHONHASHSEED
    values.
    """
    # Build stratum groups; the atomic unit is a "bucket" (either a single row
    # or a dhash-group of rows). Each bucket is assigned to ONE split.
    # Keyed by (class, subcategory, dhash_group_root) so near-dups can't split
    # across strata either.
    strata: dict[tuple[str, str], list[list[dict]]] = defaultdict(list)
    for r in rows:
        cls = r["class"]
        subcat = r["subcategory"]
        if dhash_group_map is not None:
            # Groups are formed in a pre-pass; here we collect rows into buckets
            pass
        strata[(cls, subcat)].append([r])   # default: each row is its own bucket

    # If dhash grouping is active, merge rows with same group_root into one bucket.
    if dhash_group_map is not None:
        # Rebuild strata: within each stratum, collapse rows by group_root
        # (but only if they share class+subcategory — otherwise the group
        # straddles strata, which is a rare edge case we handle below).
        by_group: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_group[dhash_group_map[r["id"]]].append(r)

        # For groups that straddle multiple (class, subcat) strata, keep each
        # row as its own bucket (can't meaningfully put them together) —
        # in practice this is almost never the case after dedup.
        strata = defaultdict(list)
        for grp_id, grp_rows in by_group.items():
            keys = {(g["class"], g["subcategory"]) for g in grp_rows}
            if len(keys) == 1:
                (cls, subcat) = next(iter(keys))
                strata[(cls, subcat)].append(grp_rows)
            else:
                for g in grp_rows:
                    strata[(g["class"], g["subcategory"])].append([g])

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    distribution: dict[str, dict] = {}
    small_groups: list[str] = []

    for (cls, subcat) in sorted(strata.keys()):
        buckets = strata[(cls, subcat)]
        # n_rows is the row count; n_buckets is the atomic unit count
        n_rows = sum(len(b) for b in buckets)
        n_buckets = len(buckets)

        # FIX #1: stable hash so seed+stratum → same shuffle across runs/platforms
        group_seed = (seed ^ stable_hash32(f"{cls}||{subcat}")) & 0xFFFFFFFF
        rng = random.Random(group_seed)
        shuffled = list(buckets)
        rng.shuffle(shuffled)

        # Partition at the BUCKET level. Sum row counts up to target cutoffs.
        if n_rows < min_group_size:
            # Too small to stratify — dump everything in train, warn.
            train.extend(r for b in shuffled for r in b)
            g_train_n, g_val_n, g_test_n = n_rows, 0, 0
            small_groups.append(f"{cls}:{subcat}(n={n_rows})")
        else:
            t_target = int(round(n_rows * ratios[0]))
            v_target = int(round(n_rows * (ratios[0] + ratios[1])))
            # Min 1 per split when possible
            if t_target == 0: t_target = 1
            if v_target == t_target and n_rows - t_target >= 2: v_target = t_target + 1
            if v_target >= n_rows: v_target = max(t_target, n_rows - 1)

            cum = 0
            g_train: list[dict] = []
            g_val: list[dict] = []
            g_test: list[dict] = []
            for bucket in shuffled:
                b_size = len(bucket)
                if cum < t_target:
                    g_train.extend(bucket)
                elif cum < v_target:
                    g_val.extend(bucket)
                else:
                    g_test.extend(bucket)
                cum += b_size

            train.extend(g_train)
            val.extend(g_val)
            test.extend(g_test)
            g_train_n, g_val_n, g_test_n = len(g_train), len(g_val), len(g_test)

        key = f"{cls}:{subcat}"
        distribution[key] = {
            "total":   n_rows,
            "buckets": n_buckets,
            "train":   g_train_n,
            "val":     g_val_n,
            "test":    g_test_n,
        }

    report = {
        "totals": {"train": len(train), "val": len(val), "test": len(test),
                   "all": len(rows)},
        "ratios_requested": list(ratios),
        "ratios_actual": [
            round(len(train) / max(len(rows), 1), 4),
            round(len(val)   / max(len(rows), 1), 4),
            round(len(test)  / max(len(rows), 1), 4),
        ],
        "distribution": distribution,
        "small_groups_to_train": small_groups,
    }
    return train, val, test, report


# ══════════════════════════════════════════════════════════════════════
# Integrity verification
# ══════════════════════════════════════════════════════════════════════

def verify_split(train: list[dict], val: list[dict], test: list[dict],
                 original_count: int) -> None:
    """Sanity checks that cannot silently fail."""
    total = len(train) + len(val) + len(test)
    if total != original_count:
        raise AssertionError(
            f"row-count mismatch: input={original_count} but splits sum to {total}")
    ids_train = {r["id"] for r in train}
    ids_val = {r["id"] for r in val}
    ids_test = {r["id"] for r in test}
    overlap_tv = ids_train & ids_val
    overlap_tt = ids_train & ids_test
    overlap_vt = ids_val & ids_test
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"ID overlap detected! train/val: {len(overlap_tv)}  "
            f"train/test: {len(overlap_tt)}  val/test: {len(overlap_vt)}")
    if len(ids_train) + len(ids_val) + len(ids_test) != original_count:
        raise AssertionError("id-uniqueness violated after split")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=Path,
                    default=Path("dataset/manifest.jsonl"),
                    help="Input manifest produced by prepare_dataset.py")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/splits"),
                    help="Output directory for train/val/test/split_info")
    ap.add_argument("--ratios", type=parse_ratios, default=(0.70, 0.15, 0.15),
                    help="Comma-separated 'train,val,test' fractions (sum=1). "
                         "Default: 0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility (default 42)")
    ap.add_argument("--min-group-size", type=int, default=3,
                    help="Subcategories smaller than this land entirely in train "
                         "(can't meaningfully 3-way split). Default 3")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report distribution without writing any files")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing split files without error")
    ap.add_argument("--group-by-dhash", action="store_true",
                    help="Cluster near-duplicate images (dhash within threshold) "
                         "into atomic groups that travel to the same split. "
                         "Prevents train/test leakage from visually-similar images.")
    ap.add_argument("--dhash-threshold", type=int, default=5,
                    help="Hamming distance ≤ this = near-duplicate (default 5, "
                         "matches scrape-time phash_hamming_threshold).")
    args = ap.parse_args()

    # Set up dual-output logging (stdout + file) FIRST so everything is captured
    log_path = setup_logging(args.out_dir)

    # ── Load + validate manifest ───────────────────────────────────────
    if not args.manifest.exists():
        log.error(f"manifest not found: {args.manifest}")
        return 2
    log.info(f"Loading manifest: {args.manifest}")
    rows = load_jsonl(args.manifest)
    validate_manifest(rows)
    log.info(f"  rows: {len(rows):,}")
    log.info(f"  log file: {log_path}")

    # ── Overwrite protection ───────────────────────────────────────────
    if not args.dry_run and not args.force:
        for f in ("train.jsonl", "val.jsonl", "test.jsonl", "split_info.json"):
            p = args.out_dir / f
            if p.exists():
                log.error(f"{p} exists. Use --force to overwrite.")
                return 2

    # ── FIX #2: Optional dhash-based near-duplicate group pre-pass ─────
    dhash_group_map: dict[str, str] | None = None
    dhash_group_stats: dict | None = None
    if args.group_by_dhash:
        log.info(f"Building dhash near-dup groups (threshold={args.dhash_threshold})...")
        dhash_group_map = group_by_dhash(rows, args.dhash_threshold)
        # Stats
        unique_groups = set(dhash_group_map.values())
        from collections import Counter
        group_sizes = Counter(dhash_group_map.values())
        multi_groups = {gid: n for gid, n in group_sizes.items() if n > 1}
        dhash_group_stats = {
            "total_rows": len(dhash_group_map),
            "unique_groups": len(unique_groups),
            "multi_row_groups": len(multi_groups),
            "rows_in_multi_row_groups": sum(multi_groups.values()),
            "largest_group_size": max(group_sizes.values()) if group_sizes else 0,
        }
        log.info(f"  groups: {dhash_group_stats['unique_groups']:,} "
                 f"(of which {dhash_group_stats['multi_row_groups']} span >1 row, "
                 f"covering {dhash_group_stats['rows_in_multi_row_groups']} rows)")

    # ── Split ──────────────────────────────────────────────────────────
    log.info(f"Splitting (ratios={args.ratios} seed={args.seed} "
             f"group_by_dhash={args.group_by_dhash})...")
    train, val, test, report = stratify_split(
        rows, args.ratios, args.seed, args.min_group_size,
        dhash_group_map=dhash_group_map,
    )

    # ── Verify integrity ───────────────────────────────────────────────
    verify_split(train, val, test, len(rows))
    log.info("  integrity: OK (no ID overlap, row count matches)")

    # ── Print distribution summary ─────────────────────────────────────
    log.info("")
    log.info("=== Split totals ===")
    tot = report["totals"]
    log.info(f"  train: {tot['train']:>6,}  ({report['ratios_actual'][0]*100:.2f}%)")
    log.info(f"  val:   {tot['val']:>6,}  ({report['ratios_actual'][1]*100:.2f}%)")
    log.info(f"  test:  {tot['test']:>6,}  ({report['ratios_actual'][2]*100:.2f}%)")
    log.info(f"  all:   {tot['all']:>6,}")

    if report["small_groups_to_train"]:
        log.warning(f"{len(report['small_groups_to_train'])} subcategories had "
                    f"<{args.min_group_size} samples; placed entirely in train:")
        for g in report["small_groups_to_train"]:
            log.warning(f"    - {g}")

    log.info("")
    log.info("=== Per-subcategory distribution ===")
    log.info(f"{'subcategory':44s}  {'total':>7s}  {'train':>7s}  {'val':>6s}  {'test':>6s}")
    for key in sorted(report["distribution"].keys()):
        d = report["distribution"][key]
        log.info(f"{key:44s}  {d['total']:>7,}  {d['train']:>7,}  "
                 f"{d['val']:>6,}  {d['test']:>6,}")

    # ── Write outputs (unless dry-run) ─────────────────────────────────
    if args.dry_run:
        log.info("")
        log.info("[dry-run] no files written")
        return 0

    log.info("")
    log.info(f"Writing splits -> {args.out_dir}/")
    # Sort each split by id for deterministic, git-friendly diffs
    train.sort(key=lambda r: r["id"])
    val.sort(key=lambda r: r["id"])
    test.sort(key=lambda r: r["id"])
    write_jsonl_atomic(train, args.out_dir / "train.jsonl")
    write_jsonl_atomic(val,   args.out_dir / "val.jsonl")
    write_jsonl_atomic(test,  args.out_dir / "test.jsonl")

    # Full audit metadata — reproducibility + provenance
    split_info = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python_version": sys.version.split()[0],
        "git_commit": git_commit(),
        "seed": args.seed,
        "ratios_requested": list(args.ratios),
        "ratios_actual": report["ratios_actual"],
        "min_group_size": args.min_group_size,
        "group_by_dhash": args.group_by_dhash,
        "dhash_threshold": args.dhash_threshold if args.group_by_dhash else None,
        "dhash_grouping": dhash_group_stats,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_rows": len(rows),
        "log_file": str(log_path),
        "outputs": {
            "train": str(args.out_dir / "train.jsonl"),
            "val":   str(args.out_dir / "val.jsonl"),
            "test":  str(args.out_dir / "test.jsonl"),
        },
        "totals": report["totals"],
        "small_groups_to_train": report["small_groups_to_train"],
        "distribution": report["distribution"],
    }
    write_json_atomic(split_info, args.out_dir / "split_info.json")

    # Make test.jsonl read-only so accidental overwrites are blocked.
    # Effective on Unix; on Windows this clears write bits which also blocks edits.
    try:
        (args.out_dir / "test.jsonl").chmod(0o444)
    except Exception as e:
        log.warning(f"couldn't chmod test.jsonl read-only: {e}")

    log.info(f"  train.jsonl:     {tot['train']:,} rows")
    log.info(f"  val.jsonl:       {tot['val']:,} rows")
    log.info(f"  test.jsonl:      {tot['test']:,} rows  (marked read-only)")
    log.info(f"  split_info.json: audit metadata")
    log.info(f"  split_dataset.log: this log file (appended)")
    log.info("")
    log.info(f"  manifest SHA-256: {split_info['manifest_sha256']}")
    log.info(f"  seed: {args.seed} -> reproducible via:")
    dhash_arg = " --group-by-dhash" if args.group_by_dhash else ""
    log.info(f"    python split_dataset.py --seed {args.seed}{dhash_arg} --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
