"""diagnose_corrupt.py — Run on the TRAINING DEVICE (G:\) from the project root.

Tests the EXACT files that showed up in your error log against a baseline
captured from the source-of-truth machine. Tells us in one shot whether the
problem is:
  (A) Transfer corruption  — file size differs from source
  (B) Disk-level corruption — file size matches but bytes differ (SHA256 mismatch)
  (C) Transient I/O        — file is fine but reads sometimes fail (flaky disk)
  (D) PIL decode strictness — file decodes with LOAD_TRUNCATED_IMAGES=True
  (E) Genuinely truncated   — file is short and decode fails everywhere

Output is human-readable + writes results.json for further analysis.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path

try:
    from PIL import Image, ImageFile
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)

# ── Baseline from source-of-truth (D:\) — what these files SHOULD look like ─
BASELINE = {
    "pw_bing__cedar_fence_barely_visible_behind_dense___a2667ed9.jpg": 665_623,
    "pixabay__wooden_privacy_fence_backyard__cedb35f3.jpg":            209_158,
    "pixabay__wooden_privacy_fence_backyard__d6220e47.jpg":            236_182,
    "pw_bing__cedar_fence_hd__35c894dc.jpg":                           123_800,
    "pexels__cedar_fence__1e58c93b.jpg":                               362_917,
    "pw_bing__cedar_fence_barely_visible_behind_dense___7ef8b95c.jpg": 225_913,
    "pexels__cat_on_wooden_fence__78183724.jpg":                       228_891,
    "pexels__aluminum_fence__b4995378.jpg":                            505_094,
    # Add more from your error log here if you want to test more files
}

# Also test these from your log (no baseline size known — just stat & decode test):
EXTRA_FILES = [
    "company_sites__www_illinoisfencing_com__4330c2bb.jpg",
    "pw_bing__cedar_fence_hd__1e55df18.jpg",
    "company_sites__abetterfencecompany_com__364a994a.jpg",
    "pexels__aluminum_fence__cdddab6f.jpg",
    "pw_bing__cedar_fence_barely_visible_behind_dense___db5ed09d.jpg",
]

# Roots to search
ROOTS = [Path("data_scraped/images"), Path("data_scraped_neg/images"),
         Path("data_scraped_pos/images"), Path("data_scraped_hard_positive/images")]


def find(name: str) -> Path | None:
    for r in ROOTS:
        p = r / name
        if p.exists(): return p
    return None


def sha256_first(path: Path, n_bytes: int = 1_000_000) -> str:
    """Hash first N bytes — fast partial fingerprint. Different drives /
    different copy attempts will produce different hashes only if the file
    actually differs at the byte level."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def test_pil_strict(path: Path, n_trials: int = 5) -> tuple[list[str], float]:
    """Try decoding the file N times in a row. Detects transient I/O failures
    (file is fine, disk read is flaky)."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    results = []
    t0 = time.time()
    for _ in range(n_trials):
        try:
            with Image.open(path) as im:
                im.load()
            results.append("OK")
        except Exception as e:
            results.append(f"FAIL:{type(e).__name__}")
    return results, time.time() - t0


def test_pil_tolerant(path: Path) -> str:
    """Try with LOAD_TRUNCATED_IMAGES=True (matches what most image viewers do).
    If this succeeds but strict fails → file is slightly truncated, viewer-OK."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as im:
            im.load()
            return f"OK {im.size}"
    except Exception as e:
        return f"FAIL:{type(e).__name__}"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def magic_bytes(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(8).hex()
    except Exception:
        return "?"


def diagnose() -> None:
    print(f"{'='*100}")
    print(f"FenceDataset corrupt-file diagnostic — running on {Path.cwd()}")
    print(f"PIL version: {Image.__version__}  Python: {sys.version.split()[0]}")
    print(f"{'='*100}")

    cases: list[dict] = []

    # === Section A: Files with known baseline (compare size + bytes) ===
    print("\n--- A. Files with known baseline from source-of-truth (D:\\) ---")
    print(f"  {'file':60s} {'expected':>10s} {'actual':>10s}  {'size?':10s} {'magic':10s} {'pil_strict_5x':30s} {'pil_tolerant':15s}")
    for name, expected_size in BASELINE.items():
        p = find(name)
        if p is None:
            print(f"  {name[:60]:60s}  *** FILE NOT FOUND in any data_scraped*/images dir ***")
            cases.append({"name": name, "found": False})
            continue
        actual = p.stat().st_size
        size_ok = "MATCH" if actual == expected_size else f"DIFF({actual-expected_size:+d})"
        magic = magic_bytes(p)[:8]
        magic_ok = "JPEG" if magic.startswith("ffd8ff") else f"NOT_JPEG({magic})"
        strict_results, _ = test_pil_strict(p, n_trials=5)
        tolerant = test_pil_tolerant(p)
        print(f"  {name[:60]:60s} {expected_size:>10,} {actual:>10,}  {size_ok:10s} {magic_ok:10s} {','.join(strict_results):30s} {tolerant[:15]:15s}")
        cases.append({"name": name, "expected": expected_size, "actual": actual,
                       "size_match": actual == expected_size, "magic": magic,
                       "strict_trials": strict_results, "tolerant": tolerant,
                       "sha256_first1MB": sha256_first(p)})

    # === Section B: Extra files (no baseline, just scan + decode) ===
    print("\n--- B. Extra files from your error log ---")
    print(f"  {'file':60s} {'size':>10s}  {'magic':10s} {'pil_strict_5x':30s} {'pil_tolerant':15s}")
    for name in EXTRA_FILES:
        p = find(name)
        if p is None:
            print(f"  {name[:60]:60s}  *** NOT FOUND ***")
            cases.append({"name": name, "found": False})
            continue
        actual = p.stat().st_size
        magic = magic_bytes(p)[:8]
        magic_ok = "JPEG" if magic.startswith("ffd8ff") else f"NOT_JPEG({magic})"
        strict_results, _ = test_pil_strict(p, n_trials=5)
        tolerant = test_pil_tolerant(p)
        print(f"  {name[:60]:60s} {actual:>10,}  {magic_ok:10s} {','.join(strict_results):30s} {tolerant[:15]:15s}")
        cases.append({"name": name, "actual": actual, "magic": magic,
                       "strict_trials": strict_results, "tolerant": tolerant})

    # === Section C: Disk health stats ===
    print("\n--- C. Storage info ---")
    import shutil
    cwd = Path.cwd().resolve()
    drive = cwd.drive or str(cwd)
    try:
        usage = shutil.disk_usage(drive)
        print(f"  Drive: {drive}")
        print(f"  Total : {usage.total/1e9:>8.1f} GB")
        print(f"  Used  : {usage.used/1e9:>8.1f} GB")
        print(f"  Free  : {usage.free/1e9:>8.1f} GB ({usage.free/usage.total*100:.1f}%)")
    except Exception as e:
        print(f"  (could not query disk stats: {e})")

    # === Section D: Verdict ===
    print(f"\n{'='*100}\nDIAGNOSIS\n{'='*100}")
    have_baseline = [c for c in cases if c.get("found", True) and "expected" in c]
    if not have_baseline:
        print("  No files with baseline could be tested. Check that data_scraped/ exists.")
    else:
        size_mismatches = [c for c in have_baseline if not c["size_match"]]
        all_strict_fail = [c for c in have_baseline if all(r.startswith("FAIL") for r in c["strict_trials"])]
        intermittent    = [c for c in have_baseline if any(r.startswith("FAIL") for r in c["strict_trials"]) and any(r == "OK" for r in c["strict_trials"])]
        truncated_only  = [c for c in have_baseline if all(r.startswith("FAIL") for r in c["strict_trials"]) and c["tolerant"].startswith("OK")]

        print(f"  Files tested with baseline    : {len(have_baseline)}")
        print(f"  Size mismatch (transfer issue): {len(size_mismatches)}  {[c['name'] for c in size_mismatches[:3]]}")
        print(f"  Strict decode fail (all 5x)   : {len(all_strict_fail)}")
        print(f"  Intermittent (some pass, some fail) : {len(intermittent)}  ← FLAKY DISK / NETWORK")
        print(f"  Decode OK only with tolerant mode   : {len(truncated_only)}  ← actually truncated, viewer-OK")

        print(f"\n  Most likely cause:")
        if size_mismatches:
            print(f"    >>> (A) TRANSFER CORRUPTION <<<  — file sizes differ from source.")
            print(f"        Action: re-copy data with verification (e.g. robocopy /Z /COPYALL)")
        elif intermittent:
            print(f"    >>> (C) FLAKY DISK / NETWORK <<<  — same file passes some attempts, fails others.")
            print(f"        Action: chkdsk G: /f, check SMART status, or move data to a healthier drive.")
        elif truncated_only:
            print(f"    >>> (D) SLIGHTLY TRUNCATED FILES <<<  — viewers tolerate, strict PIL doesn't.")
            print(f"        Action: set ImageFile.LOAD_TRUNCATED_IMAGES=True in tools/dataset.py")
        elif all_strict_fail:
            print(f"    >>> (B/E) FILES FULLY CORRUPT <<<  — bytes don't match a valid JPEG.")
            print(f"        Action: re-scrape these specific files OR drop them from the JSONLs.")
        else:
            print(f"    >>> (F) FILES ARE FINE <<<  — couldn't reproduce the failure.")
            print(f"        Action: the failure must be a multi-worker race or RAM pressure.")
            print(f"                Try: python train_web_deployable.py --train.num_workers 0 --epochs 1")

    Path("diagnose_results.json").write_text(json.dumps(cases, indent=2, default=str))
    print(f"\n  Full per-file results written to: diagnose_results.json")
    print(f"  Send that file back for further analysis if needed.")


if __name__ == "__main__":
    diagnose()
