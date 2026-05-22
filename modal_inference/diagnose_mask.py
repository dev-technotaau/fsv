"""Diagnose why an image's stain didn't show up in the browser.

Mirrors the browser pipeline exactly:
  - same model (fence_model_dinov2.onnx)
  - same input size (518x518 ImageNet normalized)
  - same soft-mask thresholds (SOFT_MASK_LOW=0.72, SOFT_MASK_HIGH=0.88)
  - same filters: CC area, building filter, junk-blob, etc.

For each stage prints:
  - count of non-zero pixels
  - max / mean / median of mask values
  - if zero: which filter killed it
"""
import sys, os, json, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "fence-staining-visualizer" / "fence_model_dinov2.onnx"
INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Browser CONFIG (current values)
SOFT_MASK_LOW = 0.72
SOFT_MASK_HIGH = 0.88
CC_MIN_BLOB_AREA_PCT = 2.0
RECOLOR_FULL_ALPHA_THRESHOLD = 0.15
RECOVERY_CORE_THR = 0.85
RECOVERY_FILL_THR = 0.45
RECOVERY_DILATE_PX = 35
JUNK_BLOB_MAX_AREA_PCT = 0.6
JUNK_BLOB_MIN_ASPECT = 1.8
CC_AXIS_TOLERANCE_DEG = 30
CC_AXIS_MIN_PX = 500
CC_AXIS_MIN_ASPECT = 2.0
BUILDING_MIN_MEAN_CONF = 0.05
BUILDING_CONF_RATIO = 0.6
BUILDING_MIN_CC_PX = 500


def stats(name, arr):
    """Print summary stats for a soft mask."""
    nz = int((arr > 0).sum())
    total = arr.size
    if nz == 0:
        print(f"  [{name}]  ZERO PIXELS  (total {total})")
        return False
    pct = 100.0 * nz / total
    vals = arr[arr > 0]
    print(f"  [{name}]  nz={nz} ({pct:.2f}%)  "
          f"max={vals.max():.3f}  mean={vals.mean():.3f}  "
          f"median={np.median(vals):.3f}")
    return True


def label_4conn(mask):
    """Simple 4-connected CC labeling. Returns (labels, num_components)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    next_label = 1
    for y in range(h):
        for x in range(w):
            if mask[y, x] <= 0:
                continue
            left = labels[y, x-1] if x > 0 else 0
            up   = labels[y-1, x] if y > 0 else 0
            if left == 0 and up == 0:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            elif left and not up:
                labels[y, x] = left
            elif up and not left:
                labels[y, x] = up
            else:
                labels[y, x] = min(left, up)
                if left != up:
                    ra, rb = find(left), find(up)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    # Flatten to roots
    flat = labels.flatten()
    for i, v in enumerate(flat):
        if v:
            flat[i] = find(v)
    return flat.reshape(h, w)


def cc_props(labels, mask):
    """Compute per-CC area, mean confidence, bbox, PCA aspect ratio."""
    h, w = labels.shape
    ys, xs = np.where(labels > 0)
    roots = labels[ys, xs]
    props = {}
    for r in np.unique(roots):
        idx = roots == r
        rys, rxs = ys[idx], xs[idx]
        n = len(rys)
        # Mean confidence
        mvals = mask[rys, rxs]
        # Bbox
        ymin, ymax = rys.min(), rys.max()
        xmin, xmax = rxs.min(), rxs.max()
        # PCA aspect
        cx, cy = rxs.mean(), rys.mean()
        Cxx = ((rxs - cx) ** 2).mean()
        Cyy = ((rys - cy) ** 2).mean()
        Cxy = ((rxs - cx) * (rys - cy)).mean()
        half_diff = (Cxx - Cyy) / 2
        disc = np.sqrt(half_diff ** 2 + Cxy ** 2)
        trace = Cxx + Cyy
        lmax = trace / 2 + disc
        lmin = trace / 2 - disc
        aspect = lmax / lmin if lmin > 1e-6 else float('inf')
        angle_deg = 0.5 * np.degrees(np.arctan2(2 * Cxy, Cxx - Cyy))
        props[int(r)] = dict(n=n, mean=float(mvals.mean()),
                             bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                             aspect=float(aspect), angle_deg=float(angle_deg))
    return props


def soft_mask(probs, low=SOFT_MASK_LOW, high=SOFT_MASK_HIGH):
    """Apply browser's soft-mask threshold logic."""
    out = np.zeros_like(probs)
    r = max(1e-6, high - low)
    above_high = probs >= high
    ramp = (probs > low) & (probs < high)
    out[above_high] = probs[above_high]
    out[ramp] = probs[ramp] * ((probs[ramp] - low) / r)
    return out


def derive_recolor_alpha(mask, thr=RECOLOR_FULL_ALPHA_THRESHOLD):
    """Mirror the browser's deriveRecolorAlpha()."""
    out = np.zeros_like(mask)
    above = mask >= thr
    below_above = (mask > 0) & (mask < thr)
    out[above] = 1.0
    out[below_above] = mask[below_above] / thr
    return out


def apply_browser_preprocess(img, max_dim=1024):
    """Mirror imageToUploadBlob() with enhanceMode='mild'.
    - Resize to max_dim preserving aspect
    - contrast(1.12), saturation(1.06) via per-channel math
    - autoLevelsCanvas with 1.5/98.5 percentile stretch
    """
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    # Saturation 1.06: blend toward luminance with factor (1 - 1.06) = -0.06
    # Browser canvas filter `saturate(1.06)` formula: c' = lum + 1.06 * (c - lum)
    lum = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114)[..., None]
    arr = lum + 1.06 * (arr - lum)
    # Contrast 1.12: c' = 128 + 1.12 * (c - 128)
    arr = 128.0 + 1.12 * (arr - 128.0)
    arr = np.clip(arr, 0, 255)

    # Auto-levels: 1.5 / 98.5 percentile stretch on luminance histogram
    lum_int = ((arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3).astype(np.int32).clip(0, 255)
    hist, _ = np.histogram(lum_int, bins=256, range=(0, 256))
    cum = hist.cumsum()
    total = lum_int.size
    lo = int(np.searchsorted(cum, total * 0.015))
    hi = int(np.searchsorted(cum, total * 0.985))
    if hi - lo >= 10 and not (lo <= 8 and hi >= 247):
        scale = 255.0 / (hi - lo)
        arr = (arr - lo) * scale
        arr = np.clip(arr, 0, 255)
        print(f"  [preprocess] auto-levels stretch: [{lo}, {hi}] → [0, 255]")
    else:
        print(f"  [preprocess] auto-levels skipped (lo={lo}, hi={hi})")

    return Image.fromarray(arr.astype(np.uint8))


def main(image_path):
    raw_img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path}")
    print(f"Original size: {raw_img.size[0]}x{raw_img.size[1]}")

    print(f"\n=== Applying browser preprocess (mild: contrast 1.12, sat 1.06, autoLevels) ===")
    img = apply_browser_preprocess(raw_img, max_dim=1024)
    orig_w, orig_h = img.size
    print(f"After preprocess: {orig_w}x{orig_h}")

    # JPEG round-trip — mirror what canvas.toBlob('image/jpeg', 0.85) does
    # in imageToUploadBlob() before sending to Modal. The lossy compression
    # CAN affect model output (especially on already-degraded images).
    import io as _io
    buf = _io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    upload_kb = buf.tell() / 1024
    img = Image.open(_io.BytesIO(buf.getvalue())).convert("RGB")
    print(f"After JPEG round-trip @ quality 85: {upload_kb:.1f} KB upload size")

    # Preprocess for model
    img518 = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img518, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # Run model
    print(f"\nLoading model: {MODEL_PATH}")
    t0 = time.time()
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(["output"], {"input": arr})[0][0, 0]  # (518, 518)
    print(f"Inference: {(time.time()-t0)*1000:.0f} ms")
    print(f"Raw model output: shape={out.shape}, dtype={out.dtype}")
    print(f"  raw range: min={out.min():.4f}  max={out.max():.4f}  mean={out.mean():.4f}")
    print(f"  raw distribution:")
    for thr in [0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
        pct = 100.0 * (out >= thr).mean()
        print(f"    p >= {thr}:  {pct:6.2f}%  ({(out >= thr).sum()} px)")

    # Upsample to original resolution (Pillow BILINEAR ≈ browser's bilinearResize)
    probs_img = Image.fromarray(out.astype(np.float32), mode="F")
    probs_img = probs_img.resize((orig_w, orig_h), Image.BILINEAR)
    probs = np.asarray(probs_img, dtype=np.float32)
    print(f"\nUpsampled probs: shape={probs.shape}")
    print(f"  upsampled max={probs.max():.4f}  mean={probs.mean():.4f}")

    # ---- Step 3: soft mask threshold ----
    print(f"\n=== Pipeline trace ===")
    mask = soft_mask(probs)
    if not stats("Step 3 (soft mask LOW=0.72 HIGH=0.88)", mask):
        print("\nDIAGNOSIS: soft mask kills everything.")
        print(f"  → raw probs never reach LOW={SOFT_MASK_LOW}.")
        print(f"  → max raw prob was only {probs.max():.3f}.")
        return

    # ---- Step 4: CC area cleanup ----
    min_area = int(probs.size * CC_MIN_BLOB_AREA_PCT / 100)
    print(f"\n  CC area min = {min_area} px ({CC_MIN_BLOB_AREA_PCT}% of {probs.size})")
    binary = (mask > 0).astype(np.uint8)
    labels = label_4conn(binary)
    props = cc_props(labels, mask)
    print(f"  CCs found: {len(props)}")
    for r, p in sorted(props.items(), key=lambda kv: -kv[1]['n'])[:10]:
        print(f"    CC root={r}: n={p['n']}, mean={p['mean']:.3f}, "
              f"aspect={p['aspect']:.2f}, angle={p['angle_deg']:.1f}°, "
              f"bbox={p['bbox']}")

    kept = np.zeros_like(mask)
    for r, p in props.items():
        if p['n'] >= min_area:
            kept[labels == r] = mask[labels == r]
    mask = kept
    if not stats("Step 4-5 (CC area cleanup)", mask):
        print("\nDIAGNOSIS: CC area cleanup killed all detections.")
        print(f"  → Largest CC was {max(p['n'] for p in props.values())} px, "
              f"below threshold {min_area} px.")
        return

    # ---- Color filters (Steps 6-8.5) — get original pixel data ----
    pixel_arr = np.asarray(img, dtype=np.float32)   # H, W, 3 (already preprocessed)
    print(f"\n  --- Color filters ---")

    # Step 6: vegetation
    if True:
        r, g, b = pixel_arr[..., 0], pixel_arr[..., 1], pixel_arr[..., 2]
        veg = (g - np.maximum(r, b)) > 25
        drops = ((mask > 0) & veg).sum()
        mask = np.where(veg, 0, mask)
        print(f"    [veg] dropped {drops} green-dominant px")
        if not stats("after veg", mask):
            print("\nDIAGNOSIS: vegetation filter dropped everything.")
            return

    # Step 6.5: sky (top 40%, lum>=175, sat<0.10)
    if True:
        top_cutoff = int(orig_h * 0.40)
        r, g, b = pixel_arr[..., 0], pixel_arr[..., 1], pixel_arr[..., 2]
        lum = (r + g + b) / 3
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        is_sky = np.zeros_like(mask, dtype=bool)
        is_sky[:top_cutoff] = (lum[:top_cutoff] >= 175) & (sat[:top_cutoff] < 0.10)
        drops = ((mask > 0) & is_sky).sum()
        mask = np.where(is_sky, 0, mask)
        print(f"    [sky] dropped {drops} sky-like px in top 40% (lum>=175, sat<0.10)")
        if not stats("after sky", mask):
            print("\nDIAGNOSIS: sky filter dropped everything.")
            return

    # Step 7: bark (adaptive — drop pixels less saturated than fence mean AND
    # brightness within 40 of fence mean)
    if True:
        strong = mask > 0.4
        if strong.sum() >= 100:
            r, g, b = pixel_arr[..., 0], pixel_arr[..., 1], pixel_arr[..., 2]
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
            bright = (r + g + b) / 3
            mean_sat = sat[strong].mean()
            mean_bright = bright[strong].mean()
            sat_cutoff = mean_sat - 0.10
            bark_mask = (mask > 0) & (sat < sat_cutoff) & (np.abs(bright - mean_bright) < 40)
            drops = bark_mask.sum()
            mask = np.where(bark_mask, 0, mask)
            print(f"    [bark-adaptive] dropped {drops} px (fence sat {mean_sat:.2f}, "
                  f"bright {mean_bright:.0f}, satGap 0.10, brightDelta 40)")
            if not stats("after bark", mask):
                print("\nDIAGNOSIS: bark filter dropped everything!")
                print(f"  → fence_mean_sat={mean_sat:.3f}, sat_cutoff={sat_cutoff:.3f}")
                print(f"  → bark check drops if pixel sat < {sat_cutoff:.3f} AND |bright - {mean_bright:.0f}| < 40")
                return
        else:
            print(f"    [bark] skipped (only {strong.sum()} strong samples)")

    # Step 8: trunk (adaptive color distance)
    if True:
        strong = mask > 0.4
        if strong.sum() >= 100:
            r, g, b = pixel_arr[..., 0], pixel_arr[..., 1], pixel_arr[..., 2]
            mean_r = r[strong].mean()
            mean_g = g[strong].mean()
            mean_b = b[strong].mean()
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
            mean_sat = sat[strong].mean()
            dist_sq = (r - mean_r)**2 + (g - mean_g)**2 + (b - mean_b)**2
            hard_drop = dist_sq > (90 * 90)
            soft_drop = (dist_sq > (55 * 55)) & (sat < mean_sat - 0.10)
            trunk_mask = (mask > 0) & (hard_drop | soft_drop)
            drops = trunk_mask.sum()
            mask = np.where(trunk_mask, 0, mask)
            print(f"    [trunk] dropped {drops} px (fence mean RGB {mean_r:.0f},{mean_g:.0f},{mean_b:.0f}, "
                  f"sat {mean_sat:.2f})")
            if not stats("after trunk", mask):
                print("\nDIAGNOSIS: trunk filter dropped everything!")
                return
        else:
            print(f"    [trunk] skipped (only {strong.sum()} strong samples)")

    # Step 8.5: CC color outliers — ADAPTIVE (k * stddev, min_dist 80, k=2.8)
    K_STDDEV = 2.8
    MIN_DIST = 80
    if True:
        binary = (mask > 0).astype(np.uint8)
        labels = label_4conn(binary)
        props = cc_props(labels, mask)
        r, g, b = pixel_arr[..., 0], pixel_arr[..., 1], pixel_arr[..., 2]
        total_drop = 0
        thr_log = []
        for root, p in props.items():
            if p['n'] < 300:
                continue
            mask_cc = labels == root
            mean_r = r[mask_cc].mean()
            mean_g = g[mask_cc].mean()
            mean_b = b[mask_cc].mean()
            dist_sq = (r - mean_r)**2 + (g - mean_g)**2 + (b - mean_b)**2
            stddev = float(np.sqrt(dist_sq[mask_cc].mean()))
            eff = max(MIN_DIST, K_STDDEV * stddev)
            thr_log.append(f"{root}:stddev={stddev:.0f},thr={eff:.0f}")
            outlier = mask_cc & (dist_sq > (eff * eff))
            d = int(outlier.sum())
            mask = np.where(outlier, 0, mask)
            total_drop += d
        print(f"    [cc-outlier-adaptive] dropped {total_drop} px ({', '.join(thr_log[:3])})")
        if not stats("after cc-outlier", mask):
            print("\nDIAGNOSIS: cc-outlier filter dropped everything!")
            return

    # ---- Step 8.7+ — recompute CCs ----
    binary = (mask > 0).astype(np.uint8)
    labels = label_4conn(binary)
    props = cc_props(labels, mask)
    print(f"\n  After color filters: {len(props)} CCs")
    for r, p in sorted(props.items(), key=lambda kv: -kv[1]['n']):
        print(f"    CC root={r}: n={p['n']}, mean={p['mean']:.3f}, "
              f"aspect={p['aspect']:.2f}, angle={p['angle_deg']:.1f}°")

    # ---- Step 8.8: Per-CC PCA (drop off-axis CCs) ----
    elongated = {r: p for r, p in props.items()
                 if p['n'] >= CC_AXIS_MIN_PX and p['aspect'] >= CC_AXIS_MIN_ASPECT}
    if len(elongated) >= 2:
        ref_root = max(elongated.items(), key=lambda kv: kv[1]['n'])[0]
        ref_angle = elongated[ref_root]['angle_deg']
        dropped = []
        for r, p in elongated.items():
            if r == ref_root:
                continue
            diff = abs(p['angle_deg'] - ref_angle)
            if diff > 90:
                diff = 180 - diff
            if diff > CC_AXIS_TOLERANCE_DEG:
                dropped.append((r, p, diff))
        print(f"\n  [Step 8.8 CC PCA] ref={ref_root} @ {ref_angle:.1f}°, "
              f"{len(elongated)} elongated CCs, {len(dropped)} off-axis")
        for r, p, diff in dropped:
            print(f"    DROP CC root={r}: n={p['n']}, angle={p['angle_deg']:.1f}° "
                  f"(diff {diff:.1f}° > {CC_AXIS_TOLERANCE_DEG}°)")
            mask[labels == r] = 0
        if not stats("Step 8.8 (per-CC PCA)", mask):
            print("\nDIAGNOSIS: per-CC PCA dropped everything.")
            return
    else:
        print(f"\n  [Step 8.8 CC PCA] only {len(elongated)} elongated CCs — skipped")

    # ---- Step 8.9: junk-blob filter ----
    max_area = int(probs.size * JUNK_BLOB_MAX_AREA_PCT / 100)
    junk_dropped = []
    for r, p in props.items():
        if p['n'] > max_area or p['n'] < 50:
            if p['n'] < 50:
                junk_dropped.append((r, p, 'tiny'))
                mask[labels == r] = 0
            continue
        if p['aspect'] < JUNK_BLOB_MIN_ASPECT:
            junk_dropped.append((r, p, f'aspect {p["aspect"]:.2f} < {JUNK_BLOB_MIN_ASPECT}'))
            mask[labels == r] = 0
    print(f"\n  [Step 8.9 junk-blob] max_area={max_area}px, min_aspect={JUNK_BLOB_MIN_ASPECT}")
    if junk_dropped:
        for r, p, reason in junk_dropped:
            print(f"    DROP CC root={r}: n={p['n']}, aspect={p['aspect']:.2f} ({reason})")
    if not stats("Step 8.9 (junk-blob filter)", mask):
        print("\nDIAGNOSIS: junk-blob filter dropped everything!")
        return

    # ---- Step 9: building filter (NEW: ratio-based, count big CCs for early return) ----
    binary = (mask > 0).astype(np.uint8)
    labels = label_4conn(binary)
    props = cc_props(labels, mask)
    big = {r: p for r, p in props.items() if p['n'] >= BUILDING_MIN_CC_PX}
    if len(big) <= 1:
        print(f"\n  [Step 9 building] only {len(big)} big CC(s) (≥{BUILDING_MIN_CC_PX} px) — "
              f"early return (no comparison signal)")
    else:
        # Use the LARGEST big CC's mean as fence baseline (not max mean).
        # See JS comment in filterBuildings for full rationale.
        largest_root = max(big.keys(), key=lambda r: big[r]['n'])
        fence_mean = big[largest_root]['mean']
        thr = max(BUILDING_MIN_MEAN_CONF, fence_mean * BUILDING_CONF_RATIO) if fence_mean > 0 else BUILDING_MIN_MEAN_CONF
        print(f"\n  [Step 9 building] big CCs={len(big)}, "
              f"baseline=largest CC#{largest_root} ({big[largest_root]['n']} px, "
              f"mean={fence_mean:.3f}), threshold={thr:.3f} (ratio {BUILDING_CONF_RATIO})")
        bld_dropped = []
        for r, p in big.items():
            if p['mean'] < thr:
                bld_dropped.append((r, p))
                mask[labels == r] = 0
        if bld_dropped:
            for r, p in bld_dropped:
                print(f"    DROP CC root={r}: n={p['n']}, mean={p['mean']:.3f} < thr={thr:.3f}")
        else:
            print(f"    no CCs dropped")
    if not stats("Step 9 (building filter)", mask):
        print("\nDIAGNOSIS: building filter dropped everything!")
        return

    # ---- Final: derived recolor alpha (what determines if stain shows) ----
    recolor = derive_recolor_alpha(mask)
    print(f"\n=== FINAL recolor alpha (what determines stain visibility) ===")
    stats("recolor alpha", recolor)
    full = (recolor >= 1.0).sum()
    partial = ((recolor > 0) & (recolor < 1)).sum()
    print(f"  full-alpha (=1.0): {full} px ({100.0*full/recolor.size:.2f}%)")
    print(f"  partial-alpha (edge ramp): {partial} px")
    if full < 1000:
        print("\nDIAGNOSIS: very few pixels reach full-alpha — stain will be invisible.")
        print(f"  → mask values mostly below RECOLOR_FULL_ALPHA_THRESHOLD ({RECOLOR_FULL_ALPHA_THRESHOLD})")

    # Save visualization
    out_path = Path(image_path).with_suffix('.diag.png')
    vis = Image.new("RGB", (orig_w * 2, orig_h))
    vis.paste(img, (0, 0))
    overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    overlay[..., 0] = (recolor * 255).astype(np.uint8)
    overlay[..., 1] = (recolor * 128).astype(np.uint8)
    blended = (0.5 * np.asarray(img) + 0.5 * overlay).clip(0, 255).astype(np.uint8)
    vis.paste(Image.fromarray(blended), (orig_w, 0))
    vis.save(out_path)
    print(f"\nVisualization (orig | recolor-alpha overlay) saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_mask.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
