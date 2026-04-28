# Golden Set — Hand-Masked Benchmark

_Generated: 2026-04-17T09:01:46+00:00_  
_Source split: `dataset\splits\test.jsonl` (SHA-256 prefix `6196a3b240c0…`)_

## What this is

A curated subset of **100 images** from the test split, stratified across 20 subcategories. These images need **pixel-perfect hand-drawn masks** by a senior annotator.

Once masked, the golden set serves as:

1. **Ground-truth benchmark** for Gemini auto-label quality — compute mean IoU of auto-labels against golden masks. Threshold: > 0.70 per-image mean.
2. **Inter-annotator agreement (IAA) target** — when a second annotator reviews these images, their IoU vs. the golden masks measures calibration. Target: > 0.90.
3. **Regression test during training** — compute model-vs-golden IoU every epoch. If it drops on a version, something regressed.
4. **Deployment sign-off** — final reported metric to client is `test.jsonl` IoU; golden set is the sanity-floor that should always exceed the test-set average.

## Contents

```
golden_set/
  manifest.jsonl       # the 100 selected rows
  images/              # NOT populated — use original paths from manifest
  masks/               # REVIEWER FILLS IN — PNGs named <id>.png
  GOLDEN_SET_README.md # this file
  selection_info.json  # audit (seed, source hash, etc.)
```

## Subcategory distribution

| Subcategory | Count |
|-------------|-------|
| `pos:style_cedar` | 22 |
| `pos:style_wood` | 17 |
| `pos:fence_general` | 16 |
| `pos:scene_context` | 11 |
| `pos:style_nonwood` | 8 |
| `pos:occlusion_mild` | 5 |
| `pos:damaged_construction` | 3 |
| `pos:occlusion` | 3 |
| `pos:multi_structure` | 2 |
| `pos:lighting` | 2 |
| `pos:general_positive` | 2 |
| `pos:fence_general_wood` | 1 |
| `pos:complex_background` | 1 |
| `pos:angle` | 1 |
| `pos:humans_animals` | 1 |
| `pos:painted_color` | 1 |
| `pos:reflection_water` | 1 |
| `pos:weather_extreme` | 1 |
| `pos:urban_rundown` | 1 |
| `pos:scale_extreme` | 1 |

## Mask file format

- **Format**: 8-bit single-channel PNG
- **Dimensions**: EXACTLY match source image dimensions
- **Pixel values**: `0` = background (not fence), `255` = fence
- **Naming**: `<manifest_id>.png` — e.g. `a1b2c3d4-....png`
- **Annotation software**: CVAT, Label Studio, LabelMe, or any tool that exports a PNG mask

## Before you start masking

1. Read [`dataset/ANNOTATION_GUIDELINES.md`](../ANNOTATION_GUIDELINES.md) end-to-end
2. Calibrate on 10 easy images first (clear cedar fences)
3. Do a second pass on your first 10 — you'll see inconsistencies
4. Then proceed with the remaining set
5. Budget ~**3–5 hours** for 100 images at pixel-perfect quality

## Reproducibility

- Seed: `42`
- Source split SHA-256: `6196a3b240c02e92d5d0ce241cd215ded4ad3732f224cbe5bf598488f2a8f135`
- Re-generate (same set): `tools/select_golden_set.py`

_Any changes to the source split → golden set becomes invalid _  
_(source hash mismatch) and should be regenerated + re-masked._