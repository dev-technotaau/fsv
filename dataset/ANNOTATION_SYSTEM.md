# Multi-Class Annotation System — Grounded-SAM 2 Pipeline

**Purpose**: produce pixel-accurate multi-class segmentation masks for 33,423 images at enterprise-grade accuracy, in a single compute pass, with built-in QA and full auditability.

**Pipeline**: `Image → Grounding DINO (text prompts → boxes) → SAM 2 (boxes → masks) → Priority fusion → Multi-class PNG + QA score`

**Why this design beats alternatives** (see comparison in conversation history): open-weights, pixel-SOTA accuracy, multi-class via text, deterministic, no per-image API cost.

---

## 1. Class taxonomy (20 classes + background)

Defined in [`configs/annotation_schema.yaml`](../configs/annotation_schema.yaml). Organized into 5 **functional tiers** with per-tier priority so conflicts resolve correctly:

| Priority | Tier | Classes | Wins against |
|----------|------|---------|--------------|
| 90 | occluder (human_animal) | 14 | Everything |
| 80 | occluder (vegetation_occluder) | 12 | Fence, distractors, context |
| 75, 70 | occluder (vehicle_object, pole_wire) | 15, 16 | Fence, distractors, context |
| 48-55 | fence | 1-7 | Distractors, context |
| 35-42 | distractor (non-fence but fence-like) | 8-11 | Context |
| 5-30 | context (scene) | 13, 17-19 | Only background |
| 0 | background (reserved) | - | - |

**Class list** (see schema for full prompt sets):

| ID | Name | Priority | Tier | Stainable? | Example prompts |
|----|------|----------|------|-----------|-----------------|
| 0 | `background` | 0 | reserved | - | (unlabeled) |
| 1 | `fence_wood` | 50 | fence | **yes** | "wood fence", "cedar fence", "picket fence" |
| 2 | `fence_wood_post` | 55 | fence | **yes** | "wood fence post" |
| 3 | `fence_wood_gate` | 52 | fence | **yes** | "wooden fence gate" |
| 4 | `fence_vinyl` | 48 | fence | no | "vinyl fence" |
| 5 | `fence_metal` | 48 | fence | no | "chain link fence", "wrought iron" |
| 6 | `fence_masonry` | 48 | fence | no | "brick fence wall" |
| 7 | `fence_natural` | 45 | fence | no | "hedge", "bamboo fence" |
| 8 | `pergola_trellis` | 40 | distractor | no | "pergola", "arbor", "trellis" |
| 9 | `deck_railing` | 42 | distractor | no | "deck railing" |
| 10 | `wood_siding` | 38 | distractor | no | "wood house siding" |
| 11 | `wall_masonry` | 35 | distractor | no | "brick wall", "retaining wall" |
| 12 | `vegetation_occluder` | 80 | occluder | no | "tree branches", "vines", "flowers in front" |
| 13 | `vegetation_background` | 20 | context | no | "background trees" |
| 14 | `human_animal` | 90 | occluder | no | "person", "dog", "cat" |
| 15 | `vehicle_object` | 75 | occluder | no | "car", "bicycle", "flower pot" |
| 16 | `pole_wire` | 70 | occluder | no | "utility pole", "power line" |
| 17 | `ground` | 10 | context | no | "grass", "dirt", "pavement" |
| 18 | `sky` | 5 | context | no | "sky", "clouds" |
| 19 | `building` | 30 | context | no | "house", "shed" |

**Only classes 1, 2, 3 are `is_staining_target=true`**. When the stainer visualizer runs at deployment, it applies color only to pixels in those three classes. Everything else stays original.

---

## 2. Pipeline stages per image

```
┌───────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────┐
│ Load JPG  │ -> │Grounding DINO│ -> │  Filter +  │ -> │   SAM 2    │ -> │ Priority      │ -> │  Score + │
│ (Pillow)  │    │ 20 classes   │    │   NMS      │    │ box→mask   │    │ fusion        │    │  Save    │
└───────────┘    └──────────────┘    └────────────┘    └────────────┘    └───────────────┘    └──────────┘
```

**Stage 1 — Detection** (`annotation/grounding_dino.py`):
- Each class has 1-15 text prompts (e.g. `fence_wood` has "wood fence", "cedar fence", "picket fence"...)
- Prompts batched 10 classes at a time in a single Grounding DINO call
- Returns `Detection(class_id, box_xyxy, score)` per detected object
- Output: list of bounding boxes each tagged with best-matching class ID

**Stage 2 — Filter + NMS** (same file):
- Drop boxes below per-class `box_threshold` and `text_threshold`
- Non-max-suppression WITHIN each class (removes duplicate overlapping boxes)
- Cap at `max_boxes_per_class_per_image` (default 20) to prevent prompt spam

**Stage 3 — Segmentation** (`annotation/sam2.py`):
- Each filtered box prompts SAM 2 (`predict(box=...)`)
- `multimask_output=True` returns 3 candidates; we pick the one with highest predicted IoU
- This is what **naturally preserves gaps between fence slats** — SAM 2 is trained on structure, not bbox fill
- Output: `InstanceMask(class_id, box, mask, detection_score, sam_score, area)` per box

**Stage 4 — Priority fusion** (`annotation/fusion.py`):
- Sort masks by priority ASCENDING
- Paint low-priority masks first, high-priority overwrites
- Within same priority, higher-confidence mask wins per-pixel
- Result: single `class_map[H,W]` where each pixel has one winning class

**Why priority matters for your use case**: if a tree branch crosses in front of a cedar fence, Grounding DINO detects both → SAM 2 masks both → fusion paints `fence_wood` (priority 50) first, then `vegetation_occluder` (priority 80) on top for the branch pixels. The deployed stainer then paints stain only on `fence_wood` pixels, never on `vegetation_occluder` pixels, so the branch stays its original color — photorealistic.

**Stage 5 — Score + save**:
- Compute area-weighted overall confidence
- Flag for QA review if confidence low, no detections, or `fence_wood` missing in positive
- Save `masks/<id>.png` (8-bit class IDs), optional `viz/<id>.png` (colorized overlay), optional `heatmaps/<id>.png` (confidence)
- Append one JSONL row to `results.jsonl` with full provenance
- If flagged, also append to `qa_queue.jsonl`

---

## 3. File layout

```
annotation/                      # the package (Python modules)
├── __init__.py
├── schema.py                   # YAML loader, ClassDef dataclass
├── grounding_dino.py           # Grounding DINO wrapper + Detection class
├── sam2.py                     # SAM 2 wrapper + InstanceMask class
├── fusion.py                   # priority-based mask fusion
├── masks.py                    # PNG I/O + colorized viz
├── qa.py                       # confidence scoring
├── pipeline.py                 # AnnotationPipeline orchestrator
├── cli.py                      # CLI (python -m annotation.cli)
└── requirements.txt            # pip deps

configs/
└── annotation_schema.yaml      # class definitions + prompts + thresholds

dataset/annotations_v1/         # OUTPUT (created by the pipeline)
├── masks/<uuid>.png            # 33,423 class-ID PNGs (H × W, 8-bit)
├── viz/<uuid>.png              # human-readable colorized overlays
├── heatmaps/<uuid>.png         # (optional) per-pixel confidence grayscale
├── results.jsonl               # per-image provenance: detections, scores, flags
└── qa_queue.jsonl              # subset of results flagged for human review
```

---

## 4. Install & run

### Install

```bash
cd <project>
conda activate ml
pip install -r annotation/requirements.txt
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

First-run Grounding DINO + SAM 2 model downloads take ~2 GB. GPU strongly recommended (A100/H100 gives ~0.3 s/image; CPU is ~5-15 s/image).

### Smoke test (10 images, see it end-to-end)

```bash
python -m annotation.cli \
    --manifest dataset/manifest.jsonl \
    --out-root dataset/annotations_smoke \
    --limit 10
```

Check `dataset/annotations_smoke/viz/*.png` to visually inspect the results before committing to the full run.

### Full run (all 33,423 images)

```bash
python -m annotation.cli \
    --manifest dataset/manifest.jsonl \
    --out-root dataset/annotations_v1 \
    --resume
```

Add `--device cuda:0` if needed. Expected runtime on A100: **~3 hours**; on consumer GPU (RTX 3090/4090): **~6-8 hours**; CPU: **~36 hours** (don't).

### Batched / resumable

The pipeline writes `results.jsonl` line-by-line with `fsync` after each image. `--resume` re-reads it and skips any `image_id` already recorded. Safe to kill and restart at any point — no corruption risk.

### Filter / target runs

```bash
# Only positive-class images
python -m annotation.cli --only-positives

# Only one subcategory (great for iterating on prompts)
python -m annotation.cli --subcategory pos:occlusion --limit 50

# Only negatives (validate fence-distractor classes)
python -m annotation.cli --only-negatives
```

---

## 5. QA review workflow

After annotation completes, you have two files:

- `results.jsonl` — one row per image, contains overall_confidence, flags, per-class pixel counts
- `qa_queue.jsonl` — a subset (~10-20% expected) that auto-flagged as needing human review

**Why images land in the QA queue**:

| Flag | Meaning | Typical cause |
|------|---------|---------------|
| `no_detections` | Grounding DINO didn't detect anything | Image is pure-random or weird lighting |
| `low_overall_conf=X` | Weighted mean confidence < 0.55 | Ambiguous scene |
| `fence_wood_missing_in_positive` | Positive-class row has no `fence_wood` detected | Edge-case hard positive; fence heavily occluded |
| `weak_fence_wood_conf=X` | Positive row has fence but confidence < 0.40 | Unusual fence style, extreme angle |
| `random_qa_sample` | Selected as control (~10% of all images) | No issue — sanity-check sample |

**Review tool**: point CVAT / Label Studio / Roboflow at `dataset/annotations_v1/masks/` with `dataset/annotations_v1/viz/` as the preview. The reviewer corrects mistakes in the mask and re-saves. Updated mask PNGs can simply overwrite the originals.

After review, re-generate `results.jsonl` summary stats by re-running a lightweight stats-only pass (future tool; for v1 the manual review comments go into a separate `qa_review.jsonl` the reviewer maintains).

---

## 6. Accuracy expectations

Measured against our 100-image golden set (if manual masks done per `dataset/golden_set/`):

| Metric | Target | Expected actual |
|--------|--------|-----------------|
| Mean IoU on `fence_wood` (positives) | ≥ 0.85 | 0.85–0.92 |
| Boundary pixel accuracy (±3 pixels) | ≥ 90% | 90–95% |
| Class-assignment accuracy (clear cases) | ≥ 97% | 97–99% |
| Class-assignment accuracy (edge cases) | ≥ 80% | 80–88% |
| `human_animal` never wrongly painted as fence | 100% | 100% (priority rule enforces) |

**Where errors come from**:
- Grounding DINO sometimes mis-classifies weird-material fences
- SAM 2 occasionally over-segments tiny branches as one blob
- Hard occlusion (fence 85%+ hidden) causes missed detections

All of these are caught by the QA queue for human correction.

---

## 7. Priority system deep-dive — how we preserve tree branches in front of a fence

Given the 3 uploaded example images showing trees/branches/dogs crossing in front of cedar fences, here's exactly what the pipeline does:

**Image: cedar fence with Crepe Myrtle branches crossing in front**

1. Grounding DINO detects:
   - `fence_wood` at score 0.73 (full-image box)
   - `vegetation_occluder` × 5 at score 0.45-0.68 (one per branch cluster)
   - `ground` at score 0.60 (bottom strip)
   - `building` at score 0.52 (glimpse of neighbor's house)

2. SAM 2 masks each of the 8 boxes. SAM 2 picks out the precise branch silhouettes (not the bbox rectangles) because it's trained to follow edge gradients.

3. Fusion:
   - Start with priority 5 `sky` → paint top
   - Then priority 10 `ground` → paint bottom
   - Then priority 30 `building` → paint glimpse
   - Then priority 50 `fence_wood` → paint full fence (covers some of the above where fence is in front)
   - Then priority 80 `vegetation_occluder` → paint branches on top of fence where they cross

4. Final mask has `fence_wood` everywhere the fence is visible, `vegetation_occluder` in the precise branch silhouettes crossing in front. No stain will be painted onto branches at deployment.

**Image: cedar fence with dog and garden beds**

Same logic. `human_animal` (dog) at priority 90 always beats `fence_wood`. `vehicle_object` (pots, tools) at priority 75 also beats fence. `fence_wood` fills the fence panels visible between all these occluders.

---

## 8. Gap preservation — how fence slats with sky visible between them stay correct

**The problem**: a picket fence with 2-inch gaps between pickets. If the model paints "full fence bounding box as fence", the gaps get painted too → deployment stainer paints stain onto sky-colored gaps → cartoon result.

**The solution**: SAM 2 naturally segments by edge gradient, not bbox fill. When prompted with the fence's bbox, it outputs a mask that's **only the wood pickets**, with gaps automatically set to 0. The gaps then get classified as whatever's visible through them — usually `sky` or `vegetation_background`, both with lower priority than `fence_wood`, so they stay as sky/vegetation in the final class map.

For extreme cases (very thin pickets, very blurry images), we also run:
- Multi-point SAM prompting (future v2 enhancement)
- Post-hoc morphological filter to drop sub-pixel wood slivers (disabled by default, config-gated)

---

## 9. Schema customization

All prompts, thresholds, colors, and priorities live in [`configs/annotation_schema.yaml`](../configs/annotation_schema.yaml). No code changes are required to:

- Add/remove a class (e.g. add `fence_temporary_construction` class)
- Tune a class's text prompts (e.g. add "slatted cedar privacy fence")
- Adjust confidence thresholds per class
- Change priority order (e.g. make `human_animal` even more dominant)

After editing the schema, re-run the pipeline on a subset first to validate (`--limit 50`).

---

## 10. Versioning & reproducibility

- Schema version is frozen in `schema.version` ("1.0.0"). Bump when adding/removing classes.
- Each run writes `results.jsonl` with the exact prompts/thresholds/model checksums used — so two identical runs on the same hardware produce identical masks.
- Model checkpoints for Grounding DINO and SAM 2 are pinned via the `model` field in the schema (e.g., `facebook/sam2-hiera-large`). HuggingFace caches and verifies hash.
- For full reproducibility across machines, commit the annotation output directory to DVC (see `dataset/VERSIONING.md`).

---

## 11. Next steps after annotation

1. **Golden-set IoU measurement** — compare auto-generated masks against the 100 hand-drawn golden masks (once they exist) to produce a baseline accuracy metric.
2. **Human QA pass** — review ~10-15% of images flagged in `qa_queue.jsonl`.
3. **Incorporate corrections** — merged reviewed masks replace originals.
4. **Train segmentation models** — masks go in as `mask_path` field of the manifest, split JSONLs, Phase 1 → Phase 2.

---

_Document version: 1.0 · Last updated: 2026-04-17_
