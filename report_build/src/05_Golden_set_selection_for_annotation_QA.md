# AREA: Golden set selection for annotation/QA

## SUMMARY
The golden set is a 100-image, stratified, seed-controlled benchmark drawn exclusively from the held-out test split (test.jsonl), built by tools/select_golden_set.py. It spans 20 positive subcategories with at least one image per subcategory, and is hand-masked at the pixel level to serve as ground truth for four QA purposes: scoring auto-label quality, measuring inter-annotator agreement, regression-testing models during development, and final deployment sign-off. As checked into the repo it is fully populated: 100 manifest rows, 100 hand-drawn masks (0/1 PNGs), plus an auto-annotation run (100 results) that flags every image into a QA review queue. Consumers include the SAM3 test pipeline (IoU vs golden), the Grounding DINO LoRA fine-tuner (per-epoch golden recall eval), and the manual SAM annotation tool. Note: the README/docstring describe details (Gemini, 0/255 masks, empty masks dir) that no longer match the current code/artifacts.

## KEY_FACTS
- tools/select_golden_set.py selects a default of N=100 images (--n default 100, select_golden_set.py:221).
- The golden set is drawn EXCLUSIVELY from the test split: default --source is dataset/splits/test.jsonl (select_golden_set.py:216-217), described as 'drawn exclusively from the TEST split (so the golden set is held out from training)' (lines 5-7).
- Selection is stratified by (class, subcategory): rows are grouped by f"{class}:{subcategory}" (select_golden_set.py:88) and sampled proportionally to bucket size (raw = n_target * len(items) / total, line 96).
- ensure_minority (default ON, --no-ensure-minority to disable) clamps every subcategory with >=1 test row to at least 1 sample: max(1, round(raw)) (select_golden_set.py:97).
- Selection is pos-only by default: negatives are excluded unless --include-neg is passed (select_golden_set.py:82-83, 224-225); rationale is that negative masks are trivially empty and low-value for IAA (lines 76-77).
- Determinism: global seed default 42 (--seed, line 223); per-subcategory RNG = random.Random(seed ^ stable_hash_seed(key)) where stable_hash_seed is sha256-derived (lines 59-60, 121); items sorted by id before sampling and final list sorted by id (lines 122, 126).
- The exact-count adjustment adds/trims from the largest buckets to hit n_target exactly, never trimming a bucket below 1 (select_golden_set.py:99-112).
- The source test split is hashed (sha256 streamed in 1 MiB chunks) and recorded for reproducibility; mismatch invalidates the golden set (lines 241-245, 205-206).
- ACTUAL realized golden set (dataset/golden_set/selection_info.json): n=100, seed=42, n_subcats=20, include_neg=false, ensure_minority=true, copied_images=false, generated_at 2026-04-17T09:01:46Z.
- Recorded source_sha256 of test.jsonl = 6196a3b240c02e92d5d0ce241cd215ded4ad3732f224cbe5bf598488f2a8f135 (selection_info.json:7).
- Realized subcategory distribution (all 'pos'): style_cedar 22, style_wood 17, fence_general 16, scene_context 11, style_nonwood 8, occlusion_mild 5, damaged_construction 3, occlusion 3, multi_structure 2, lighting 2, general_positive 2, and 9 singletons (fence_general_wood, complex_background, angle, humans_animals, painted_color, reflection_water, weather_extreme, urban_rundown, scale_extreme) = 1 each (selection_info.json:12-32).
- dataset/golden_set/manifest.jsonl has exactly 100 lines (verified via wc -l).
- The source test split dataset/splits/test.jsonl contains 5,016 rows, so the golden set is ~2% of the test split (100/5016).
- dataset/golden_set/masks/ is fully populated: 100 hand-drawn PNG masks (NOT empty as the docstring at line 19 claims).
- Actual golden masks are 8-bit single-channel PNGs with pixel values {0,1} (verified on samples), NOT the {0,255} the README claims (GOLDEN_SET_README.md:57).
- Consumer sam3_test_pipeline.py computes IoU vs golden masks: compute_iou treats golden==1 as foreground (line 436), compares class_map>0 against the golden array (line 650), and reports mean/median/P25/P75 plus >=0.5, >=0.7, >=0.9 buckets (lines 674-683).
- Consumer finetune_dino_lora.py uses the golden set as its per-epoch eval set: --eval-manifest defaults to dataset/golden_set/manifest.jsonl and --eval-masks to dataset/golden_set/masks (lines 524-525); evaluate() reports detection recall and mean_max_score on positive images, treating has_fence = (mask>0).sum()>256 (lines 318-372).
- manual_sam.py is the tool that produces the golden masks: documented usage targets --manifest dataset/golden_set/manifest.jsonl --out-root dataset/golden_set, writing masks/ + masks_preview/ + viz/ (manual_sam.py:8-10, defaults lines 269-271).
- QA thresholds documented in the generated README: Gemini auto-label IoU > 0.70 per-image mean; inter-annotator agreement target IoU > 0.90; per-epoch model-vs-golden IoU as regression guard; golden IoU should always exceed the test-set average (GOLDEN_SET_README.md:12-15).
- Effort budget documented: ~3-5 hours for a senior annotator to pixel-mask 100 images (GOLDEN_SET_README.md:67; select_golden_set.py:197).
- An auto-annotation QA run exists at dataset/golden_set/auto_annotations/: 100 results.jsonl rows and 100 qa_queue.jsonl rows, 100 auto masks (tri-class {0,1,2}: 0=bg,1=fence_wood,2=not_target).
- In that auto run, all 100 images carry the 'random_qa_sample' flag and needs_review=true; mean overall_confidence = 0.304; 5 positives flagged 'fence_wood_missing_in_positive'.
- select_golden_set.py has overwrite protection: refuses to overwrite an existing manifest.jsonl unless --force (lines 271-273), and supports --dry-run (line 232).
- Writes are atomic (write_jsonl_atomic uses a .tmp file then replace, lines 50-56).

## FILE_ROLES
- [current] tools/select_golden_set.py — The golden-set selector: stratified, seeded, pos-only sampling from test.jsonl; writes manifest.jsonl, selection_info.json, GOLDEN_SET_README.md and empty images/masks dirs. 326 lines.
- [current] dataset/golden_set/manifest.jsonl — The 100 selected golden rows (id, path, class, subcategory, sha256, dims, vision_label, etc.). 88,217 bytes, 100 lines.
- [current] dataset/golden_set/selection_info.json — Reproducibility audit (seed=42, n=100, source sha256, 20-subcat distribution, flags).
- [current] dataset/golden_set/GOLDEN_SET_README.md — Auto-generated usage doc (QA roles, thresholds, mask format). Partly STALE: claims 0/255 masks and empty masks dir; references Gemini auto-label as the thing being benchmarked.
- [current] dataset/golden_set/masks/ — 100 hand-drawn ground-truth PNG masks (values 0/1), filled by the senior annotator via manual_sam. This is the QA ground truth.
- [artifact] dataset/golden_set/masks_preview/ — 100 human-viewable preview overlays of the hand masks (artifact of the annotation tool).
- [artifact] dataset/golden_set/viz/ — 100 visualization overlays from annotation (artifact).
- [current] dataset/golden_set/images/ — Empty (copied_images=false); manifest references original data_scraped paths instead.
- [current] dataset/golden_set/auto_annotations/ — An automated annotation/QA run over the golden images: results.jsonl (100), qa_queue.jsonl (100), tri-class masks/ (0/1/2), heatmaps/, viz/. Used to compare auto vs golden and to build a human review queue.
- [current] annotation/sam3_test_pipeline.py — Consumer: runs SAM3 annotation pipeline and scores predicted masks via IoU against golden masks; defaults --golden-dir to dataset/golden_set/masks.
- [current] annotation/finetune_dino_lora.py — Consumer: Grounding DINO LoRA fine-tuner that evaluates on the golden set every epoch (detection recall + mean score) and saves best adapter by golden recall.
- [current] annotation/manual_sam.py — Producer: interactive SAM2 click-to-segment tool used to hand-mask the golden manifest into golden_set/masks.
- [current] annotation/scene_classifier.py — Related QA helper: CLIP-based OOD scene filter; docstring notes the golden set sometimes contains non-fence images to flag.
- [backup] annotation/refine_backup.py — Backup of an edge/mask refiner; mentions the SAM2 golden_set bug it fixes. Backup copy.
- [current] dataset/splits/test.jsonl — Source split the golden set is drawn from (5,016 rows); sha256 recorded in selection_info.json.

## NARRATIVE
## Golden set: how the QA benchmark is built and used

The golden set is the project's hand-curated ground-truth benchmark, and the single source of truth for "did this model get better or worse." It is produced by `tools/select_golden_set.py`, a self-contained, dependency-light script (stdlib only) that picks a small, diverse, hand-maskable subset of images and lays out a directory the annotation team fills in.

### Where it sits in the workflow

The golden set is drawn *exclusively from the held-out test split* — `dataset/splits/test.jsonl` is the default and only intended source (`select_golden_set.py:216-217`, with the docstring at lines 5-7 spelling out the reasoning: "drawn exclusively from the TEST split (so the golden set is held out from training + is representative of deployment inputs)"). That positioning matters: because none of these images were used in training, model-vs-golden IoU is an honest generalization signal, and because they come from the same distribution that ships to production, the benchmark is representative of real inputs. The flow is: scrape → label → split into train/val/test → select golden subset from test → senior annotator hand-masks it → everything downstream (auto-label QA, model training, sign-off) measures itself against those masks.

### How images are selected

Selection is **stratified, proportional, seeded, and pos-only by default**. The script groups every candidate row by a composite `class:subcategory` key (`select_golden_set.py:88`) and computes a proportional target per bucket — `raw = n_target * len(items) / total` (line 96). With the default `--n 100`, that gives bigger subcategories more slots while still guaranteeing coverage of the rare, hard ones: `ensure_minority` (on by default) clamps every subcategory that has at least one test row up to a floor of one image via `max(1, round(raw))` (line 97). A small fix-up loop then nudges the totals up or down from the largest buckets so the final count lands exactly on the target, but it never trims a bucket below one (lines 99-112). Negatives are dropped unless you pass `--include-neg`, on the stated logic that empty negative masks are low-value for inter-annotator agreement (lines 76-77, 82-83).

Reproducibility is taken seriously. There is a global seed (default 42, line 223), but each subcategory gets its own deterministic RNG seeded by `seed XOR stable_hash_seed(key)`, where `stable_hash_seed` is the first four bytes of a SHA-256 of the bucket name (lines 59-60, 121). Items are sorted by `id` before sampling and the final selection is re-sorted by `id` (lines 122, 126), so the same inputs always yield byte-identical output. On top of that, the script streams a SHA-256 of the entire source split (1 MiB chunks, lines 241-245) and records it in the audit file; the README explicitly states that if the source split changes, the golden set's source-hash no longer matches and the set must be regenerated and re-masked (lines 205-206). Writes are atomic (temp file + replace, lines 50-56), and there's overwrite protection — it refuses to clobber an existing `manifest.jsonl` without `--force` (lines 271-273) — plus a `--dry-run` that prints the distribution and writes nothing.

The script emits five things into `dataset/golden_set/`: `manifest.jsonl` (the selected rows), `selection_info.json` (the audit), `GOLDEN_SET_README.md` (an auto-generated how-to), and empty `images/` and `masks/` directories. By default images are *not* copied — the manifest just references the originals under `data_scraped/` — unless you pass `--copy` (lines 229-231, 306-317).

### The actual golden set on disk

The realized set matches the defaults exactly. From `dataset/golden_set/selection_info.json`: **100 images, seed 42, 20 subcategories, positives only, images not copied**, generated 2026-04-17. The recorded test-split hash is `6196a3b2…f2a8f135`. The manifest has exactly 100 lines, confirmed. The 100 images are about 2% of the 5,016-row test split.

The subcategory spread is naturally long-tailed, which is exactly what you want from proportional-plus-minority sampling: the common wood/cedar fences dominate (`style_cedar` 22, `style_wood` 17, `fence_general` 16, `scene_context` 11, `style_nonwood` 8), while a long tail of hard or rare conditions each gets a guaranteed slot — `occlusion_mild` 5, `damaged_construction` and `occlusion` 3 each, `multi_structure`/`lighting`/`general_positive` 2 each, and nine singletons covering the genuinely hard corners: extreme scale, weather extremes, reflections on water, painted color, odd angles, humans/animals in frame, urban-rundown scenes, complex backgrounds, and `fence_general_wood`. So the "hard subcategory" stratification the client cares about is real and visible in the distribution.

Critically, **the masks are not empty**: `dataset/golden_set/masks/` contains all 100 hand-drawn PNGs. The docstring (line 19) still calls this an "empty dir; reviewer fills with PNGs," which was true at generation time but is now stale — the annotation pass has been completed. There are also `masks_preview/` and `viz/` directories (100 files each) that the annotation tool produces as human-viewable overlays; those are artifacts, not the benchmark itself.

One real discrepancy worth flagging to the client: the README/docstring say masks should be 8-bit PNGs with values `0`/`255` (`GOLDEN_SET_README.md:57`), but the actual checked-in golden masks use values `0`/`1`. The consumers are written for `0`/`1` — `sam3_test_pipeline.compute_iou` treats `golden_mask == 1` as foreground (`sam3_test_pipeline.py:436`) and the LoRA evaluator uses `mask > 0` (`finetune_dino_lora.py:335`) — so the pipeline is internally consistent; it's the documentation that drifted.

### The masks are produced by manual SAM

The hand-masking is done with `annotation/manual_sam.py`, an interactive SAM-2 click-to-segment tool. Its documented invocation points straight at the golden set (`--manifest dataset/golden_set/manifest.jsonl --out-root dataset/golden_set`, lines 8-10), and it writes into exactly the `masks/ + masks_preview/ + viz/` layout the rest of the pipeline expects. Left-click adds a positive point, right-click a negative point, space saves and advances; advancing with no clicks saves an empty mask (the correct behavior for a true negative). The README budgets 3-5 hours for a senior annotator to mask all 100 at pixel quality.

### How it's used for regression QA after each model update

The generated README lays out four QA roles (lines 12-15): (1) ground-truth benchmark for auto-label quality, with a pass threshold of mean IoU > 0.70 per image; (2) inter-annotator agreement target of IoU > 0.90 when a second annotator re-masks the same images; (3) a per-epoch regression test — recompute model-vs-golden IoU every epoch and treat any drop as a regression; and (4) deployment sign-off, where the headline number reported to the client is test-set IoU and the golden set is the sanity floor that should always sit above the test-set average.

Two consumers wire this up concretely. `annotation/sam3_test_pipeline.py` runs the SAM3 annotation pipeline over the golden manifest and scores each predicted mask against the golden mask, defaulting `--golden-dir` to `dataset/golden_set/masks` (lines 548-550). Its summary block is the regression dashboard: mean/median/P25/P75 IoU plus counts at the 0.5/0.7/0.9 thresholds (lines 674-683) — so after any pipeline change you re-run and watch those buckets. `annotation/finetune_dino_lora.py` uses the golden set as its *eval set every epoch* (`--eval-manifest`/`--eval-masks` default to the golden paths, lines 524-525); its `evaluate()` reports detection recall and mean max detection score on positive images and the trainer keeps the best adapter by golden recall (docstring lines 15-16). That is the "if it drops on a version, something regressed" mechanism in code.

There is also an automated QA run captured under `dataset/golden_set/auto_annotations/`: 100 `results.jsonl` rows and a 100-row `qa_queue.jsonl`, with tri-class auto masks (0=background, 1=fence_wood, 2=not_target), heatmaps and viz. In that run every image is tagged `random_qa_sample` and `needs_review=true`, mean overall confidence is 0.304, and five positives are flagged `fence_wood_missing_in_positive` — i.e., the automated annotator missed the fence on five images the golden masks confirm contain one. That is precisely the kind of miss the golden set exists to catch.

### Documentation drift to note for the client

The script docstring (line 11) and the generated README (line 12) describe the auto-label benchmark target as *Gemini* auto-labels. Per the project's own memory notes, the current labeling pipeline is Grounding DINO + SAM, not Gemini, so this wording is stale boilerplate carried forward from the older pipeline. The selection logic itself is pipeline-agnostic (it just samples images and expects hand masks), so the staleness is cosmetic, but it should be corrected before this text goes in front of a client.

## UNCERTAINTIES
- The README/docstring reference 'Gemini auto-label quality' as the benchmark target, but project memory says the current labeling pipeline is Grounding DINO + SAM (not Gemini). The selection script is pipeline-agnostic, so this appears to be stale documentation rather than a functional issue, but I could not confirm from code within this subsystem which auto-labeler the IoU>0.70 threshold is currently applied to.
- I did not numerically compute IoU of the auto_annotations masks vs the hand masks, nor did I find a checked-in report that records the realized golden IoU for any production model. The QA thresholds (0.70/0.90) and the per-epoch regression mechanism are documented and wired, but I could not verify an actual achieved golden-set IoU number for the DINOv3 production model from these files.
- Mask pixel encoding mismatch: README says 0/255, on-disk golden masks are 0/1. Consumers expect 0/1 so the pipeline is consistent, but if any external tool or doc assumes 0/255 it would silently break — worth reconciling.
- selection_info.json records command as just 'tools/select_golden_set.py' (no flags), consistent with all-defaults; I could not independently re-run the selector against the current test.jsonl to confirm the recorded source_sha256 still matches the current file (the README itself warns the set is invalidated if the split changed).
- The 100 hand masks' provenance (which annotator, single vs double pass, whether the IAA second-pass described in the README was actually performed) is not recorded in any file I inspected.