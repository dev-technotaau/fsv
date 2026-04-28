"""End-to-end annotation orchestrator.

Runs: image -> Grounding DINO detect -> SAM 2 segment -> fuse -> score -> save.
Handles batching, resume, error recovery, and per-image provenance logging.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from annotation.fusion import fuse_masks
from annotation.grounding_dino import Detection, GroundingDINODetector
from annotation.masks import (
    save_class_mask,
    save_class_mask_preview,
    save_colorized_viz,
    save_confidence_heatmap,
)
from annotation.qa import QAScore, score_annotation
from annotation.sam2 import InstanceMask, SAM2Segmenter
from annotation.schema import Schema


@dataclass
class AnnotationResult:
    image_id: str
    image_path: str
    mask_path: str
    viz_path: str | None
    heatmap_path: str | None
    n_detections: int
    n_classes_present: int
    overall_confidence: float
    fence_wood_coverage: float
    fence_wood_confidence: float | None
    flags: list[str]
    needs_review: bool
    per_class_pixel_counts: dict[int, int]
    instance_detections: list[dict]        # for audit / debugging
    elapsed_s: float


class AnnotationPipeline:
    """Annotate one image or a whole manifest.

    Usage:
        schema = load_schema("configs/annotation_schema.yaml")
        pipe = AnnotationPipeline(schema)
        pipe.warm_up()
        result = pipe.annotate_one(Path("img.jpg"), image_id="xyz", image_class="pos")
    """

    def __init__(self, schema: Schema, device: str | None = None,
                 save_viz: bool = True, save_heatmap: bool = False,
                 amp_dtype: str = "none",
                 low_vram: bool = False,
                 two_pass_fence: bool = True,
                 tta: bool = False,
                 scene_filter: bool = True) -> None:
        """
        amp_dtype: "none" | "fp16" | "bf16" — mixed-precision mode for inference.
            Default "none" = full fp32. bf16 gives ~2× speed on Ampere+ GPUs
            with negligible accuracy drop. fp16 for older GPUs.
            THIS IS THE ONLY QUALITY TRADE-OFF KNOB.
        low_vram: if True, enable pure-enhancement memory optimizations:
            (1) memory-efficient SDPA kernel (mathematically identical output),
            (2) explicit cleanup between images (prevents fragmentation).
            Zero impact on mask output, costs ~5-15 ms per image.
        """
        self.schema = schema
        self.device = device
        self.save_viz = save_viz
        self.save_heatmap = save_heatmap
        self.amp_dtype = amp_dtype
        self.low_vram = low_vram
        self.two_pass_fence = two_pass_fence
        self.tta = tta
        self.scene_filter = scene_filter
        self._detector: GroundingDINODetector | None = None
        self._segmenter: SAM2Segmenter | None = None
        self._scene_classifier = None   # lazy-loaded when scene_filter enabled
        if low_vram:
            self._enable_memory_efficient_attention()

    @staticmethod
    def _enable_memory_efficient_attention() -> None:
        """Enable all three SDPA backends so PyTorch picks the one that fits.
        Each backend produces mathematically equivalent results — this is not
        a quality trade-off, just a memory/speed trade-off that PyTorch
        resolves per-tensor at runtime."""
        try:
            import torch
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    # ── Lazy model loaders (let CLI show progress before slow load) ───

    def warm_up(self) -> None:
        """Force-load models now so errors surface before processing images.
        Also prints a GPU / CUDA diagnostic so you know immediately whether
        you're about to run at GPU or CPU speed."""
        self._print_compute_diagnostic()
        _ = self.detector
        _ = self.segmenter

    def _print_compute_diagnostic(self) -> None:
        """Print what device + precision we're running on. Makes silent-CPU
        fallback impossible to miss."""
        try:
            import torch
        except ImportError:
            print("[compute] torch not installed — annotation will fail to load")
            return

        cuda_ok = torch.cuda.is_available()
        resolved = self.device or ("cuda" if cuda_ok else "cpu")
        print(f"[compute] torch={torch.__version__}")
        print(f"[compute] CUDA available: {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"[compute]   GPU {i}: {name}  ({vram_gb:.1f} GB)  "
                      f"compute={props.major}.{props.minor}")
            print(f"[compute] resolved device: {resolved}")
            print(f"[compute] AMP dtype: {self.amp_dtype}")
            if self.amp_dtype == "bf16" and props.major < 8:
                print(f"[compute] WARN: bf16 requires compute capability >= 8.0 "
                      f"(Ampere+). Your GPU is {props.major}.{props.minor} — "
                      f"falling back to fp16 is recommended.")
        else:
            print(f"[compute] resolved device: {resolved}  <<  NO CUDA AVAILABLE")
            print(f"[compute] This will run on CPU (10-30x slower than GPU).")
            print(f"[compute] To enable CUDA:")
            print(f"[compute]   1. Check NVIDIA driver:  nvidia-smi")
            print(f"[compute]   2. Install CUDA PyTorch:")
            print(f"[compute]      pip install torch torchvision --index-url "
                  f"https://download.pytorch.org/whl/cu121")
            print(f"[compute]      (replace cu121 with your CUDA version: "
                  f"cu118 / cu121 / cu124)")

    @property
    def scene_classifier(self):
        """Lazy-load the CLIP scene classifier."""
        if self._scene_classifier is None:
            from annotation.scene_classifier import SceneClassifier
            self._scene_classifier = SceneClassifier(device=self.device)
        return self._scene_classifier

    @property
    def detector(self) -> GroundingDINODetector:
        if self._detector is None:
            self._detector = GroundingDINODetector(
                model_name=self.schema.pipeline.grounding_dino_model,
                device=self.device,
                amp_dtype=self.amp_dtype,
            )
        return self._detector

    @property
    def segmenter(self) -> SAM2Segmenter:
        if self._segmenter is None:
            self._segmenter = SAM2Segmenter(
                model_name=self.schema.pipeline.sam2_model,
                device=self.device,
                amp_dtype=self.amp_dtype,
            )
        return self._segmenter

    # ── Core single-image annotation ─────────────────────────────────

    def annotate_one(
        self,
        image_path: Path,
        image_id: str,
        image_class: str,
        mask_out_path: Path,
        viz_out_path: Path | None = None,
        heatmap_out_path: Path | None = None,
        sample_for_qa: bool = False,
        prefetched_image: Image.Image | None = None,
    ) -> AnnotationResult:
        t0 = time.time()
        # Use a pre-loaded image if the prefetcher already opened it from disk,
        # otherwise read from disk now. Prefetching saves the ~0.3-0.5s/image
        # I/O wait that would otherwise leave the GPU idle.
        image = prefetched_image if prefetched_image is not None else Image.open(image_path).convert("RGB")
        W, H = image.size

        # ── 0. Scene-type pre-filter (CLIP) ──────────────────────────
        # Flag out-of-distribution images (interior rooms, documents, abstract
        # close-ups) so downstream QA can prioritize them. Doesn't skip — still
        # produces a mask — but the result gets a flag and dampened confidence.
        scene_ood = False
        scene_kind = None
        if self.scene_filter:
            try:
                sr = self.scene_classifier.classify(image)
                scene_kind = sr.kind
                scene_ood = sr.is_ood
            except Exception:
                pass   # classifier load failure shouldn't break the run

        # ── 1. Detect all classes via Grounding DINO ─────────────────
        def _run_detection(class_defs_to_use):
            if self.tta:
                return self.detector.detect_multiclass_tta(
                    image=image,
                    class_defs=class_defs_to_use,
                    batch_size_prompts=self.schema.pipeline.grounding_dino_batch_prompts,
                    iou_merge_threshold=self.schema.pipeline.iou_merge_threshold,
                )
            return self.detector.detect_multiclass(
                image=image,
                class_defs=class_defs_to_use,
                batch_size_prompts=self.schema.pipeline.grounding_dino_batch_prompts,
            )

        detections: list[Detection] = _run_detection(list(self.schema.classes))

        # Fallback: if NO detections at normal thresholds, retry with
        # thresholds scaled down 40% to recover edge-case images where DINO
        # barely missed. Flagged separately so QA knows to prioritize review.
        fallback_used = False
        if len(detections) == 0:
            from dataclasses import replace
            relaxed_classes = [
                replace(c,
                        box_threshold=max(0.10, c.box_threshold * 0.6),
                        text_threshold=max(0.08, c.text_threshold * 0.6))
                for c in self.schema.classes
            ]
            detections = _run_detection(relaxed_classes)
            fallback_used = len(detections) > 0

        # ── 2. Filter + NMS ───────────────────────────────────────────
        detections = self.detector.nms_within_class(
            detections, iou_threshold=self.schema.pipeline.iou_merge_threshold,
        )
        detections = self.detector.filter_boxes(
            detections,
            max_per_class=self.schema.pipeline.max_boxes_per_class_per_image,
        )
        # Two-pass spatial filter: suppress aggressive-occluder boxes that
        # heavily overlap strong fence detections. Saves SAM 2 compute AND
        # prevents these classes from claiming fence pixels even before fusion.
        # Derives IDs from schema "tier" field.
        # IMPORTANT: only "occluder" tier is suppressed, NOT "distractor".
        # Distractors include "absorber" classes (like not_target in the 3-class
        # schema) that are DESIGNED to win over fence via priority. Suppressing
        # them would break that intentional routing.
        if self.two_pass_fence:
            fence_ids = {c.id for c in self.schema.classes if c.tier == "fence"}
            suppress_ids = {
                c.id for c in self.schema.classes
                if c.tier == "occluder"
                and c.name != "human_animal"
            }
            detections = self.detector.fence_first_spatial_filter(
                detections,
                fence_class_ids=fence_ids,
                suppress_class_ids=suppress_ids,
            )

        # ── 3. Run SAM 2 to get masks ────────────────────────────────
        instance_masks: list[InstanceMask] = []
        if detections:
            boxes = np.array([d.box_xyxy for d in detections], dtype=np.float32)
            min_area = max(
                self.schema.pipeline.min_mask_area_pixels,
                int(self.schema.pipeline.min_mask_area_ratio * H * W),
            )
            instance_masks = self.segmenter.segment_boxes(
                image=image,
                boxes_xyxy=boxes,
                class_ids=[d.class_id for d in detections],
                class_names=[d.class_name for d in detections],
                detection_scores=[d.score for d in detections],
                multimask_output=self.schema.pipeline.sam2_multimask_output,
                min_mask_area=min_area,
            )

        # ── 4. Fuse into multi-class map ─────────────────────────────
        class_map, conf_map = fuse_masks(
            instance_masks=instance_masks,
            image_hw=(H, W),
            schema=self.schema,
            strategy=self.schema.pipeline.overlap_priority_strategy,
        )

        # ── 5. Save outputs ──────────────────────────────────────────
        save_class_mask(class_map, mask_out_path)
        # Also save a human-viewable B/W preview where ONLY stain-target
        # pixels are white. Absorber pixels (not_target) and background both
        # render as black — matching how the stainer reads the mask.
        preview_path = mask_out_path.parent.parent / "masks_preview" / mask_out_path.name
        stain_ids = {c.id for c in self.schema.classes if c.is_staining_target}
        save_class_mask_preview(class_map, preview_path, stain_class_ids=stain_ids)
        if self.save_viz and viz_out_path is not None:
            save_colorized_viz(class_map, self.schema, viz_out_path, image=image)
        if self.save_heatmap and heatmap_out_path is not None:
            save_confidence_heatmap(conf_map, heatmap_out_path)

        # ── 6. Score + QA ─────────────────────────────────────────────
        qa = score_annotation(
            image_id=image_id, image_class=image_class,
            instance_masks=instance_masks, class_map=class_map,
            schema=self.schema, sample_for_qa=sample_for_qa,
        )
        if scene_ood:
            qa.flags.append(f"scene_ood={scene_kind}")
            qa.needs_review = True
            qa.overall_confidence *= 0.5   # heavy penalty for OOD scenes
        if fallback_used:
            qa.flags.append("fallback_detection_used")
            qa.needs_review = True
            qa.overall_confidence *= 0.7   # moderate penalty for fallback

        # ── 7. Build result row ──────────────────────────────────────
        classes_present = sum(
            1 for cid in qa.per_class_counts if cid > 0 and qa.per_class_counts[cid] > 0
        )
        return AnnotationResult(
            image_id=image_id,
            image_path=str(image_path),
            mask_path=str(mask_out_path),
            viz_path=str(viz_out_path) if viz_out_path else None,
            heatmap_path=str(heatmap_out_path) if heatmap_out_path else None,
            n_detections=len(instance_masks),
            n_classes_present=classes_present,
            overall_confidence=qa.overall_confidence,
            fence_wood_coverage=qa.fence_wood_coverage,
            fence_wood_confidence=qa.fence_wood_confidence,
            flags=qa.flags,
            needs_review=qa.needs_review,
            per_class_pixel_counts=qa.per_class_counts,
            instance_detections=[
                {
                    "class_id": im.class_id,
                    "class_name": im.class_name,
                    "box_xyxy": list(im.box_xyxy),
                    "detection_score": im.detection_score,
                    "sam_score": im.sam_score,
                    "area_pixels": im.area_pixels,
                }
                for im in instance_masks
            ],
            elapsed_s=round(time.time() - t0, 3),
        )

    # ── Image prefetcher ─────────────────────────────────────────────

    @staticmethod
    def _prefetch_iterator(manifest_rows: list[dict], done_ids: set[str],
                           prefetch_size: int = 4):
        """Yield (i, row, image_or_None, error_or_None) tuples.

        A background thread reads images from disk into memory while the GPU
        is busy on the previous one. Bounded queue (default 4) caps RAM use.
        Done-IDs are skipped at the producer side so we don't waste I/O on them.

        Returns image as PIL.Image (None if file_not_found or load_error,
        with error message in the error slot).
        """
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=prefetch_size)
        sentinel = object()

        def producer():
            for i, row in enumerate(manifest_rows):
                image_id = row.get("id", "")
                if image_id in done_ids:
                    continue
                image_path = Path(row["path"])
                if not image_path.exists():
                    q.put((i, row, None, "file_not_found"))
                    continue
                try:
                    img = Image.open(image_path).convert("RGB")
                    q.put((i, row, img, None))
                except Exception as e:
                    q.put((i, row, None,
                           f"load_error: {type(e).__name__}: {str(e)[:120]}"))
            q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                return
            yield item

    # ── Batch manifest processing ────────────────────────────────────

    def annotate_manifest(
        self,
        manifest_rows: list[dict],
        mask_dir: Path,
        viz_dir: Path | None,
        heatmap_dir: Path | None,
        results_jsonl: Path,
        qa_queue_jsonl: Path,
        resume: bool = True,
        qa_sample_rate: float | None = None,
        qa_seed: int = 42,
        progress_callback=None,
        retry_missing: bool = False,
    ) -> tuple[int, int]:
        """Annotate every row in the manifest. Returns (n_processed, n_flagged)."""
        sample_rate = (qa_sample_rate
                       if qa_sample_rate is not None
                       else self.schema.qa.sample_rate_for_qa)
        rng = random.Random(qa_seed)

        # Resume support — skip IDs already in results file
        done_ids: set[str] = set()
        if resume and results_jsonl.exists():
            with results_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done_ids.add(r["image_id"])
                    except Exception:
                        pass

        mask_dir.mkdir(parents=True, exist_ok=True)

        # --retry-missing: drop "done" IDs whose mask file isn't actually on
        # disk. Catches images that errored (logged in results.jsonl with no
        # mask saved) or whose async save tasks didn't complete before exit.
        if retry_missing and done_ids:
            actual_files = {p.stem for p in mask_dir.glob("*.png")}
            before = len(done_ids)
            ghosts = done_ids - actual_files
            done_ids = done_ids & actual_files
            n_missing = before - len(done_ids)
            print(f"[retry-missing] {len(actual_files)} mask files on disk, "
                  f"{before} 'done' in results.jsonl, "
                  f"{n_missing} have no mask file → will reprocess")
            if ghosts and n_missing <= 20:
                print(f"  ghosts: {sorted(ghosts)[:20]}")
        if viz_dir: viz_dir.mkdir(parents=True, exist_ok=True)
        if heatmap_dir: heatmap_dir.mkdir(parents=True, exist_ok=True)
        results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        qa_queue_jsonl.parent.mkdir(parents=True, exist_ok=True)

        n_processed = 0
        n_flagged = 0
        with results_jsonl.open("a", encoding="utf-8") as fout, \
             qa_queue_jsonl.open("a", encoding="utf-8") as fqa:

            # Prefetch images in a background thread so disk I/O overlaps with
            # GPU inference on the previous image. Bounded queue caps RAM at
            # ~4 × image_size (typically 20-40 MB).
            iterator = self._prefetch_iterator(manifest_rows, done_ids,
                                                prefetch_size=4)

            for i, row, prefetched_image, prefetch_err in iterator:
                image_id = row["id"]
                image_path = Path(row["path"])

                if prefetch_err == "file_not_found":
                    fout.write(json.dumps({
                        "image_id": image_id, "error": "file_not_found",
                        "path": str(image_path),
                    }) + "\n")
                    continue
                if prefetch_err is not None:
                    fout.write(json.dumps({
                        "image_id": image_id, "error": prefetch_err,
                        "path": str(image_path),
                    }) + "\n")
                    continue

                mask_out = mask_dir / f"{image_id}.png"
                viz_out = (viz_dir / f"{image_id}.png") if viz_dir else None
                heat_out = (heatmap_dir / f"{image_id}.png") if heatmap_dir else None
                sample_this = rng.random() < sample_rate

                result = None
                err = None
                for attempt in range(2):
                    try:
                        result = self.annotate_one(
                            image_path=image_path,
                            image_id=image_id,
                            image_class=row.get("class", "unknown"),
                            mask_out_path=mask_out,
                            viz_out_path=viz_out,
                            heatmap_out_path=heat_out,
                            sample_for_qa=sample_this,
                            prefetched_image=prefetched_image,
                        )
                        break
                    except Exception as e:
                        msg = str(e).lower()
                        is_oom = (
                            "out of memory" in msg
                            or "cuda oom" in msg
                            or type(e).__name__ == "OutOfMemoryError"
                        )
                        if is_oom and attempt == 0:
                            # Free everything and retry once with a clean slate.
                            self._free_cuda_memory()
                            continue
                        err = f"{type(e).__name__}:{str(e)[:200]}"
                        break

                if result is None:
                    fout.write(json.dumps({
                        "image_id": image_id,
                        "error": err or "unknown",
                        "path": str(image_path),
                    }) + "\n")
                    fout.flush()
                    if progress_callback:
                        progress_callback(i + 1, len(manifest_rows),
                                          image_id, error=err)
                    if self.low_vram:
                        self._free_cuda_memory()
                    continue

                fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                fout.flush()

                if result.needs_review:
                    n_flagged += 1
                    fqa.write(json.dumps({
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_out),
                        "viz_path": str(viz_out) if viz_out else None,
                        "overall_confidence": result.overall_confidence,
                        "fence_wood_coverage": result.fence_wood_coverage,
                        "fence_wood_confidence": result.fence_wood_confidence,
                        "flags": result.flags,
                        "image_class": row.get("class"),
                        "subcategory": row.get("subcategory"),
                    }, ensure_ascii=False) + "\n")
                    fqa.flush()

                n_processed += 1
                if progress_callback:
                    progress_callback(i + 1, len(manifest_rows),
                                      image_id, result=result)

                if self.low_vram:
                    self._free_cuda_memory()

        return n_processed, n_flagged

    @staticmethod
    def _free_cuda_memory() -> None:
        """Release cached CUDA memory + run Python GC. Prevents fragmentation
        OOM on long runs. Mathematically a no-op for model outputs."""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
