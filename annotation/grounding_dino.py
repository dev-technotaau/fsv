"""Grounding DINO wrapper — text-prompted open-vocabulary detection.

Loads HuggingFace-packaged IDEA-Research/grounding-dino model and exposes a
simple `detect(image, class_defs)` API that returns per-class boxes + scores.

Grounding DINO expects prompts joined by ". " separator. We batch multiple
class prompts per inference call for throughput.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from annotation.schema import ClassDef


@dataclass
class Detection:
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]    # absolute pixel coords
    score: float                                     # detection confidence [0, 1]


class GroundingDINODetector:
    """Wrapper around transformers' GroundingDinoForObjectDetection."""

    def __init__(self, model_name: str, device: str | None = None,
                 amp_dtype: str = "none") -> None:
        """
        amp_dtype: "none" | "fp16" | "bf16" — inference mixed-precision mode.
        """
        try:
            import torch
            from transformers import (
                AutoProcessor,
                GroundingDinoForObjectDetection,
            )
        except ImportError as e:
            raise RuntimeError(
                "Grounding DINO requires `transformers>=4.40` and `torch`. "
                "Install: pip install torch transformers Pillow"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.amp_dtype = amp_dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self._torch = torch
        self._amp_dtype_torch = self._resolve_amp_dtype(amp_dtype, torch)

    @staticmethod
    def _resolve_amp_dtype(name: str, torch_module):
        if name == "bf16":
            return torch_module.bfloat16
        if name == "fp16":
            return torch_module.float16
        return None    # fp32 — no autocast

    def _format_prompt(self, phrases: list[str]) -> str:
        """Grounding DINO expects lowercase phrases separated by '. '.
        Each phrase is a targetable class — the model returns boxes labeled
        with the phrase text that best matches."""
        cleaned = []
        for p in phrases:
            p = p.strip().lower().rstrip(".")
            if p:
                cleaned.append(p)
        return " . ".join(cleaned) + " ."

    # Grounding DINO's text-position-embedding is fixed at 256 tokens.
    # Exceeding it raises "size of tensor a (N) must match ..." in post-
    # process. We keep joined phrases under this with headroom.
    _MAX_TEXT_TOKENS = 256
    _TOKEN_HEADROOM = 230

    def _estimate_prompt_tokens(self, prompts: list[str]) -> int:
        """Token count for a list of phrases as they would be joined for DINO."""
        phrases = [p.strip().lower().rstrip(".") for p in prompts if p.strip()]
        if not phrases:
            return 0
        prompt_text = " . ".join(phrases) + " ."
        enc = self.processor.tokenizer(prompt_text, add_special_tokens=True,
                                       return_attention_mask=False)
        return len(enc["input_ids"])

    def _estimate_batch_tokens(self, class_defs_chunk: list[ClassDef]) -> int:
        """Ask the real tokenizer how many tokens this batch's prompt would emit."""
        all_prompts = [p for cls in class_defs_chunk for p in cls.prompts]
        return self._estimate_prompt_tokens(all_prompts)

    def _split_class_prompts(self, cls: ClassDef) -> list[ClassDef]:
        """If a single class has more prompts than fit in the token budget,
        split its prompts into multiple virtual classes (same id/thresholds,
        subset of prompts each). Used when a class's prompt list alone would
        exceed _TOKEN_HEADROOM — otherwise truncation=True drops tail prompts
        and hurts recall for that class."""
        from dataclasses import replace
        sub_classes: list[ClassDef] = []
        current: list[str] = []
        for p in cls.prompts:
            candidate = current + [p]
            if self._estimate_prompt_tokens(candidate) > self._TOKEN_HEADROOM and current:
                sub_classes.append(replace(cls, prompts=tuple(current)))
                current = [p]
            else:
                current = candidate
        if current:
            sub_classes.append(replace(cls, prompts=tuple(current)))
        return sub_classes

    def _split_batches_by_tokens(
        self, class_defs: list[ClassDef], max_per_batch: int,
    ) -> list[list[ClassDef]]:
        """Greedy split: grow a batch until adding the next class would blow
        the token headroom, then start a new batch. If a single class's
        prompt list itself exceeds the headroom, split that class into
        multiple virtual classes (each getting its own batch). Guarantees
        every batch tokenizes under _TOKEN_HEADROOM so no prompt is truncated."""
        # Phase 1: normalize — if any class alone is too long, split it first.
        normalized: list[ClassDef] = []
        for cls in class_defs:
            if self._estimate_prompt_tokens(list(cls.prompts)) > self._TOKEN_HEADROOM:
                normalized.extend(self._split_class_prompts(cls))
            else:
                normalized.append(cls)

        # Phase 2: pack normalized classes into batches.
        batches: list[list[ClassDef]] = []
        current: list[ClassDef] = []
        for cls in normalized:
            candidate = current + [cls]
            if len(candidate) > max_per_batch or (
                self._estimate_batch_tokens(candidate) > self._TOKEN_HEADROOM
                and current
            ):
                batches.append(current)
                current = [cls]
            else:
                current = candidate
        if current:
            batches.append(current)
        return batches

    def detect_multiclass(
        self,
        image: Image.Image,
        class_defs: list[ClassDef],
        batch_size_prompts: int = 10,
    ) -> list[Detection]:
        """Run detection for multiple classes in batched prompt calls.

        Batches class prompts together to reduce GPU round-trips. Each batch
        combines prompts from several classes into a single grounded-dino
        call; the returned phrases are matched back to their source class
        via substring overlap.
        """
        torch = self._torch
        all_detections: list[Detection] = []

        # Token-aware split so no batch ever exceeds the 256-token text cap.
        batches = self._split_batches_by_tokens(class_defs, batch_size_prompts)
        for batch in batches:
            # Collect all phrases from this batch, keeping a phrase→class_id map
            phrases: list[str] = []
            phrase_to_class: dict[str, int] = {}
            per_class_thresholds: dict[int, tuple[float, float]] = {}
            for cls in batch:
                for p in cls.prompts:
                    lp = p.strip().lower().rstrip(".")
                    phrases.append(lp)
                    phrase_to_class[lp] = cls.id
                per_class_thresholds[cls.id] = (cls.box_threshold, cls.text_threshold)

            prompt_text = self._format_prompt(phrases)

            # truncation=True + max_length is a safety net — token-aware
            # splitting above should already keep us under the cap, but this
            # prevents a crash if a single class has unusually long prompts.
            inputs = self.processor(
                images=image, text=prompt_text, return_tensors="pt",
                truncation=True, max_length=self._MAX_TEXT_TOKENS,
            ).to(self.device)

            # Use mixed precision (bf16/fp16) when requested + on CUDA.
            # Falls back to fp32 on CPU or when amp_dtype=none.
            if self._amp_dtype_torch is not None and self.device.startswith("cuda"):
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=self._amp_dtype_torch,
                ):
                    outputs = self.model(**inputs)
            else:
                with torch.inference_mode():
                    outputs = self.model(**inputs)

            # Use the LOWEST box/text threshold across the batch's classes so
            # no class is over-filtered; we'll re-filter per-class below.
            min_box = min(t[0] for t in per_class_thresholds.values())
            min_text = min(t[1] for t in per_class_thresholds.values())

            # HuggingFace postprocessor returns target-sized boxes.
            # Arg name changed across transformers versions: older uses
            # `box_threshold`, newer (>=4.44) uses `threshold`. Try new first,
            # fall back to old so the code works on either.
            pp_kwargs = dict(
                target_sizes=[image.size[::-1]],  # (H, W)
                text_threshold=min_text,
            )
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=min_box, **pp_kwargs,
                )[0]
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=min_box, **pp_kwargs,
                )[0]

            boxes = results["boxes"].detach().cpu().numpy()     # (N, 4) xyxy
            scores = results["scores"].detach().cpu().numpy()   # (N,)
            # Key renamed in newer transformers: "labels" → "text_labels"
            labels = results.get("text_labels", results.get("labels", []))

            for box, score, label in zip(boxes, scores, labels):
                # Map the detected phrase to a class_id
                label_clean = str(label).strip().lower().rstrip(".")
                # First try exact match
                cid = phrase_to_class.get(label_clean)
                if cid is None:
                    # Fallback: best substring overlap
                    best, best_overlap = None, 0
                    for p, c in phrase_to_class.items():
                        # count shared words
                        shared = len(set(p.split()) & set(label_clean.split()))
                        if shared > best_overlap:
                            best, best_overlap = c, shared
                    cid = best
                if cid is None:
                    continue

                # Apply per-class thresholds (stricter re-filter)
                box_thr, _text_thr = per_class_thresholds[cid]
                if float(score) < box_thr:
                    continue

                all_detections.append(Detection(
                    class_id=cid,
                    class_name=next(c.name for c in batch if c.id == cid),
                    box_xyxy=tuple(float(v) for v in box),       # type: ignore[arg-type]
                    score=float(score),
                ))
        return all_detections

    @staticmethod
    def filter_boxes(
        detections: list[Detection],
        max_per_class: int,
    ) -> list[Detection]:
        """Keep at most N highest-confidence boxes per class (prevents prompt
        spam flooding the SAM 2 pipeline with dozens of overlapping boxes)."""
        by_class: dict[int, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)
        out: list[Detection] = []
        for cid, dets in by_class.items():
            dets.sort(key=lambda d: -d.score)
            out.extend(dets[:max_per_class])
        return out

    def detect_multiclass_tta(
        self,
        image: Image.Image,
        class_defs: list[ClassDef],
        batch_size_prompts: int = 10,
        iou_merge_threshold: float = 0.55,
    ) -> list[Detection]:
        """Test-time augmentation: detect on original AND horizontally-flipped
        image, unflip boxes from the flipped pass, merge via cross-pass NMS.
        ~2x the detection compute; typically +3-8% recall on tricky images.
        """
        # Pass 1: original
        dets_orig = self.detect_multiclass(image, class_defs, batch_size_prompts)
        # Pass 2: horizontally flipped
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        dets_flip = self.detect_multiclass(flipped, class_defs, batch_size_prompts)
        # Un-flip boxes: x' = W - x
        W = image.size[0]
        dets_flip_remapped = []
        for d in dets_flip:
            x1, y1, x2, y2 = d.box_xyxy
            new_x1 = W - x2
            new_x2 = W - x1
            dets_flip_remapped.append(Detection(
                class_id=d.class_id, class_name=d.class_name,
                box_xyxy=(float(new_x1), float(y1), float(new_x2), float(y2)),
                score=d.score,
            ))
        merged = dets_orig + dets_flip_remapped
        # Cross-pass NMS within class (same threshold as normal NMS)
        return self.nms_within_class(merged, iou_threshold=iou_merge_threshold)

    @staticmethod
    def fence_first_spatial_filter(
        detections: list[Detection],
        fence_class_ids: set[int],
        suppress_class_ids: set[int] | None = None,
        strong_fence_score: float = 0.30,
        overlap_threshold: float = 0.5,
    ) -> list[Detection]:
        """Two-pass approach: where a STRONG fence is detected, suppress other
        classes (non-human) whose boxes heavily overlap that fence region.
        This stops vegetation_occluder / foreground_object / construction from
        matching fence textures and stealing pixels at detection time.

        `fence_class_ids` — class IDs that count as fences (from schema)
        `suppress_class_ids` — class IDs to drop if they overlap a strong
                                fence. If None, defaults to an empty set
                                (no-op) — caller should pass occluder IDs.
        `strong_fence_score` — min fence detection score to count as "strong"
        `overlap_threshold` — fraction of the non-fence box that must overlap
                              a strong-fence box to be suppressed (IoU-like)
        """
        strong_fences = [d for d in detections
                         if d.class_id in fence_class_ids
                         and d.score >= strong_fence_score]
        if not strong_fences:
            return detections  # no strong fence → nothing to filter

        SUPPRESS_IF_OVERLAPS = frozenset(suppress_class_ids or set())
        if not SUPPRESS_IF_OVERLAPS:
            return detections  # no classes configured to suppress

        kept = []
        for d in detections:
            if d.class_id not in SUPPRESS_IF_OVERLAPS:
                kept.append(d)
                continue
            # Compute max overlap ratio (intersection / d.box area) with any strong fence
            x1, y1, x2, y2 = d.box_xyxy
            d_area = max((x2 - x1) * (y2 - y1), 1e-6)
            max_overlap_ratio = 0.0
            for f in strong_fences:
                fx1, fy1, fx2, fy2 = f.box_xyxy
                ix1, iy1 = max(x1, fx1), max(y1, fy1)
                ix2, iy2 = min(x2, fx2), min(y2, fy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                overlap = (iw * ih) / d_area
                if overlap > max_overlap_ratio:
                    max_overlap_ratio = overlap
            if max_overlap_ratio < overlap_threshold:
                kept.append(d)
            # else: dropped — this box mostly sits inside a strong fence detection
        return kept

    @staticmethod
    def nms_within_class(
        detections: list[Detection],
        iou_threshold: float,
    ) -> list[Detection]:
        """Non-max suppression inside each class to remove duplicate boxes."""
        import numpy as np
        by_class: dict[int, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)

        kept: list[Detection] = []
        for cid, dets in by_class.items():
            if len(dets) <= 1:
                kept.extend(dets)
                continue
            dets.sort(key=lambda d: -d.score)
            boxes = np.array([d.box_xyxy for d in dets], dtype=np.float32)
            suppressed = np.zeros(len(dets), dtype=bool)
            for i in range(len(dets)):
                if suppressed[i]:
                    continue
                kept.append(dets[i])
                for j in range(i + 1, len(dets)):
                    if suppressed[j]:
                        continue
                    xx1 = max(boxes[i, 0], boxes[j, 0])
                    yy1 = max(boxes[i, 1], boxes[j, 1])
                    xx2 = min(boxes[i, 2], boxes[j, 2])
                    yy2 = min(boxes[i, 3], boxes[j, 3])
                    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
                    inter = w * h
                    a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                    a2 = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                    union = a1 + a2 - inter
                    if union > 0 and inter / union > iou_threshold:
                        suppressed[j] = True
        return kept
