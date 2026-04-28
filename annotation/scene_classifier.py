"""Zero-shot scene type classifier — filters out-of-distribution images.

The golden set sometimes contains non-fence images (interior rooms, document
covers, abstract texture close-ups) that mis-label as fences downstream.
This module uses CLIP to flag those before expensive DINO/SAM inference.

Usage:
    clf = SceneClassifier()
    kind, score = clf.classify(image)
    if kind != "outdoor_fence":
        # skip / flag this image
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


# Candidate scene types. The first is what we want; others are OOD flags.
SCENE_PROMPTS = [
    ("outdoor_fence",
     "an outdoor photo of a fence or wall in a yard or landscape"),
    ("interior_room",
     "an interior photo of a room inside a building"),
    ("document_or_poster",
     "a scan of a document or a text poster or a book cover"),
    ("abstract_closeup",
     "an extreme close-up of a single object or texture with no scene"),
    ("product_shot",
     "a product photograph on a white studio background"),
]


@dataclass
class SceneResult:
    kind: str        # the matched scene key
    score: float     # softmax probability for that key
    is_ood: bool     # True if kind != "outdoor_fence"


class SceneClassifier:
    """Zero-shot CLIP classifier. Loads lazily on first classify() call."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._text_embeds = None  # precomputed

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise RuntimeError(
                "Scene classifier requires `transformers` (already a dep)."
            ) from e
        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        resolved = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(resolved).eval()
        self._device = resolved
        # Precompute text embeddings once (tiny)
        prompts = [p for _, p in SCENE_PROMPTS]
        with torch.inference_mode():
            inputs = self._processor(text=prompts, return_tensors="pt",
                                     padding=True).to(resolved)
            text_feats = self._model.get_text_features(**inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self._text_embeds = text_feats  # (K, D)

    def classify(self, image: Image.Image) -> SceneResult:
        """Return the most-probable scene type + its softmax probability."""
        self._load()
        torch = self._torch
        with torch.inference_mode():
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            img_feat = self._model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            # Cosine similarity → softmax
            logits = (img_feat @ self._text_embeds.T) * 100.0  # CLIP temperature
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        best_idx = int(np.argmax(probs))
        kind = SCENE_PROMPTS[best_idx][0]
        score = float(probs[best_idx])
        return SceneResult(
            kind=kind, score=score,
            is_ood=(kind != "outdoor_fence"),
        )
