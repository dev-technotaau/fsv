"""LoRA fine-tune of Grounding DINO on cedar fence masks.

Trains DINO to better detect "wood fence" prompts by adapting cross-attention
weights via LoRA. Fits in 6GB VRAM thanks to:
  - LoRA adapters (only ~5M trainable params)
  - bf16 mixed precision
  - Batch size 1 + gradient accumulation
  - Adam states only for adapters

Workflow:
  1. Loads training_set masks + manifest
  2. Converts each mask → bounding boxes (one box per fence connected component)
  3. Wraps DINO with LoRA on attention projections
  4. Trains for N epochs with cosine LR schedule
  5. Evaluates on golden_set every epoch (detection recall + mean score)
  6. Saves best adapter weights based on golden recall

Usage:
  python -m annotation.finetune_dino_lora
  python -m annotation.finetune_dino_lora --epochs 50 --lr 5e-5
  python -m annotation.finetune_dino_lora --out-dir dataset/finetune_runs/dino_lora_v2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Reduce CUDA memory fragmentation — must be set BEFORE torch import allocates
# any CUDA memory. Critical for 6GB GPU + broad LoRA configurations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Mask → bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_boxes(mask: np.ndarray, min_area: int = 256) -> list[list[float]]:
    """Convert a binary mask (H, W) of fence pixels into bounding boxes.

    Uses cv2.connectedComponents to extract one box per connected fence region.
    Boxes returned in pixel coords [x1, y1, x2, y2]. Tiny components filtered.
    """
    import cv2
    binary = (mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    boxes = []
    for cid in range(1, n):
        ys, xs = np.where(labels == cid)
        if len(xs) < min_area:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def boxes_to_normalized_cxcywh(boxes: list[list[float]],
                                img_h: int, img_w: int) -> torch.Tensor:
    """xyxy pixel → cxcywh normalized [0, 1]. Returns (N, 4) tensor."""
    out = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        out.append([cx, cy, w, h])
    if not out:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(out, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FenceDataset(Dataset):
    """Manifest+mask dataset for Grounding DINO fine-tuning.

    Each item:
      pixel_values, input_ids, attention_mask, pixel_mask, token_type_ids,
      labels = {class_labels: [N], boxes: [N, 4]}

    Grounding DINO's `class_labels` field expects TOKEN POSITION indices
    (which token in input_ids the box corresponds to), NOT phrase indices.
    For prompt "wood fence ." the tokens are roughly:
        [CLS, "wood", "fence", ".", SEP]  (positions 0, 1, 2, 3, 4)
    A box detecting "wood fence" should point to the first content token
    (position 1 = "wood"). We precompute this position from the tokenizer.
    """
    def __init__(self, manifest_path: Path, masks_dir: Path,
                 processor, prompt: str = "wood fence",
                 keep_empty: bool = True) -> None:
        with manifest_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        # Filter to rows with a corresponding mask file
        kept = []
        skipped_no_mask = 0
        skipped_no_image = 0
        for r in rows:
            mp = masks_dir / f"{r['id']}.png"
            ip = Path(r["path"])
            if not mp.exists():
                skipped_no_mask += 1
                continue
            if not ip.exists():
                skipped_no_image += 1
                continue
            kept.append(r)
        print(f"  loaded {len(kept)} rows  "
              f"(skipped: {skipped_no_mask} no-mask, {skipped_no_image} no-image)")

        # Optionally exclude images with empty masks (no fence)
        if not keep_empty:
            before = len(kept)
            filtered = []
            for r in kept:
                m = np.array(Image.open(masks_dir / f"{r['id']}.png"))
                if (m > 0).sum() > 256:
                    filtered.append(r)
            kept = filtered
            print(f"  filtered empty masks: {before} → {len(kept)}")

        self.rows = kept
        self.masks_dir = Path(masks_dir)
        self.processor = processor
        self.prompt = prompt.strip().lower().rstrip(".") + " ."

        # Precompute the token position of the first CONTENT token in the prompt.
        # For "wood fence ." with BERT tokenizer: [CLS, "wood", "fence", ".", SEP]
        # → first content token is at position 1.
        # All boxes will be labeled with this token position.
        tokenized = processor.tokenizer(self.prompt, add_special_tokens=True,
                                        return_tensors="pt")
        token_strs = processor.tokenizer.convert_ids_to_tokens(
            tokenized["input_ids"][0].tolist()
        )
        # Find first non-special token
        special = set(processor.tokenizer.all_special_tokens)
        self.fence_token_pos = 1  # safe default (after [CLS])
        for i, tok in enumerate(token_strs):
            if tok not in special and tok not in (".", ","):
                self.fence_token_pos = i
                break
        print(f"  prompt tokens: {token_strs}")
        print(f"  fence token position (for class_labels): "
              f"{self.fence_token_pos} ('{token_strs[self.fence_token_pos]}')")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        mask = np.array(Image.open(self.masks_dir / f"{row['id']}.png"))
        H, W = mask.shape

        # Mask → boxes → normalized cxcywh
        pixel_boxes = mask_to_boxes(mask)
        norm_boxes = boxes_to_normalized_cxcywh(pixel_boxes, H, W)

        # Process image + text
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        # Squeeze batch dim from processor output (it added 1)
        item = {k: v[0] for k, v in inputs.items()}

        # class_labels=0 — single-class binary detection. Token-position
        # indexing (class_labels=fence_token_pos) caused CUDA index-out-of-
        # bounds in this model build, so we stick with the safe single-class
        # convention. The loss number will be high (DETR-style unnormalized
        # sum across decoder layers) but training proceeds and recall climbs.
        item["labels"] = {
            "class_labels": torch.zeros(len(norm_boxes), dtype=torch.long),
            "boxes": norm_boxes,
        }
        return item


def collate_fn(batch: list[dict]) -> dict:
    """Pad pixel_values + input_ids; keep labels as list-of-dicts."""
    # Pad pixel_values to max H, W in batch
    max_h = max(b["pixel_values"].shape[1] for b in batch)
    max_w = max(b["pixel_values"].shape[2] for b in batch)
    px_padded = []
    px_mask = []
    for b in batch:
        pv = b["pixel_values"]
        c, h, w = pv.shape
        pad_h, pad_w = max_h - h, max_w - w
        pv_p = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h))
        m = torch.zeros((max_h, max_w), dtype=torch.long)
        m[:h, :w] = 1
        px_padded.append(pv_p)
        px_mask.append(m)

    # Pad input_ids/attention_mask/token_type_ids to max length
    max_len = max(b["input_ids"].shape[0] for b in batch)
    ids = []
    am = []
    tt = []
    for b in batch:
        L = b["input_ids"].shape[0]
        pad = max_len - L
        ids.append(torch.nn.functional.pad(b["input_ids"], (0, pad), value=0))
        am.append(torch.nn.functional.pad(b["attention_mask"], (0, pad), value=0))
        if "token_type_ids" in b:
            tt.append(torch.nn.functional.pad(b["token_type_ids"], (0, pad), value=0))

    out = {
        "pixel_values": torch.stack(px_padded),
        "pixel_mask": torch.stack(px_mask),
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(am),
        "labels": [b["labels"] for b in batch],
    }
    if tt:
        out["token_type_ids"] = torch.stack(tt)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model + LoRA setup
# ─────────────────────────────────────────────────────────────────────────────

def build_model_with_lora(model_name: str, lora_r: int, lora_alpha: int,
                          lora_dropout: float):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(model_name)
    model = GroundingDinoForObjectDetection.from_pretrained(model_name)

    # PEFT 0.12+ compatibility patch: PEFT calls get_input_embeddings() to
    # check for tied weights, but GroundingDinoModel doesn't override it.
    # Patch with text-encoder word embeddings (or empty fallback).
    import torch.nn as _nn
    def _patched_get_input_embeddings(self):
        for attr in ("text_backbone", "text_encoder", "bert", "language_backbone"):
            if hasattr(self, attr):
                try:
                    return getattr(self, attr).get_input_embeddings()
                except Exception:
                    continue
        return _nn.Module()  # empty → tied-weights check sees no params

    base = model.model if hasattr(model, "model") else model
    base.get_input_embeddings = _patched_get_input_embeddings.__get__(base)

    # Target list calibrated to Grounding DINO's actual module naming.
    # `dense` (108 layers) was excluded due to 6GB VRAM constraint —
    # GroundingDINOForObjectDetection doesn't support gradient checkpointing
    # in transformers, so we can't store all those activations. Without dense
    # the model still gets cross-modal + decoder + FFN + deformable adaptation.
    target_modules = [
        # BERT text encoder attention (skipping `dense` to fit 6GB)
        "query", "key", "value",
        # DETR-style decoder
        "out_proj", "fc1", "fc2",
        # Deformable attention (vision encoder feature aggregation)
        "value_proj", "output_proj",
        # Cross-modal fusion projections
        "text_proj", "vision_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None,   # Grounding DINO isn't in PEFT's task registry
    )

    # NOTE: gradient_checkpointing not supported by GroundingDinoForObjectDetection
    # in transformers — that's why we skip the `dense` layers in target_modules
    # to fit memory budget instead.
    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  total params:     {total / 1e6:.1f}M")
    print(f"  trainable (LoRA): {trainable / 1e6:.2f}M  ({100*trainable/total:.2f}%)")

    # Diagnostic: print which module names got LoRA applied
    lora_modules = set()
    for name, _ in model.named_modules():
        if "lora_" in name:
            # Extract the module path before "lora_"
            parts = name.split(".lora_")[0].split(".")
            if parts:
                lora_modules.add(parts[-1])
    print(f"  LoRA attached to module names: {sorted(lora_modules)}")

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Eval on golden set
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, processor, golden_dataset: FenceDataset, device: str,
             score_threshold: float = 0.20) -> dict:
    """Run DINO on golden images. Report:
      - recall: % images where DINO detected ANY box for positive images
      - mean_max_score: average max detection score across positive images
    """
    model.eval()
    n_pos = 0
    n_pos_detected = 0
    score_sum = 0.0
    score_count = 0

    for i in range(len(golden_dataset)):
        row = golden_dataset.rows[i]
        image = Image.open(row["path"]).convert("RGB")
        mask = np.array(Image.open(golden_dataset.masks_dir / f"{row['id']}.png"))
        has_fence = (mask > 0).sum() > 256

        inputs = processor(
            images=image, text=golden_dataset.prompt,
            return_tensors="pt", truncation=True, max_length=256,
        ).to(device)
        outputs = model(**inputs)

        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=score_threshold, text_threshold=0.15,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=score_threshold, text_threshold=0.15,
                target_sizes=[image.size[::-1]],
            )[0]
        scores = results["scores"].detach().cpu().numpy()

        if has_fence:
            n_pos += 1
            if len(scores) > 0:
                n_pos_detected += 1
                score_sum += float(scores.max())
                score_count += 1

    recall = n_pos_detected / max(n_pos, 1)
    mean_max_score = score_sum / max(score_count, 1)
    model.train()
    return {
        "n_positive_images": n_pos,
        "n_detected": n_pos_detected,
        "recall": recall,
        "mean_max_score": mean_max_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Build model + LoRA ───────────────────────────────────────────
    print("\n[1/5] Loading Grounding DINO + LoRA...")
    model, processor = build_model_with_lora(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    model.to(device)

    # ── Datasets ─────────────────────────────────────────────────────
    print("\n[2/5] Loading datasets...")
    print("Training set:")
    train_ds = FenceDataset(
        Path(args.train_manifest), Path(args.train_masks),
        processor, prompt=args.prompt, keep_empty=True,
    )
    print("Eval set (golden):")
    eval_ds = FenceDataset(
        Path(args.eval_manifest), Path(args.eval_masks),
        processor, prompt=args.prompt, keep_empty=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Baseline eval ────────────────────────────────────────────────
    print("\n[3/5] Baseline eval (zero-shot DINO on golden)...")
    baseline = evaluate(model, processor, eval_ds, device)
    print(f"  baseline recall: {baseline['recall']:.1%}")
    print(f"  baseline mean max score: {baseline['mean_max_score']:.3f}")

    # ── Optimizer ────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum + 1) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    log_path.write_text("")  # truncate

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n[4/5] Training {args.epochs} epochs (LR={args.lr}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum})...")
    best_recall = baseline["recall"]
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch_dev = {
                k: (v.to(device) if torch.is_tensor(v) else
                    [{kk: vv.to(device) for kk, vv in d.items()} for d in v])
                for k, v in batch.items()
            }
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(**batch_dev)
                loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += float(loss) * args.grad_accum
            n_batches += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch+1}/{args.epochs}]  "
              f"loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        # Eval on golden
        ev = evaluate(model, processor, eval_ds, device)
        print(f"  golden recall: {ev['recall']:.1%}  "
              f"mean max score: {ev['mean_max_score']:.3f}")

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "eval_recall": ev["recall"],
            "eval_mean_max_score": ev["mean_max_score"],
            "elapsed_s": elapsed,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save best
        if ev["recall"] > best_recall:
            best_recall = ev["recall"]
            best_epoch = epoch + 1
            model.save_pretrained(out_dir / "adapter_best")
            print(f"  ✓ new best — saved to {out_dir / 'adapter_best'}")

        # Always save last
        model.save_pretrained(out_dir / "adapter_last")

        # Free VRAM
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Final report ─────────────────────────────────────────────────
    print(f"\n[5/5] Done.")
    print(f"  baseline recall:   {baseline['recall']:.1%}")
    print(f"  best recall:       {best_recall:.1%}  (epoch {best_epoch})")
    print(f"  improvement:       {(best_recall - baseline['recall']):+.1%}")
    print(f"  best adapter:      {out_dir / 'adapter_best'}")
    print(f"  last adapter:      {out_dir / 'adapter_last'}")
    print(f"  training log:      {log_path}")

    # Save summary
    (out_dir / "summary.json").write_text(json.dumps({
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_recall": best_recall,
        "args": vars(args),
    }, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--train-manifest", default="dataset/training_set/manifest.jsonl")
    ap.add_argument("--train-masks", default="dataset/training_set/masks")
    ap.add_argument("--eval-manifest", default="dataset/golden_set/manifest.jsonl")
    ap.add_argument("--eval-masks", default="dataset/golden_set/masks")
    ap.add_argument("--prompt", default="wood fence",
                    help="Single text prompt used for training + eval")
    ap.add_argument("--out-dir", default="dataset/finetune_runs/dino_lora_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
