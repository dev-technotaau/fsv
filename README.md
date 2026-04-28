# Fence Staining Visualizer — ML Training & Web Deployment

AI-powered fence detection and color staining visualizer for **Ninja Fence Staining**.
End-to-end pipeline covering data acquisition, auto-labeling, training across 5 model
families (UNet++, SegFormer, Mask2Former, SAM, YOLOv8), and browser deployment via
ONNX Runtime Web.

## Quick start

All scripts assume **CWD = project root**.

```bash
# Install deps for your target model
pip install -r requirements/segformerb5.txt          # or unetplusplus, yolo, sam, etc.

# Data pipeline (scrape → auto-label → masks)
python data_pipeline/scrape_fence_images_v4.py
python data_pipeline/auto_label_advanced_v2.py
python data_pipeline/convert_labels_v2.py
python data_pipeline/check_masks_v2.py

# Train a model
python src/training/train_SegFormerB5_PREMIUM.py
python src/training/train_UNetPlusPlus.py
python src/training/train_YOLO.py

# Export to browser ONNX
python src/export/export_UNet++_model_to_reuse.py
python src/export/export_segformer.py

# Run web demo
python web/server/start_web_server.py
# Open http://localhost:8080/web/segformer/index_segformer_web.html

# Inference
python src/inference/predict.py image.jpg
python src/inference/recolor_fence_v2.py image.jpg blue
python src/inference/inference_UNetPlusPlus.py --image image.jpg --visualize
```

## Project layout

```
.
├── src/                          # importable Python source
│   ├── datasets/                 # dataset.py, dataset_v2.py
│   ├── training/                 # all train_*.py, robust_train*.py
│   ├── inference/                # predict, recolor, inference_*
│   │   └── legacy_training_copies/   # older training/-dir copies (untouched logic)
│   ├── export/                   # export_*.py (ONNX)
│   └── utils/                    # utils.py
│
├── data_pipeline/                # scrape → auto_label → convert → check
├── tests/                        # verify_*, validate_*, test_*, check_*
│   └── reports/                  # *.json verification reports
│
├── requirements/                 # requirements/<model>.txt (renamed from requirements_<model>.txt)
├── scripts/setup/                # setup_*.ps1 — Windows dev environment bootstrap
├── quick_start/                  # quick_start_yolo.py
│
├── models/
│   ├── pytorch/                  # .pth checkpoints (best_fence_model_*, last_fence_model_*)
│   └── onnx/                     # browser-ready .onnx (fence_model_*_browser.onnx)
│       └── legacy_training_copies/
│
├── web/                          # dev/exploration frontends (browser-side ONNX inference)
│   ├── unet/                     # index.html (UNet)
│   ├── unet_plusplus/            # index_unet_plusplus.html + _web.html
│   ├── segformer/                # index_segformer.html + index_segformer_web.html ⭐
│   ├── legacy_training_copies/   # duplicates from old training/ dir
│   ├── server/                   # start_web_server.py (serves from project root)
│   └── assets/                   # ninja_logo.png, ninja_logo_light.png.webp
│
├── docs/
│   ├── training_guides/          # 10 per-model training guides
│   ├── comparisons/              # 4 model-comparison docs
│   ├── issues_and_fixes/         # 4 issue-log docs (incl. CRITICAL_ISSUES_FOUND)
│   ├── verification/             # 8 verification reports
│   ├── roadmap/                  # FUTURE_DATA_ENHANCEMENTS
│   └── misc/                     # LOGO_SETUP
│
├── data/                         # images/, masks/, no_fence_images/, mask_overlays/
├── checkpoints/                  # training outputs (per-model subdirs)
├── logs/                         # training logs + TensorBoard
├── training_visualizations/      # per-epoch prediction images
├── onnx_models/                  # segformer_fence_detector.onnx (in-training export)
│
├── fence-staining-visualizer/    # ⭐ PRODUCTION DELIVERABLE (standalone git repo, untouched)
│                                 #   Single-page brand-themed visualizer; loads
│                                 #   fence_model_unet_browser.onnx locally.
│
└── training/                     # now empty — kept per "nothing removed" rule
```

## Authoritative model verdict

Per `docs/comparisons/YOLO_vs_MASK2FORMER_vs_SAM.md`, **YOLOv8x-seg** is the
recommended production model (60-90 FPS inference, 30% faster training, 20-25% better
accuracy, 60% cheaper operating cost).

**Currently deployed in `fence-staining-visualizer/`**: `fence_model_unet_browser.onnx`
(derived from UNet++ B7 training). YOLO deployment not yet wired to the browser.

## Model families trained

| Family | Backbone | Params | Res | Trainer |
|--------|----------|--------|-----|---------|
| UNet++ | EfficientNet-B7 | 66M | 512 | `src/training/train_UNetPlusPlus.py` |
| SegFormer B5 | MiT-B5 | 84M | 640 | `src/training/train_SegFormerB5_PREMIUM.py` |
| Mask2Former | SegFormer-B5 | 98.6M | 384/512 | `src/training/train_Mask2Former{,_Detectron2}.py` |
| SAM | ViT-B | 93M | 512 | `src/training/train_SAM.py` |
| YOLOv8x-seg | CSPDarknet53 | 71.8M | 640 | `src/training/train_YOLO.py` |

## Dataset

- `data/images/` (mixed `.jpg/.jpeg/.png` + 803 LabelMe `.json` — JSONs filtered by all training scripts)
- `data/masks/` (binary PNG, 0=background, 255=fence; `(mask > 127).astype(uint8)` is canonical conversion)
- Severe class imbalance: ~2.65% fence pixels
- Train/val split: 85/15 (680/120)

## Environment

- Windows 11 primary dev (WSL Ubuntu paths), shell: bash
- Primary GPU: RTX 3060 6GB (hyperparams tuned for this)
- Colab T4 16GB variant: `src/training/train_SegFormerB5_PREMIUM_googleColab.py`
- Python 3.8+, PyTorch 2.0+, CUDA 11.8

## Setup (Windows PowerShell)

```powershell
# From project root
.\scripts\setup\setup_unetplusplus.ps1           # UNet++
.\scripts\setup\setup_mask2former.ps1            # Mask2Former standalone
.\scripts\setup\setup_mask2former_detectron2.ps1 # Mask2Former enterprise (Detectron2 + DDP)
.\scripts\setup\setup_sam.ps1                    # SAM
```

## Critical notes

- **Existing Mask2Former checkpoint is CORRUPT** (per `docs/issues_and_fixes/CRITICAL_ISSUES_FOUND.md`).
  Four bugs now fixed (EMA save order, hard-negative-mining dim, duplicate focal loss,
  duplicate logging). **Retraining from scratch required.**
- **EMA save convention**: apply → save → restore (never save before applying — old bug).
- **Class weights are intentionally extreme** (e.g. `[0.05, 20.0]` for Detectron2 M2F) due to 2.65% fence imbalance. Don't "normalize" these.
- **`fence-staining-visualizer/` has its own `.git`** — treat as separate repo; don't restructure.

## See also

- `docs/training_guides/` — per-model training guides
- `docs/comparisons/` — decision rationale for model choice
- `docs/issues_and_fixes/` — what went wrong, what got fixed
- `docs/verification/` — authoritative verification reports
