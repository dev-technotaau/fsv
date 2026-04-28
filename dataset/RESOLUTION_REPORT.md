# Dataset Resolution Report

_Generated: 2026-04-17T08:08:24+00:00_

**Total images**: 33,423  
**HQ threshold**: shorter-edge ≥ 1024px

## Tier distribution

| Tier | Shorter edge | Count | % | Pos | Neg |
|------|--------------|-------|---|-----|-----|
| **ULTRA** | >= 2048 | 1,286 | 3.8% | 721 | 565 |
| **HD** | 1536-2047 | 6,260 | 18.7% | 3,269 | 2,991 |
| **STANDARD** | 1024-1535 | 13,154 | 39.4% | 9,987 | 3,167 |
| **LOW** | 800-1023 | 12,723 | 38.1% | 7,437 | 5,286 |

## Cumulative — images retained at each threshold

| Shorter edge ≥ | Count | % of total | Pos | Neg |
|----------------|-------|-----------|-----|-----|
| 1024px | 20,700 | 61.9% | 13,977 | 6,723 |
| 1280px | 11,542 | 34.5% | 7,326 | 4,216 |
| 1536px | 7,546 | 22.6% | 3,990 | 3,556 |
| 1792px | 2,412 | 7.2% | 1,636 | 776 |
| 2048px | 1,286 | 3.9% | 721 | 565 |
| 2560px | 768 | 2.3% | 418 | 350 |
| 3072px | 336 | 1.0% | 144 | 192 |

## HQ subset (for Phase 2 fine-tune)

- **Threshold**: shorter-edge ≥ 1024px
- **HQ total**: 20,700 images (61.9% of full set)
  - pos: 13,977
  - neg: 6,723
- **HQ manifest**: `dataset/manifest_hq.jsonl`
- **HQ splits**: `dataset/splits/{train,val,test}_hq.jsonl`

## Training strategy (Option C)

1. **Phase 1 (pretrain)**: train at 512×512 on all 33,423 images using `manifest.jsonl` + `splits/{train,val}.jsonl`
2. **Phase 2 (finetune)**: fine-tune at 1024×1024 on HQ subset (20,700 images) using `manifest_hq.jsonl` + `splits/{train,val}_hq.jsonl`
3. **Final eval**: run test_hq.jsonl at 1024 input for deployment sign-off metric

No upscaling is performed — low-resolution images contribute at their native size to Phase 1 and are simply excluded from Phase 2.