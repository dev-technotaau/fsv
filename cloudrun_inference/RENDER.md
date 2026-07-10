# /render — Qwen renovation on the existing DINOv3 service

`/render` was added **next to** the unchanged `/detect`. Same L4, same container, one instance.
`/detect` is byte-for-byte the same; Qwen lazy-loads on the first `/render`, so `/detect` keeps
working even if the Qwen half has a problem.

```
POST /render   multipart: image, colorHex, family=general|semi-transparent|semi-solid,
                          tone (optional), seed (optional), mask (optional — reuses /detect's)
  -> image/jpeg   headers: X-DeltaE, X-Seg-Ms, X-Render-Ms, X-Total-Ms
```
Inside: segment (reuses the loaded DINOv3, or the supplied mask) → Qwen renovate (GPU) →
exact-swatch color-lock + composite (CPU). No CPU segmentation anywhere.

## Architecture (weights are BAKED — no bucket)
The **pre-quantized 4-bit checkpoint** `ovedrive/Qwen-Image-Edit-2509-4bit` (~12–15 GB, quality
layers kept) + the Lightning LoRA are **baked into the image** at `/model/qwen-edit` and
`/model/qwen-lightning`, exactly like the DINOv3 ONNX. They load from **local disk (~1–2 min)** —
no GCS bucket, no gcsfuse mount, no runtime quantization. (The full 40 GB bf16 checkpoint stalled
the Docker layer-commit; the ~12 GB pre-quantized one bakes fine.)

## Deploy
```powershell
cd cloudrun_inference
.\deploy.ps1            # Windows;  bash deploy.sh on Linux
```
Set `$env:GCP_PROJECT` first. Defaults to the safe **fsv-dinov3-v2** service (live `fsv-dinov3`
untouched); set `FSV_SERVICE=fsv-dinov3` to cut over. First build bakes ~12 GB (~15–20 min);
deploys as a new revision (instant rollback below).

## Test
```powershell
$URL = gcloud run services describe fsv-dinov3-v2 --region us-central1 --format='value(status.url)'
curl.exe -F image=@fence.jpg -F colorHex=#7d4f28 -F family=general "$URL/render" -o out.jpg -D -
```
Cold start: ~1–2 min load + ~2 min render. Warm: ~2 min. `/detect` responds normally throughout.

## Env knobs
| var | default | note |
|---|---|---|
| `QWEN_QUANT` | `prequant` | checkpoint is already NF4; `4bit`=runtime-quantize a base model; `none`=bf16 |
| `QWEN_STEPS` | `8` (Lightning) / `20` (base) | auto-falls back to 20 if the LoRA doesn't apply |
| `QWEN_CFG` | `1.0` (Lightning) / `4.0` (base) | |
| `QWEN_LIGHTNING_LORA` | `/model/qwen-lightning` | `""` disables → 20-step base |
| `FSV_WORKING_RES` | `1024` | 768 = faster/softer |

## Config lessons baked in (don't regress these)
- **transformers `<5`** — 5.x breaks diffusers-0.39's Qwen2.5-VL forward → first-inference 500.
- **`torchvision`** required (Qwen2.5-VL processor imports it).
- **Single CUDA stack** — dropped the `nvidia-cudnn-cu12==9.10` pin so torch's cuDNN 9.1 serves
  both torch and onnxruntime-gpu. Startup log `onnxruntime available providers` must list `CUDAExecutionProvider`.
- **Never `fuse_lora`** on 4-bit (lossy, PEFT #2321) — load as a live adapter.
- **`enable_model_cpu_offload`**, never `.to('cuda')`, for a 4-bit pipeline.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the load-time allocator.

## Rollback
```bash
gcloud run revisions list --service fsv-dinov3-v2 --region us-central1
gcloud run services update-traffic fsv-dinov3-v2 --region us-central1 --to-revisions <PREV>=100
```

## Status
CPU finisher verified locally (dE=0, clean composite). Segmentation + CUDA coexistence confirmed
working on Cloud Run. The Qwen render + Lightning proves out on the baked deploy.
