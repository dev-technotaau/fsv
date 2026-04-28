# Fence Segmentation — Full Training Guide

End-to-end commands from a fresh setup through to a deployable ONNX model.
Targeted at: **Windows 11 + RTX 3090 24 GB + Anaconda Python 3.11**.

---

## 0. Prerequisites (verify these are installed)

| Requirement | Check command | Notes |
|---|---|---|
| NVIDIA driver supporting CUDA 12.6+ | `nvidia-smi` | Driver >= 550 |
| Anaconda or Miniconda | `conda --version` | |
| Git for Windows | `git --version` | |
| Python 3.11 (via conda) | `conda search python` | 3.12 also fine |
| Visual Studio 2022 Build Tools (C++) | `where cl` (only inside the VS Native Tools cmd) | Needed for `pydensecrf` |
| ~200 GB free disk on training drive | | Checkpoints + logs grow large |

---

## 1. One-Time Setup (~30 min)

### 1.1 Open the right shell

Use **"x64 Native Tools Command Prompt for VS 2022"** (Start menu → Visual Studio 2022 → ...). This pre-loads `cl.exe` + `INCLUDE`/`LIB` env vars needed to compile `pydensecrf`.

### 1.2 Create the conda env

```cmd
conda create -n ml python=3.11 -y
conda activate ml
cd /d "D:\Ubuntu\TECHNOTAU (2)\Project_management_and_training_NOV_11_2025"
```

### 1.3 Install dependencies

```cmd
pip install -r requirements_training.txt
```

This takes ~10-15 min on first install (downloads ~5 GB).

### 1.4 HuggingFace authentication (DINOv3 is gated)

```cmd
huggingface-cli login
```

Paste your HF token from <https://huggingface.co/settings/tokens>.
Then visit <https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m>
in your browser and click **"Agree and access repository"**.

### 1.5 Disable Windows sleep (so training survives the night)

In a regular admin PowerShell:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

(Restore later with `powercfg /change standby-timeout-ac 30`.)

### 1.6 Sanity-check GPU + imports

```cmd
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"
```
Expected: `NVIDIA GeForce RTX 3090 25.4 GB`

```cmd
python -c "from training.config import TrainingConfig; from training.model import build_model; cfg = TrainingConfig.from_yaml('configs/phase1.yaml'); print('config OK; backbone =', cfg.model.backbone_name)"
```
Expected: `config OK; backbone = facebook/dinov3-vith16plus-pretrain-lvd1689m`

---

## 1.7 Dual-GPU Setup (RTX 3090 + RTX 5070)

This system has two GPUs; we **only train on the 3090** (24 GB). The 5070
(12 GB) is too small to fit our model and is the wrong architecture for DDP.
Reserve it for parallel inference / dev work.

### 1.7.1 Identify which device index is which

```cmd
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Sample output:
```
index, name, memory.total [MiB]
0, NVIDIA GeForce RTX 3090, 24576 MiB
1, NVIDIA GeForce RTX 5070, 12288 MiB
```

Note which index is the 3090 (typically `0`).

### 1.7.2 Restrict training to the 3090 — set this BEFORE every training command

For the current cmd window only:
```cmd
set CUDA_VISIBLE_DEVICES=0
```

Or persistent across new cmd windows (until you `setx ... ""` to clear):
```cmd
setx CUDA_VISIBLE_DEVICES 0
```

### 1.7.3 Confirm PyTorch sees only the 3090

```cmd
python -c "import torch; print('GPUs visible:', torch.cuda.device_count()); print('GPU 0:', torch.cuda.get_device_name(0))"
```
Expected:
```
GPUs visible: 1
GPU 0: NVIDIA GeForce RTX 3090
```

If you see 2 GPUs or the wrong name, the env var didn't take effect — re-run
`set CUDA_VISIBLE_DEVICES=0` in the same shell as your training command.

### 1.7.4 Use the 5070 in parallel for inference (optional, after a checkpoint exists)

In a SEPARATE cmd window (don't share the env var with the training shell):

```cmd
:: This terminal: 5070 only, for inference / dev work
set CUDA_VISIBLE_DEVICES=1
python -m training.infer ^
    --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt ^
    --input preview_photos\ ^
    --output preview_predictions\ ^
    --post-process --dense-crf
```

Two GPUs, two processes, each sees its assigned device as `cuda:0`. No
conflict, no DDP overhead. Useful for iterating on web UI / customer photos
while training continues for days on the 3090.

### Why we don't use DDP across both

- DDP holds a full model copy on each GPU; 12 GB can't fit our model
- All-reduce waits for the slower GPU each step (Ampere + Blackwell mismatch)
- NCCL is Linux-only; on Windows we'd be forced to slow Gloo backend
- Blackwell sm_120 (5070) has incomplete PyTorch kernel coverage on Windows
- Net: DDP across these two GPUs is slower than 3090 alone

---

## 2. Data Preparation (one-time, ~1 min)

### 2.1 Build the wooden non-fence pool

```cmd
python -m tools.build_occluder_pool --wooden-negatives
```

This produces ~200 procedural wooden cutouts in `dataset/hard_negatives/wood/` for the `HardNegativeWoodPaste` augmentation.

### 2.2 Verify the occluder pool is in place

```cmd
python -m tools.build_occluder_pool --inspect --out-dir dataset/hard_negatives
```
Expected: `total occluder PNGs: 200` (plus per-subdir breakdown).

### 2.3 (Optional) Free disk space

If you've reviewed all annotation viz files:
```cmd
rmdir /s /q dataset\annotations_v1\viz
```
(Reclaims ~6-30 GB.)

---

## 3. Sanity Check Before Training (~30 sec)

Verify the dataset splits exist + are readable:

```cmd
python -c "from tools.dataset import verify_split_integrity; s = verify_split_integrity(); [print(f'  {k}: {v[\"rows\"]} rows') for k, v in s.items()]"
```

Expected output:
```
  train:    23394 rows
  val:       5013 rows
  test:      5016 rows
  train_hq: 14529 rows
  val_hq:    3087 rows
  test_hq:   3084 rows
```

If any split is missing, `verify_split_integrity` will raise an `AssertionError` telling you which file.

---

## 4. Phase 1 Training (~3-5 days on RTX 3090)

### 4.1 Launch (foreground, recommended)

In your activated env:

```cmd
python -m training.train --config configs/phase1.yaml
```

The first epoch downloads DINOv3-H+ (~3.4 GB) + DPT-Hybrid (~120 MB) into `~/.cache/huggingface/`. Expect 5-10 min before training actually starts.

### 4.2 Launch in background (so you can close the terminal)

Easiest: use `start /b` from a **persistent** shell:

```cmd
start /b python -m training.train --config configs/phase1.yaml > train_phase1.log 2>&1
```

Or use `nssm` (Non-Sucking Service Manager) to run as a Windows service — more robust but heavier setup. For most cases, `start /b` is enough.

### 4.3 Watch progress

In a second terminal:

```cmd
:: Tail the log
type train_phase1.log
:: Or follow it (PowerShell only):
powershell Get-Content train_phase1.log -Wait -Tail 50
```

### 4.4 TensorBoard live metrics

```cmd
:: In a third terminal
conda activate ml
tensorboard --logdir outputs/training_v2/phase1/logs
```

Open <http://localhost:6006/> in your browser.

### 4.5 What you'll see in the log

Early epochs (first ~5):
```
Verifying split integrity...
  train     rows=23,394  pos=11,234  neg=12,160  manual=8,500
  val       rows= 5,013  pos= 2,580  ...
Backbone: facebook/dinov3-vith16plus-pretrain-lvd1689m  patch_size=16
Model params: 1066.1M total, 943.7M trainable
Optimizer param groups: 32 (LR range 1.07e-06 .. 3.00e-04)
Starting training (100 epochs remaining)
[ep   1/100  it    50/5848  step 12]  loss=4.8231  lr=1.50e-05  |g|=2.103
[ep   1/100  it   100/5848  step 25]  loss=3.1245  lr=3.00e-05  |g|=1.842
...
Peak GPU memory: alloc=18.45GB  reserved=21.32GB  /  total=25.40GB  (16% headroom)
Epoch 1 done in 4521.3s   train_loss=2.7531
Val (380.2s):  val_iou=0.5234 val_dice=0.6541 val_boundary_iou=0.4321 ...
NEW BEST val_iou=0.5234 -> saved best.pt (EMA-swapped) + best_inference.pt
```

### 4.6 If training crashes mid-run

The pipeline has resume-from-checkpoint built in. Just relaunch with:

```cmd
python -m training.train --config configs/phase1.yaml --resume-from outputs/training_v2/phase1/checkpoints/latest.pt
```

It picks up at the last completed epoch with the same RNG state.

### 4.7 If you hit OOM (rare in phase 1)

Edit `configs/phase1.yaml`:
- `train.batch_size: 4 → 2`
- `optim.grad_accumulation_steps: 4 → 8` (keeps effective batch = 16)

Then resume from latest checkpoint (4.6).

---

## 5. Phase 2 Training (~2-3 days on RTX 3090)

After phase 1 finishes (you'll see a final test-set eval block in the log).

### 5.1 Verify phase 1 best exists

```cmd
dir outputs\training_v2\phase1\checkpoints
```
You should see `best.pt`, `best_inference.pt`, `latest.pt`, `ema.pt`, plus a few `epoch_NNN.pt`.

### 5.2 Launch phase 2

```cmd
python -m training.train --config configs/phase2.yaml
```

`configs/phase2.yaml` has `init_from: outputs/training_v2/phase1/checkpoints/best.pt` — phase 2 automatically loads phase 1's best weights as initialization.

### 5.3 If phase 2 OOMs (likely on RTX 3090 at 1024²)

Three escape hatches, ranked least → most quality cost. Pick ONE and resume:

**Option A** — drop depth in phase 2 only (saves ~1.5 GB):
```yaml
# configs/phase2.yaml
model:
  refinement_use_depth: false
```

**Option B** — single iteration refinement (saves ~1-2 GB):
```yaml
model:
  refinement_iterations: 1
```

**Option C** — drop edge head + iter aux (saves ~0.5-1 GB):
```yaml
model:
  refinement_iterations: 1
  refinement_use_edge_head: false
loss:
  edge_loss_weight: 0.0
  refinement_iter_aux_weight: 0.0
```

After editing, resume:
```cmd
python -m training.train --config configs/phase2.yaml --resume-from outputs/training_v2/phase2/checkpoints/latest.pt
```

---

## 6. Standalone Evaluation (~30-60 min per phase)

Validate your trained model against the test set with TTA + DenseCRF post-processing.

### 6.1 Eval phase 1 best on test split

```cmd
python -m tools.eval_checkpoint ^
    --checkpoint outputs/training_v2/phase1/checkpoints/best_inference.pt ^
    --split test ^
    --image-size 512 ^
    --batch-size 8 ^
    --tta-scales 0.75 1.0 1.25 ^
    --tta-flip ^
    --post-process ^
    --dense-crf ^
    --out-dir outputs/eval/phase1_test
```

### 6.2 Eval phase 2 best on test_hq

```cmd
python -m tools.eval_checkpoint ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --split test_hq ^
    --image-size 1024 ^
    --batch-size 2 ^
    --tta-scales 0.75 1.0 1.25 ^
    --tta-flip ^
    --post-process ^
    --dense-crf ^
    --out-dir outputs/eval/phase2_test_hq
```

Outputs:
- `outputs/eval/<run>/eval_summary.json` — aggregate IoU / Dice / boundary IoU / etc.
- `outputs/eval/<run>/eval_per_image.jsonl` — per-image scores with subcategory + review_source labels (for sliced analysis).

---

## 7. Inference on Real Images

### 7.1 Single image

```cmd
python -m training.infer ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --input photo.jpg ^
    --output mask.png ^
    --image-size 1024 ^
    --tta-scales 0.75 1.0 1.25 ^
    --tta-flip ^
    --post-process ^
    --dense-crf ^
    --save-overlay
```

Produces:
- `mask.png` — binary mask (0/255 PNG)
- `mask_overlay.png` — original photo with mask overlaid in red

### 7.2 Folder of images

```cmd
python -m training.infer ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --input customer_photos\ ^
    --output predictions\ ^
    --image-size 1024 ^
    --tta-scales 0.75 1.0 1.25 ^
    --tta-flip ^
    --post-process ^
    --dense-crf ^
    --save-overlay
```

### 7.3 Two-checkpoint ensemble (for hardest customer photos)

```cmd
python -m training.infer ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
                 outputs/training_v2/phase1/checkpoints/best_inference.pt ^
    --input customer_photos\ ^
    --output predictions_ensemble\ ^
    --image-size 1024 ^
    --tta-scales 0.75 1.0 1.25 ^
    --tta-flip ^
    --post-process ^
    --dense-crf
```

Probability-averages both models. ~2× inference time, slightly sharper boundaries.

---

## 8. ONNX Export (~2-5 min)

### 8.1 Export the phase 2 model to ONNX

```cmd
python -m tools.export_onnx ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --output models/fence_dinov3_phase2.onnx ^
    --image-size 1024 ^
    --opset 17
```

Produces:
- `models/fence_dinov3_phase2.onnx` — the model (~4.3 GB)
- `models/fence_dinov3_phase2.json` — sidecar with input/output specs

It auto-validates parity vs PyTorch (max abs diff should be < 5e-3).

### 8.2 Quantize to int8 (smaller, CPU-faster)

```cmd
python -m tools.export_onnx ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --output models/fence_dinov3_phase2.onnx ^
    --image-size 1024 ^
    --quantize-dynamic
```

Produces an additional `models/fence_dinov3_phase2_int8.onnx` (~1.1 GB).

### 8.3 Smaller dynamic-batch export (if you want to batch inference server-side)

```cmd
python -m tools.export_onnx ^
    --checkpoint outputs/training_v2/phase2/checkpoints/best_inference.pt ^
    --output models/fence_dinov3_phase2_dynamic.onnx ^
    --image-size 1024 ^
    --dynamic-batch
```

---

## 9. Files You'll End Up With

```
outputs/training_v2/phase1/
├── checkpoints/
│   ├── best.pt                  ~15.5 GB  (full training state, EMA-swapped)
│   ├── best_inference.pt        ~4.2 GB   (weights only — for shipping)
│   ├── latest.pt                ~15.5 GB  (resume point)
│   ├── ema.pt                   ~4.2 GB
│   └── epoch_NNN.pt × 3         ~46.5 GB  (last 3 periodic snapshots)
├── val_samples/                 sample mask PNGs per epoch
├── logs/                        TensorBoard event files
├── config.yaml                  resolved config snapshot
├── train.log                    full text log
├── val_metrics.jsonl            per-epoch val metrics
└── test_metrics.jsonl           final test eval

outputs/training_v2/phase2/
└── (same layout as phase1)

outputs/eval/<run>/
├── eval_summary.json
└── eval_per_image.jsonl

models/
├── fence_dinov3_phase2.onnx
├── fence_dinov3_phase2.json
└── fence_dinov3_phase2_int8.onnx  (if quantized)
```

---

## 10. After Training — Cleanup + Deploy

### 10.1 What to keep

- `best_inference.pt` from phase 2 — your production model
- `models/*.onnx` — for deployment
- `outputs/.../config.yaml` snapshots — for reproducibility
- `outputs/.../val_metrics.jsonl` + `test_metrics.jsonl` — for documentation

### 10.2 What's safe to delete (free up ~150 GB)

```cmd
:: Phase 1 periodic checkpoints (you have phase2 best now)
del outputs\training_v2\phase1\checkpoints\epoch_*.pt
:: Phase 1 latest.pt (no need to resume, it's done)
del outputs\training_v2\phase1\checkpoints\latest.pt
:: Phase 1 best.pt (full state — keep best_inference.pt instead)
del outputs\training_v2\phase1\checkpoints\best.pt
:: Same for phase 2 periodics (after you're sure best_inference is good)
del outputs\training_v2\phase2\checkpoints\epoch_*.pt
```

### 10.3 Production deployment

Once you have `best_inference.pt`, ship it:

- **Server-side (recommended)**: Modal.com or RunPod Serverless — see earlier discussion
- **Browser-only**: not feasible at this size (~1.1 GB int8 minimum); use one of your smaller existing models for browser preview

---

## 11. Troubleshooting

### Phase 1 fails immediately with `huggingface_hub.utils._errors.GatedRepoError`
You haven't accepted the DINOv3 license. Visit
<https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m>
and click "Agree and access repository", then retry.

### Phase 1 fails immediately with `RuntimeError: CUDA out of memory`
The first epoch may hit a slightly higher memory peak as PyTorch warms up the allocator. Drop `train.batch_size: 4 → 2` and `grad_accumulation_steps: 4 → 8` in `configs/phase1.yaml` and relaunch.

### Phase 1 takes way longer than estimated
Open Task Manager → Performance → GPU. If GPU usage is < 80% during training, you're CPU/IO-bound. Bump `train.num_workers: 6 → 12` in `configs/phase1.yaml`.

### `pydensecrf` install error: "Compiler: cl is not found"
You're not in the VS Native Tools shell. Either:
- Open "x64 Native Tools Command Prompt for VS 2022" and reinstall, OR
- In your current cmd:
  ```cmd
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
  pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
  ```

### TensorBoard shows nothing
TensorBoard reads `outputs/training_v2/<run>/logs/`. Make sure you launched with the correct `--logdir`. Also wait until at least one epoch completes — events are buffered.

### `best_inference.pt` is missing after training
This file is only written when val improves a metric. If your val never improved (rare — almost certainly an early bug), use `best.pt` instead and load with `--config configs/phase2.yaml`.

---

## Quick reference card

| Stage | Command |
|---|---|
| Setup | `pip install -r requirements_training.txt && huggingface-cli login` |
| Wood pool | `python -m tools.build_occluder_pool --wooden-negatives` |
| Phase 1 | `python -m training.train --config configs/phase1.yaml` |
| Phase 2 | `python -m training.train --config configs/phase2.yaml` |
| Resume | `python -m training.train --config <yaml> --resume-from outputs/training_v2/<run>/checkpoints/latest.pt` |
| Eval | `python -m tools.eval_checkpoint --checkpoint <ckpt> --split test_hq --image-size 1024 --tta-flip --post-process --dense-crf` |
| Infer | `python -m training.infer --checkpoint <ckpt> --input <img-or-dir> --output <dst> --post-process --dense-crf --save-overlay` |
| ONNX | `python -m tools.export_onnx --checkpoint <ckpt> --output <onnx-path> --image-size 1024 --quantize-dynamic` |
| TensorBoard | `tensorboard --logdir outputs/training_v2/<run>/logs` |
