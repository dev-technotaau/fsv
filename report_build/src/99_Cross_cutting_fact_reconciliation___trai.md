# AREA: Cross-cutting fact reconciliation + training cost/time math (client report)

## SUMMARY
This pass reconciles the numbers that appear across every discovery reader and turns the user's authoritative spend figures into the headline training-cost narrative for the client PDF. The money-derived headline is firm: $220 / $1.14 per GPU-hour = 192.98 ≈ 193 GPU-hours of rental, about 8.0 days of continuous single-GPU time, with bandwidth/storage carved out of that $220 so pure compute sits slightly under 193h. The DINOv3 ViT-L/16 Phase-1 run is confirmed at ~14,115 s/epoch (~3.92h) over 24 completed epochs (val_iou 0.4244 → 0.5014), but the logs actually record 37 epoch-rows (~146h) once restarts are counted, with the remainder of the ~193h absorbed by the earlier DINOv2 web_deployable run, ONNX exports, experiments and multi-day model downloads. Two material contradictions need the orchestrator's attention: the run provenance says NVIDIA A100-SXM4-80GB (not the H100 the user stated), and training/config.py defaults to the H+ backbone even though the shipped checkpoint is unambiguously ViT-L/16.

## KEY_FACTS
- HEADLINE training rental: $220 total spend / $1.14 per GPU-hour = 192.98 ≈ ~193 GPU-hours of rental-equivalent (verified arithmetic).
- 193 GPU-hours / 24 = 8.04 ≈ ~8.0 days of continuous single-GPU rental.
- The $220 is all-in (compute + 'plus bandwidth charges' = bandwidth/storage included), so pure compute is slightly UNDER ~193h; ~193h is the correct headline as rental-equivalent and bandwidth trims the compute portion.
- Per-epoch wall time confirmed from outputs/training_v2/phase1/val_metrics.jsonl: mean 14,115.4 s/epoch (~3.92h) across the 24 unique epochs; min 13,951.8 s, max 14,952.5 s.
- 24 unique completed epochs x ~3.92h = 94.1 GPU-hours of clean forward progress (this is the SANITY cross-check, NOT the headline).
- The val_metrics.jsonl file actually holds 37 epoch-rows, not 24 — duplicates from ~4 restarts (v1-v4). Summing all 37 logged rows = 146.43 hours of actually-logged training compute.
- The gap between ~193h rental and ~146h logged training = ~47h, consistent with the earlier DINOv2 web_deployable run, ONNX exports, experiments/restarts, idle warm time, and multi-day weight downloads (aria2 download logs in models/ span May 27, ~hundreds of MB of retry logs).
- val_iou progression verified: epoch 1 = 0.4243555 (val_dice 0.5958562), epoch 24 = 0.5013831 (val_dice 0.6678950). Climb of +0.077 IoU over 24 epochs.
- DEPLOYED model = epoch 24, val_iou 0.5014 (per live best_inference.pt meta cited by the models/ reader); the models/fence_dinov3_phase1.json.bak sidecar I read is an OLDER ep18 snapshot (checkpoint_meta.epoch=18, global_step=26190, val_iou null in that field) — NOT the deployed weights.
- Phase 1 progress: 24 of 120 epochs = 20% complete; Phase 2 (70 epochs) NOT started (outputs/training_v2/ contains only phase1/).
- CONTRADICTION (hardware): training_provenance.gpu.name in models/fence_dinov3_phase1.json.bak = 'NVIDIA A100-SXM4-80GB' (85.09 GB, compute_capability 8.0, cuda 12.6), NOT the H100 80GB the user listed as ground truth. The vast.ai container hostname is d6e2e6426981.
- Train set after filtering = 23,291 samples (min_fence_pixels_for_pos=100); dataset_summary in provenance confirms 'train_samples_after_filter': 23291.
- Cloud Run spec VERIFIED from cloudrun_inference/deploy.sh: --gpu 1 --gpu-type nvidia-l4, --memory 16Gi --cpu 4, --concurrency 4, --min-instances 0 --max-instances 1, --region us-central1, --no-cpu-throttling, --execution-environment gen2, --timeout 300s, --allow-unauthenticated (deploy.sh:56-66).
- Cloud Run WARM (min-instances=1, 730h/month) realistic cost RANGE: ~$330-$520/month all-in (L4 GPU + 4 vCPU + 16 GiB, billed per-second with CPU always allocated). L4 on-demand ~$0.71/GPU-hr x 730 = ~$518 GPU alone; Cloud Run gen2 GPU billing plus 4 vCPU + 16 GiB pushes the defensible band to roughly $350-$550/mo. Present as 'a few hundred USD/month'.
- Cloud Run SCALE-TO-ZERO (current, min-instances=0): near-$0 at idle; cost scales with traffic — at low volume (hundreds-to-thousands of detects/month) likely <$5-$30/month, plus cold-start latency (~30-60s GPU boot + 2.45 GB VRAM load) on the first request after each scaledown.
- Modal sibling (modal_inference/app_dinov3.py) uses gpu='T4', scaledown_window=600, min_containers=0, max_containers=4 — a cheaper-GPU scale-to-zero equivalent; always-on Modal min_containers=1 adds ~$10+/mo per the README estimate but a warm GPU container is materially higher.
- Production ONNX is identical across models/ and cloudrun_inference/: graph 13,902,638 bytes (~13.9 MB) + external weights fence_dinov3_phase1.onnx.data = 2,451,570,688 bytes (~2.45 GB), total ~2.47 GB.
- Image-count reconciliation (positive corpus, NOT reconciled — three different ledgers): metadata.jsonl = 21,674 lines; dedup.sqlite images table = 21,518 rows; files on disk = 21,414. Negative corpus reconciles cleanly at 12,009 across metadata/sqlite/disk.
- Raw scrape combined total = 33,423 images AFTER cross-set dedup (21,414 pos + 12,009 neg on disk; 251 cross-set duplicates removed, all reason 'cross-set-dup').
- Class balance FLIPPED during manual review: auto-label baseline manifest.jsonl = 21,414 pos / 12,009 neg; manifest_final.jsonl (post manual review, class_source='manual_review') = 13,328 pos / 20,095 neg — ~8,000 images reclassified pos->neg.
- CURRENT splits (from manifest_final.jsonl, seed 42, --group-by-dhash): train 23,394 / val 5,013 / test 5,016 = 33,423 total (ratios_actual 0.6999/0.15/0.1501). HQ splits: train_hq 14,529 / val_hq 3,087 / test_hq 3,084 (manifest_hq_final.jsonl 20,700 rows, shorter-edge>=1024).
- DATASHEET.md is STALE: quotes pre-review 21,414/12,009 class counts and 23,397/5,011/5,015 split counts; actual is 13,328/20,095 and 23,394/5,013/5,016.
- License audit (33,423): SAFE 12,367 (37.0%), SAFE_ATTR 3,519 (10.5%, wikimedia attribution), RISKY 17,537 (52.5%); recommended commercial subset manifest_safe.jsonl = 15,886 rows.
- export_onnx.py: opset 17 requested but auto-upgraded to 18 at export (ScatterND lacks opset-17 adapter, onnx_export.log); temperature baked into graph (default 1.0); output 'mask_prob' (1,1,512,512) sigmoid; parity tol 5e-3 (ep18 parity passed max_abs_diff 3.32e-5; ep24 parity OOM'd on laptop, unvalidated).
- CONTRADICTION (backbone default): training/config.py:26 sets backbone_name default to 'facebook/dinov3-vith16plus-pretrain-lvd1689m' (H+, ~840M) — so H+ is a literal code default, not merely a comment; but configs/phase1.yaml:13 and phase2.yaml:13 override to 'facebook/dinov3-vitl16-pretrain-lvd1689m' (ViT-L/16) and the shipped checkpoint meta confirms vitl16, image_size 512, patch_size 16. Production = ViT-L/16, confirmed.
- phase1.yaml multi_block_n: 6 and pixel_decoder_layers: 12 carry 'rolled back from 6/12' comments that equal the live value — the comments are stale edit-history; the ACTIVE numeric values are authoritative.

## FILE_ROLES
- [current] outputs/training_v2/phase1/val_metrics.jsonl — Per-epoch validation+timing log (37 rows, 24 unique epochs); source of the ~14,115 s/epoch and IoU progression figures
- [backup] models/fence_dinov3_phase1.json.bak — ONNX sidecar snapshot at epoch 18 (training_provenance with A100 GPU, training_config); NOT the deployed ep24 sidecar
- [backup] models/fence_dinov3_phase1.json.bak_ep23 — ONNX sidecar snapshot at epoch 23
- [current] models/fence_dinov3_phase1.onnx — Deployed ep24 ONNX graph (13.9 MB), Jun 4 export
- [current] models/fence_dinov3_phase1.onnx.data — Deployed ep24 external weights (2.45 GB)
- [backup] models/fence_dinov3_phase1.onnx.data.bak — ep18 weights backup (May 28)
- [backup] models/fence_dinov3_phase1.onnx.data.bak_ep23 — ep23 weights backup (May 29)
- [current] models/onnx_export.log — Export run log: opset 17->18 upgrade, ep24 parity OOM on laptop GPU
- [artifact] models/aria2_*.log, models/aria2_input*.txt, models/_make_aria2_*.py — Multi-day vast.ai weight-download tooling/logs (bandwidth-heavy); explains part of the ~47h non-training rental
- [current] cloudrun_inference/deploy.sh — Cloud Run GPU deploy script; authoritative source for L4/16Gi/4vCPU/min0-max1 spec
- [current] cloudrun_inference/fence_dinov3_phase1.onnx + .onnx.data — Copies of deployed ONNX baked into Cloud Run image (byte-identical to models/)
- [current] modal_inference/app_dinov3.py — Modal serverless equivalent (T4 GPU, scale-to-zero); cost-comparison reference

## NARRATIVE
## Training cost and time — derived from the money, not the epochs

The single most defensible number for the client report is the one that comes straight out of the invoice. Total vast.ai spend was **$220**, the on-demand rate was **$1.14 per GPU-hour**, so the rental-equivalent compute is:

> $220 ÷ $1.14/hr = **192.98 ≈ ~193 GPU-hours** ≈ **~8.0 days** of continuous single-GPU rental (193 ÷ 24 = 8.04 days).

The user was right to insist this be derived from the money rather than from an epoch count. An epoch-based estimate undercounts because it ignores restarts, idle warm time, experiments, ONNX exports, and the genuinely large amount of time spent just *downloading* the gated DINOv3 weights and pushing the 2.45 GB model artifacts around. The phrase "plus bandwidth charges" means the $220 is all-in: bandwidth and storage are a *subset* of that figure sitting on top of raw compute. So the honest framing for the report is: **~193 GPU-hours is the all-in rental-equivalent; the pure-compute slice is slightly under that because bandwidth/storage ate part of the $220.** I'd present it as "roughly eight days of H100-class GPU time, all-in," and avoid implying that every one of those 193 hours was spent in the training loop.

### The epoch cross-check (sanity only)

I confirmed the per-epoch timing directly from `outputs/training_v2/phase1/val_metrics.jsonl`. Across the 24 unique completed epochs, wall-clock time averaged **14,115 seconds per epoch (~3.92 hours)**, with a tight spread (min 13,952 s at epoch 23, max 14,953 s at epoch 1). That is the ~14,000 s/epoch figure the brief asked me to verify — confirmed, on an A100-80GB. So:

- 24 clean epochs × 3.92 h = **94.1 GPU-hours** of forward progress.

But that 94h is *not* the whole story, and this is where the epoch-based estimate falls down. The same JSONL file actually contains **37 epoch-rows, not 24** — the training was restarted roughly four times (the discovery readers labelled these v1 through v4, including one "v4_buggy" block from a code bug that was later fixed). If you sum the `epoch_seconds` across all 37 logged rows you get **146.4 hours** of compute that the trainer actually burned, because restarts re-ran epochs that had already been done once. That single fact closes most of the gap between the clean 94h and the money-derived 193h.

The remaining ~47 hours (193 − 146) is exactly the kind of overhead you'd expect and that the `models/` directory documents in passing: the earlier DINOv2 `web_deployable` run (which produced the 135 MB DINOv2-small browser model, val_iou 0.4253 at epoch 19), repeated ONNX exports, the laptop-side parity checks, and a multi-day saga of downloading and re-uploading weights — the `models/` folder is littered with `aria2` download logs, mirror lists, and shard-retry scripts from May 27 that show how painful the bandwidth was (the SSH cap was ~1 MB/s, which is why a 2.45 GB artifact becomes an hours-long, retry-heavy operation that still bills GPU rental the whole time the box is up).

So the layered picture for the client is clean and honest:
- **~193 GPU-hours / ~8 days** = total rental, money-derived (headline).
- **~146 hours** = compute actually logged by the trainer (includes restarts).
- **~94 hours** = the 24 unique epochs of clean DINOv3 ViT-L progress that produced the deployed model.
- The balance = DINOv2 run, exports, experiments, idle, and downloads.

### Training outcome

The deployed model is **epoch 24, val_iou 0.5014** (val_dice 0.6679). IoU climbed from 0.4244 at epoch 1 to 0.5014 at epoch 24 — steady, not plateaued, which is consistent with the run being stopped at only **24 of a planned 120 epochs (20%)**. Phase 2 (1024px, 70 epochs) never started; `outputs/training_v2/` contains only `phase1/`. No held-out test metrics exist anywhere, because training was interrupted before the finish-time test hook — every quoted number is a *validation* metric.

One hardware contradiction the orchestrator must resolve before the report goes out: the user listed the instance as **1× NVIDIA H100 80GB**, but the actual run provenance baked into the checkpoint says **NVIDIA A100-SXM4-80GB** (85.09 GB, compute capability 8.0, CUDA 12.6, torch 2.11.0+cu126, on vast.ai container `d6e2e6426981`). The cost math is unaffected — $220 ÷ $1.14 holds regardless of card — but the report should not claim "H100" if the evidence on disk says A100. My recommendation is to describe it as "an 80 GB data-center GPU (A100/H100-class)" unless the user can confirm the H100 from a separate invoice, and to flag the discrepancy to them directly.

## Hosting cost — Cloud Run warm vs. the alternatives

The current production host is Google Cloud Run, and I verified the exact spec from `cloudrun_inference/deploy.sh:56-66`: **1× nvidia-l4 GPU, 4 vCPU, 16 GiB RAM, --no-cpu-throttling (CPU always allocated), gen2 execution, us-central1, --min-instances 0 --max-instances 1, 300s timeout.** Today it is **scale-to-zero**, which is why the page warms the container with a health ping and why the first request after idle eats a ~30-60s cold start (GPU boot plus loading the 2.45 GB model into VRAM).

**(a) Cloud Run, kept warm 24/7 (min-instances=1, 730 h/month).** This is the option the client asked to price. Cloud Run bills GPU + vCPU + memory per second, and with `--no-cpu-throttling` the CPU is billed for the full warm period, not just during requests. An L4 on-demand is roughly **~$0.71/GPU-hour**; 730 hours of GPU alone is ~$518. Add 4 always-allocated vCPU and 16 GiB and the all-in lands in a defensible band of roughly **$350-$550 per month**. The honest way to state it: *"keeping the GPU warm around the clock is a few hundred dollars a month — call it $350-$550 depending on exact Cloud Run GPU rates and committed-use discounts."* The benefit is no cold starts; every visitor gets a sub-second-to-few-second response.

**(b) Cloud Run as-is (scale-to-zero) vs. a serverless/dedicated alternative.** Left at min-instances=0, the bill tracks traffic: at low volume (hundreds to low-thousands of detections per month) it is realistically **under $5-$30/month**, at the cost of an occasional cold start. The Modal sibling (`modal_inference/app_dinov3.py`) is the same protocol on a cheaper **T4** GPU with `min_containers=0` and a 10-minute warm window — a sensible cheaper-GPU scale-to-zero option; Modal's own README pegs an always-warm CPU container at ~$10/mo, but an always-warm *GPU* container is materially more (low hundreds). A dedicated L4 or A10 VM (Compute Engine, RunPod, Lambda) runs in the same **~$300-$600/month** range as warm Cloud Run but trades the serverless convenience for you managing the box.

**(c) Buy/build a small local GPU server.** A used RTX 3090 (24 GB) box runs maybe **$1,000-$1,800 one-time**; a new RTX 4090 build is **~$2,500-$3,500**. Either has plenty of VRAM for the 2.47 GB model. Electricity at ~350-450 W under load and a typical ~$0.15/kWh is on the order of **$15-$40/month** if it runs continuously. So the crossover vs. warm cloud (~$400/mo) is roughly **3-6 months** — capex pays for itself fast if the simulator sees steady traffic, with the usual caveats (you own uptime, networking, security, and there's no auto-scale for spikes). For a single-tenant marketing tool on the client's own site, a small local/colo box is genuinely competitive; for bursty public traffic, scale-to-zero Cloud Run remains the cheapest floor.

The pragmatic recommendation to put in front of the client: **keep Cloud Run scale-to-zero as the default** (near-zero idle cost, accept the cold start), and only move to warm min-instances=1 (~$350-$550/mo) if cold-start latency becomes a conversion problem — at which point a local 4090/3090 box becomes the cheaper long-run answer.

## Fact reconciliation across the readers

The discovery passes broadly agree; the contradictions worth surfacing are narrow and I've resolved most of them:

- **Production backbone is DINOv3 ViT-L/16**, confirmed: `configs/phase1.yaml:13` overrides to `facebook/dinov3-vitl16-pretrain-lvd1689m`, and the shipped checkpoint meta says vitl16 / image_size 512 / patch_size 16. The wrinkle is that `training/config.py:26` *defaults* to the **H+** string — so H+ is a real code default, not just a comment, and that's why the export log printed it. The shipped weights are unambiguously ViT-L; the H+ default is a latent footgun, not the deployed model.
- **Positive image counts genuinely do not reconcile** across the three ledgers: metadata.jsonl 21,674 lines, dedup.sqlite 21,518 rows, files on disk 21,414. The negative side reconciles perfectly at 12,009. Disk is authoritative for "what exists."
- **Class balance flipped during manual review**: the auto-label baseline was 21,414 pos / 12,009 neg; after manual review `manifest_final.jsonl` is 13,328 pos / 20,095 neg (~8,000 images moved pos→neg). The committed splits (23,394 / 5,013 / 5,016) come from `manifest_final.jsonl`, not the original manifest.
- **DATASHEET.md is stale** — it still quotes the pre-review class counts and slightly different split counts (23,397/5,011/5,015). It should be refreshed or footnoted before it goes in the deliverable.
- **Deployed epoch is 24** (val_iou 0.5014); the `.json.bak` sidecars in `models/` are older ep18/ep23 snapshots and there is no current sidecar matching the live ep24 ONNX. Calibration (temperature 1.0, threshold 0.5) is assumed unchanged for ep24 — reasonable but not separately recorded.
- The "Modal" naming in the browser CONFIG (`MODAL_ENDPOINT`) is a stale label; the live endpoint is a **Cloud Run** URL (`fsv-dinov3-...us-central1.run.app`). Cloud Run is the current host; Modal was the earlier one.

## UNCERTAINTIES
- HARDWARE MISMATCH: user states 1x H100 80GB, but checkpoint training_provenance says NVIDIA A100-SXM4-80GB. The $220/$1.14 math is unaffected, but the report must not claim H100 unless the user confirms it from a separate invoice. Needs orchestrator/user reconciliation.
- The $1.14/GPU-hour rate is user-provided ground truth; I did not find a vast.ai billing record in-repo to independently confirm the rate or the $220 total. Both are taken as authoritative per the brief.
- Cloud Run GPU pricing is not in the repo (deploy.sh defines only the resource spec). My ~$350-$550/mo warm range and ~$0.71/GPU-hr L4 figure are from general GCP pricing knowledge, not a repo citation — present as a defensible range, not exact, and re-verify against current GCP Cloud Run GPU rates at report time.
- epoch_seconds in val_metrics.jsonl includes per-epoch validation time, so ~14,115 s/epoch is train+val wall time, not pure training compute. The 146h all-rows sum likewise includes val.
- I attributed the ~47h gap (193h rental - 146h logged) to DINOv2 run + exports + downloads + idle by inference from the models/ artifacts (aria2 logs, web_deployable run); I could not itemize the hours precisely. The breakdown is directional, not exact.
- The positive-corpus three-way count drift (21,674 / 21,518 / 21,414) is real but its root cause is undetermined from the data alone.
- Phase-2 cost is purely hypothetical (never ran). If the client asks 'cost to finish', deriving remaining-hours x $1.14 would require an assumption about the A100 vs H100 rate and whether Phase 2's 1024px epochs run slower than Phase 1's 512px ~3.92h/epoch — they will be materially slower, so any 'cost to complete' estimate needs its own caveat.
- The deployed ep24 ONNX never passed an end-to-end parity check (the post-export validation OOM'd on the laptop); ep18 parity passed at 3.3e-5. Architecture is unchanged so it is almost certainly correct, but unvalidated.