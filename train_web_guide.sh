# Quick start (default 80-epoch run)

python train_web_deployable.py

# That's it. Reads tools/dataset.py + dataset/splits/, trains the ~35M-param model, runs final test eval with TTA, exports ONNX (fp32 + fp16) at the end. Outputs to outputs/web_deployable/web_v1/.



---------
# other commands and flags ->
Common scenarios

# Bigger batch + fewer epochs (faster iteration)
python train_web_deployable.py --epochs 50 --batch-size 16

# Higher LR (if loss isn't moving)
python train_web_deployable.py --lr 6e-4

# Different run name (so it doesn't overwrite previous)
python train_web_deployable.py --run-name web_v2

# Resume from where you left off
python train_web_deployable.py --resume-from outputs/web_deployable/web_v1/checkpoints/latest.pt

# Init weights from a flagship checkpoint (only model, not optimizer)
python train_web_deployable.py --init-from outputs/training_v2/phase1/checkpoints/best.pt --run-name web_initfromflagship

# Bigger backbone (DINOv2-Base = 86M, fp32 ONNX ~340MB, fp16 ~170MB — over 100MB limit)
python train_web_deployable.py --backbone facebook/dinov2-base --run-name web_base

# Train at 1024² instead (auto-switches to phase2 augmentation since > 768)
python train_web_deployable.py --image-size 1024 --batch-size 4 --run-name web_1024
Generic overrides — any TrainingConfig field via --section.field VALUE
The script also accepts --section.field value for ANY field in the config (mirrors flagship). Auto-coerces true/false/int/float. Examples organized by section:

--model.* (architecture)

# Bigger decoder
python train_web_deployable.py --model.decoder_dim 256 --model.decoder_num_layers 9

# More mask queries
python train_web_deployable.py --model.decoder_num_queries 32

# Smaller refinement (saves params)
python train_web_deployable.py --model.refinement_channels 48 --model.refinement_num_blocks 2

# Turn ON ViT-Adapter (will push params past 100MB ONNX limit — for experiments only)
python train_web_deployable.py --model.use_vit_adapter true --run-name web_with_adapter

# Turn ON DPT depth (slows training ~25%, biggers checkpoint)
python train_web_deployable.py --model.refinement_use_depth true

# Disable PointRend module
python train_web_deployable.py --model.refinement_use_pointrend_module false

# Freeze backbone for first N transformer layers
python train_web_deployable.py --model.backbone_freeze_first_n_layers 6

# Disable gradient checkpointing (faster but more VRAM)
python train_web_deployable.py --model.gradient_checkpointing false

# torch.compile (PyTorch 2.4+)
python train_web_deployable.py --model.torch_compile true
--loss.* (loss weights)

# Push boundary quality harder
python train_web_deployable.py --loss.boundary_weight 1.5 --loss.edge_loss_weight 1.5

# Stronger CGM gating
python train_web_deployable.py --loss.cgm_weight 1.0

# Stronger BDR (boundary distance)
python train_web_deployable.py --loss.boundary_distance_weight 0.6

# More OHEM (top-K hardest pixels)
python train_web_deployable.py --loss.ohem_top_k_ratio 0.4

# Disable focal loss (fall back to plain BCE)
python train_web_deployable.py --loss.focal_gamma 0

# Stronger Tversky recall focus
python train_web_deployable.py --loss.tversky_alpha 0.85 --loss.tversky_beta 0.15

# Disable PointRend importance sampling
python train_web_deployable.py --loss.use_pointrend false

# Turn on EMA self-distillation (Mean Teacher consistency loss)
python train_web_deployable.py --loss.ema_distill_weight 0.3
--optim.* (optimizer / scheduler / AMP)

# Higher backbone LR
python train_web_deployable.py --optim.backbone_lr 1e-4

# More gradient accumulation (effective batch = batch_size × accum)
python train_web_deployable.py --optim.grad_accumulation_steps 8

# Switch to fp16 AMP (instead of bf16 — needed if your GPU is < Ampere)
python train_web_deployable.py --optim.amp_dtype fp16

# Disable AMP entirely (debugging)
python train_web_deployable.py --optim.use_amp false

# Switch to SGD
python train_web_deployable.py --optim.optimizer sgd --optim.momentum 0.9

# Tighter gradient clipping
python train_web_deployable.py --optim.grad_clip_norm 0.5

# Longer warmup
python train_web_deployable.py --optim.warmup_epochs 8
--train.* (training behavior)

# Disable CutMix
python train_web_deployable.py --train.cutmix_p 0.0

# Stronger CutMix
python train_web_deployable.py --train.cutmix_p 0.5

# Disable multi-scale training
python train_web_deployable.py --train.multi_scale_train false

# Wider multi-scale range (more augmentation)
python train_web_deployable.py --train.multi_scale_min_factor 0.5 --train.multi_scale_max_factor 1.5

# Disable balanced sampler (use natural data distribution)
python train_web_deployable.py --train.use_balanced_sampler false

# Inverse-frequency sampling (more aggressive class balance)
python train_web_deployable.py --train.balance_alpha 1.0

# Disable EMA
python train_web_deployable.py --train.use_ema false

# Slower EMA decay (more responsive to recent weights)
python train_web_deployable.py --train.ema_decay 0.999

# Validate every 2 epochs instead of every epoch (faster training)
python train_web_deployable.py --train.val_every_n_epochs 2

# Turn on early stopping (8 epochs no improvement → break)
python train_web_deployable.py --train.early_stop_patience 8

# Disable final test eval
python train_web_deployable.py --train.run_test_eval_on_finish false

# Different seed (for ensemble training)
python train_web_deployable.py --train.seed 123 --run-name web_seed123

# Deterministic mode (slower, bit-exact reproducibility)
python train_web_deployable.py --train.deterministic true

# Enable TTA at validation too (every epoch, slower)
python train_web_deployable.py --train.use_tta true

# Disable TTA at final test eval
python train_web_deployable.py --train.tta_at_final_test false

# More workers
python train_web_deployable.py --train.num_workers 12

# Don't skip non-finite loss batches (debugging)
python train_web_deployable.py --train.skip_step_on_nonfinite_loss false
--data.* (dataset)

# Train on the HQ subset instead of full data (rare — the HQ is for phase 2 normally)
python train_web_deployable.py --data.train_split train_hq --data.val_split val_hq --data.test_split test_hq

# Different splits directory
python train_web_deployable.py --data.splits_dir /path/to/other/splits
--post.* (final test eval post-processing)

# Disable DenseCRF (faster eval)
python train_web_deployable.py --post.use_dense_crf false

# Disable connected-component cleanup
python train_web_deployable.py --post.use_cc_cleanup false

# Wider CC blob filter (drops more specks)
python train_web_deployable.py --post.cc_min_blob_area 500

# Disable post-processing entirely
python train_web_deployable.py --post.enabled false
--log.* + --ckpt.*

# Custom output directory
python train_web_deployable.py --log.log_dir outputs/my_experiments

# Log more frequently
python train_web_deployable.py --log.log_every_n_steps 10

# Save more sample prediction PNGs per epoch
python train_web_deployable.py --log.save_sample_predictions 16

# Disable TensorBoard
python train_web_deployable.py --log.use_tensorboard false

# Save periodic checkpoints every 10 epochs (instead of 5)
python train_web_deployable.py --ckpt.save_every_n_epochs 10

# Keep more periodic checkpoints
python train_web_deployable.py --ckpt.keep_last_n 10

# Track best by Dice instead of IoU
python train_web_deployable.py --ckpt.save_best_metric val_dice

# Don't save optimizer state (smaller checkpoints, no resume support)
python train_web_deployable.py --ckpt.save_optimizer_state false
Combined examples (real-world recipes)

# Faster iteration: quick sanity-check run
python train_web_deployable.py --epochs 10 --batch-size 8 --train.val_every_n_epochs 5 --run-name web_smoke

# High-quality: longer training with stronger augmentation + EMA distill
python train_web_deployable.py \
    --epochs 120 \
    --train.cutmix_p 0.5 \
    --loss.ema_distill_weight 0.3 \
    --loss.boundary_weight 1.0 \
    --loss.edge_loss_weight 1.0 \
    --run-name web_hq_v1

# 3-model ensemble (run 3 times with different seeds + run names)
python train_web_deployable.py --train.seed 42  --run-name web_ens_seed42
python train_web_deployable.py --train.seed 123 --run-name web_ens_seed123
python train_web_deployable.py --train.seed 999 --run-name web_ens_seed999

# Maximum-capacity run (turn on the things we left off for budget)
python train_web_deployable.py \
    --model.refinement_channels 96 \
    --model.refinement_num_blocks 4 \
    --model.refinement_iterations 3 \
    --model.use_vit_adapter true \
    --backbone facebook/dinov2-base \
    --run-name web_max
# (Will produce a ~80-100M model — fp16 ONNX ~160-200MB, OVER browser limit. Run for analysis only.)


#Where outputs land

outputs/web_deployable/<run-name>/
├── config.yaml                          # exact config used (audit/reproduction)
├── train.log                            # console log
├── val_metrics.jsonl                    # per-epoch val metrics
├── test_metrics.jsonl                   # final test eval metrics
├── logs/
│   └── tensorboard/                     # TB events
├── checkpoints/
│   ├── best.pt                          # best.pt with EMA-swapped weights + optimizer state
│   ├── best_inference.pt                # weights-only, EMA-swapped (used for ONNX)
│   ├── latest.pt                        # most recent state (for resume)
│   ├── ema.pt                           # EMA-only weights
│   └── epoch_NNN.pt                     # periodic snapshots (every save_every_n_epochs)
├── val_samples/
│   ├── epoch_001/*.png                  # per-epoch validation sample panels
│   ├── epoch_002/*.png
│   └── test_final/*.png                 # final test eval sample panels
└── onnx/
    ├── model.onnx                       # ~140 MB fp32 (analysis copy)
    ├── model_fp16.onnx                  # ~70 MB fp16  ← ★ DEPLOY THIS TO BROWSER ★
    └── model.json                       # sidecar (preprocessing + parity stats + provenance)

#Help

python train_web_deployable.py --help

# Shows all explicit flags + the example block from the script's epilog.