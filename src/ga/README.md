# GA search over 18 fence-segmentation model combos

Two-stage genetic algorithm:

1. **Stage 1 — model-family search.** Mutates *which* of the 18 combos to use + their per-combo hyperparams. Finds the best architectural family. Uses niching quotas to preserve diversity.
2. **Stage 2 — hyperparameter GA.** Locks the Stage 1 winner's architecture, runs a longer hyperparam GA on its params only.

## Compute warning

A GA with `population=12` × `generations=10` = **120 full training runs**. Each run is 1–8 hours depending on architecture and `proxy_epochs`. **Budget weeks of wall-clock time** on a single 6GB GPU.

**Cost-reduction features built in:**
- Short proxy training per individual (`fitness.proxy_epochs`).
- Early kill if IoU < threshold at 30% of budget.
- Fitness cache — never re-evaluate an identical genome.
- Resume from checkpoint (SIGKILL-safe, atomic writes).
- Dry-run mode with random fitness to validate the GA loop end-to-end in seconds.

## Layout

```
src/ga/
├── __init__.py
├── cli.py                    # Typer CLI entry point
├── config.py                 # Pydantic config + YAML loader
├── registry.py               # 18 combo definitions + search spaces
├── genome.py                 # Genome + mutation/crossover operators
├── population.py             # GA loop, selection, elitism, HoF
├── fitness.py                # subprocess-isolated fitness eval + composite score
├── fitness_cache.py          # JSONL-backed memoization
├── checkpoint.py             # atomic save/resume
├── logger.py                 # rich + TensorBoard + CSV + JSONL audit trail
├── validators.py             # environment preflight
├── exceptions.py             # typed exceptions
└── adapters/
    ├── base.py                                    # ModelAdapter ABC + metric helpers (IoU, BF1, TV)
    ├── _common.py                                 # Shared dataset, losses, decoders, train loop
    ├── _timm_backbone.py                          # timm features_only helper
    ├── _subprocess_helpers.py                     # Subprocess trainer wrapper helpers
    ├── combo_01_dinov2_l_m2f.py                   ✅ full (DINOv2-L + M2F-lite)
    ├── combo_02_dinov2_g_upernet.py               ✅ full (DINOv2-G, falls back to L on OOM)
    ├── combo_03_sam2_encoder_m2f.py               ✅ full (SAM 2 preferred, SAM v1 fallback)
    ├── combo_04_sam2_full_finetune.py             ✅ full (auto-mask inference)
    ├── combo_05_eva02_l_m2f.py                    ✅ full (timm EVA-02-L)
    ├── combo_06_internimage_l_upernet.py          ✅ full (falls back to ConvNeXt-L if DCNv3 unavailable)
    ├── combo_07_swinv2_l_m2f.py                   ✅ full (timm Swin-V2-L + deep supervision)
    ├── combo_08_convnextv2_l_upernet.py           ✅ full (timm ConvNeXt-V2-L)
    ├── combo_09_segformer_b5_premium.py           ✅ full (subprocess → project trainer)
    ├── combo_10_m2f_segformer_b5.py               ✅ full (subprocess → project trainer)
    ├── combo_11_unetpp_b7.py                      ✅ full (subprocess → project trainer)
    ├── combo_12_modnet_matting.py                 ✅ full (simplified MODNet w/ pseudo-alpha)
    ├── combo_13_beitv2_l_m2f.py                   ✅ full (timm BEiT-V2-L)
    ├── combo_14_ensemble_dino_sam2_segformer.py   ✅ full (needs 01/03/09 ckpts + CRF)
    ├── combo_15_cascade_sam2_segformer.py         ✅ full (SAM coarse → 4ch SegFormer refiner)
    ├── combo_16_sam_hq.py                         ✅ full (SAM-HQ preferred, SAM v1 fallback)
    ├── combo_17_dinov2_l_unetpp_decoder.py        ✅ full (DINOv2-L + nested-skip UNet++)
    └── combo_18_segnext_l.py                      ✅ full (mscan_large, falls back to ConvNeXt-L)

configs/
├── ga_stage1_model_search.yaml
└── ga_stage2_hyperparam_search.yaml

requirements/
└── ga.txt
```

## Installation

```bash
pip install -r requirements/ga.txt
# Per-adapter optional deps: uncomment relevant lines in ga.txt.
```

## Quick start

```bash
# From project root, always.

# 1. See all combos + which adapters are fully implemented
python -m src.ga.cli list-combos

# 2. Validate config + adapters + environment without running training
python -m src.ga.cli dry-run --config configs/ga_stage1_model_search.yaml

# 3. Preflight (data, GPU, disk)
python -m src.ga.cli preflight --config configs/ga_stage1_model_search.yaml

# 4. Run Stage 1 (model-family search)
python -m src.ga.cli run --config configs/ga_stage1_model_search.yaml

# 5. Pick the winning combo from runs/ga/stage1/hall_of_fame.json; then:
python -m src.ga.cli stage2 01_dinov2_l_m2f  # or whichever key won

# 6. Resume an interrupted run
python -m src.ga.cli resume runs/ga/stage1/ga_checkpoint.pkl \
    --config configs/ga_stage1_model_search.yaml
```

## CLI overrides

Any scalar in the YAML can be overridden with `--set`:

```bash
python -m src.ga.cli run \
    --config configs/ga_stage1_model_search.yaml \
    --set ga.population_size=8 \
    --set ga.generations=5 \
    --set fitness.proxy_epochs=4 \
    --set runtime.parallel_workers=2 \
    --set runtime.per_worker_gpu=[0,1]
```

## Implementing a stub

Each stub currently raises `AdapterNotImplementedError`. To implement one:

1. Open `src/ga/adapters/combo_XX_*.py`.
2. Read the TODO list in the module docstring.
3. Follow the reference pattern in `combo_01_dinov2_l_m2f.py` (native) or `combo_09_segformer_b5_premium.py` (subprocess wrapper).
4. The GA picks it up automatically via the registry — no engine changes needed.

## Adapter contract

Every adapter must:

- Train for `self.fitness_cfg.proxy_epochs` (or `proxy_minutes`).
- Respect `self.fitness_cfg.early_kill_iou` at `early_kill_at_fraction`.
- Save predictions to `self.work_dir / "preds" / *.png` (binary 0/255).
- Return an `AdapterResult` with at minimum `metrics["iou"]`. Ideally also `boundary_f1` and `tv_penalty`.
- Survive crashes gracefully — return `AdapterResult(status="crashed", error="...", metrics={"iou": 0.0})` rather than raising. The subprocess harness wraps exceptions anyway but clean returns make logs readable.

## Outputs

```
runs/ga/stage1/
├── ga.log                      # plaintext file log
├── history.csv                 # per-individual fitness records
├── history.jsonl               # full JSON audit trail
├── hall_of_fame.json           # top 20 at end of run
├── fitness_cache.jsonl         # genome_hash → fitness (memoization)
├── ga_checkpoint.pkl           # resumable state (+ .json sibling for inspection)
├── tb/                         # TensorBoard logs
└── generations/
    └── gen_000/
        └── ind_000_01_dinov2_l_m2f_abc123/
            ├── genome.json
            ├── config.json
            ├── result.json
            ├── subprocess.log
            ├── adapter.log
            ├── preds/*.png
            ├── gt/*.png
            └── decoder.pt (or best_model.pth for subprocess adapters)
```

## Fitness definition

```
composite = iou_weight * IoU
          + boundary_f1_weight * BoundaryF1
          - tv_smoothness_penalty * TotalVariation
```

IoU is mean per-image IoU on val split. BoundaryF1 uses 2-pixel tolerance. TV penalizes blocky/jagged masks (critical — pure IoU can reward coarse predictions that look terrible under the visualizer's alpha-blend).

## SAM 3 note

As of knowledge cutoff (May 2025): not aware of a public SAM 3 release. The registry's combo 03 adapter (`Sam2EncoderMask2FormerAdapter`) is designed so swapping to SAM 3 is a one-line change in the import. Check `facebookresearch/sam2` repo for updates.

## Troubleshooting

**"Cannot import adapter module":** The stub file is missing or import-time error. Check `src/ga/adapters/combo_XX_*.py`.

**"All individuals fitness=-inf":** Likely all adapters are stubs or all crashed. Run `dry-run` first to confirm 3+ full adapters are present.

**Fitness seems wrong / no boundary_f1 reported:** Subprocess-wrapper adapters (09, 10, 11) only parse IoU from log — they don't save pred PNGs. To get full metrics add a `--ga-eval-dump <dir>` flag to the underlying training script and call `base.compute_iou_and_bf1(dump_dir)` in the adapter.

**CUDA OOM:** Reduce `input_size` in the search space, set `ga.combo_pool` to smaller models only, or lower `population_size`.

**GA stuck on one combo:** Raise `niching_quota[combo_key]` to a lower number, or reduce `combo_mutation_rate`.
