"""Template for adapter stubs. Copy this pattern when implementing each stub.

Stub adapters intentionally fail fast with AdapterNotImplementedError so the GA
doesn't pretend to evaluate models whose training isn't wired up yet.

Implementation checklist for turning a stub into a real adapter:
  1. Import backbone (torch.hub / timm / HuggingFace / custom).
  2. Freeze / unfreeze per genome.params["unfreeze_last"] (if applicable).
  3. Build decoder/head.
  4. Build Dataset + DataLoader using self.data_cfg.
  5. Run proxy training for self.fitness_cfg.proxy_epochs.
  6. Save predictions to self.work_dir / "preds".
  7. Call self.compute_iou_and_bf1(preds_dir) for metrics.
  8. Save best checkpoint → return AdapterResult(..., ckpt_path=...).
  9. Respect early_kill_iou at early_kill_at_fraction of the budget.

See combo_01_dinov2_l_m2f.py for a full reference implementation.
See combo_09_segformer_b5_premium.py and combo_11_unetpp_b7.py for subprocess-based variants
that reuse existing project trainers.
"""
