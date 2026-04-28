"""Registry of 18 model combos and their per-combo hyperparameter search spaces.

A "combo" is the gene at Stage 1 (architectural choice). Each combo declares:
  - key:       stable short id (used in filenames, logs)
  - name:      human-readable name
  - adapter:   dotted import path to its Adapter class
  - tier:      A/B/C priority — helps bias initial GA population
  - notes:     short description (shown in CLI listings)
  - search_space: dict of hyperparameter name -> sampling spec
                  (used at Stage 2 to mutate this combo's params)

Search space sampling specs:
  {"type": "float",   "low": x, "high": y, "log": True}
  {"type": "int",     "low": x, "high": y}
  {"type": "choice",  "values": [...]}
  {"type": "bool"}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Combo:
    key: str
    name: str
    adapter: str                              # dotted import path
    tier: str                                 # "A" (strong), "B" (mid), "C" (baseline)
    notes: str
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


# Common per-combo defaults that the GA will always search over,
# regardless of architecture. These get merged into each combo's search_space.
UNIVERSAL_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "input_size":        {"type": "choice", "values": [512, 640, 768]},
    "augmentation":      {"type": "choice", "values": ["light", "medium", "aggressive"]},
    "ema_decay":         {"type": "choice", "values": [0.999, 0.9995, 0.9999]},
    "tta_mode":          {"type": "choice", "values": ["none", "hflip", "hflip+vflip"]},
    "post_pipeline":     {"type": "choice", "values": ["thresh", "morph", "bilateral", "crf", "guided"]},
    "threshold":         {"type": "float",  "low": 0.35, "high": 0.65},
    "threshold_softness":{"type": "float",  "low": 0.0,  "high": 0.20},
    "morph_kernel":      {"type": "choice", "values": [0, 1, 3, 5]},
}


COMBOS: list[Combo] = [
    Combo(
        key="01_dinov2_l_m2f",
        name="DINOv2-L (frozen) + Mask2Former decoder",
        adapter="src.ga.adapters.combo_01_dinov2_l_m2f.DinoV2LMask2FormerAdapter",
        tier="A",
        notes="Top pick: self-supervised features discriminate wood-vs-wood textures.",
        search_space={
            "decoder_lr":     {"type": "float",  "low": 1e-5, "high": 5e-4, "log": True},
            "boundary_weight":{"type": "float",  "low": 0.5,  "high": 3.0},
            "num_queries":    {"type": "choice", "values": [50, 100, 200]},
            "mask_feat_dim":  {"type": "choice", "values": [128, 256]},
            "unfreeze_last":  {"type": "int",    "low": 0,    "high": 4},
        },
    ),
    Combo(
        key="02_dinov2_g_upernet",
        name="DINOv2-G (frozen) + UPerNet",
        adapter="src.ga.adapters.combo_02_dinov2_g_upernet.DinoV2GUPerNetAdapter",
        tier="A",
        notes="Ceiling test — DINOv2-Giant (~1.1B params) with simple UPerNet head.",
        search_space={
            "decoder_lr":       {"type": "float",  "low": 5e-6, "high": 2e-4, "log": True},
            "decoder_channels": {"type": "choice", "values": [256, 384, 512]},
            "aux_loss_weight":  {"type": "float",  "low": 0.2,  "high": 0.6},
        },
    ),
    Combo(
        key="03_sam2_encoder_m2f",
        name="SAM 2 Hiera-L encoder (frozen) + Mask2Former decoder",
        adapter="src.ga.adapters.combo_03_sam2_encoder_m2f.Sam2EncoderMask2FormerAdapter",
        tier="A",
        notes="Reuse SAM 2 image encoder, swap in M2F decoder. Substitute SAM 3 when available.",
        search_space={
            "decoder_lr":    {"type": "float",  "low": 1e-5, "high": 5e-4, "log": True},
            "num_queries":   {"type": "choice", "values": [50, 100, 200]},
            "unfreeze_last": {"type": "int",    "low": 0,    "high": 4},
        },
    ),
    Combo(
        key="04_sam2_full_finetune",
        name="SAM 2 full (auto-mask grid prompts, decoder fine-tuned)",
        adapter="src.ga.adapters.combo_04_sam2_full_finetune.Sam2FullFinetuneAdapter",
        tier="A",
        notes="Near-zero-shot baseline with decoder fine-tuning. Strong on thin structures (branches).",
        search_space={
            "points_per_side":       {"type": "choice", "values": [16, 32, 64]},
            "pred_iou_thresh":       {"type": "float",  "low": 0.60, "high": 0.95},
            "stability_score_thresh":{"type": "float",  "low": 0.85, "high": 0.98},
            "decoder_lr":            {"type": "float",  "low": 1e-6, "high": 1e-4, "log": True},
        },
    ),
    Combo(
        key="05_eva02_l_m2f",
        name="EVA-02-L (MIM pretrained) + Mask2Former",
        adapter="src.ga.adapters.combo_05_eva02_l_m2f.Eva02LMask2FormerAdapter",
        tier="A",
        notes="Top of several seg leaderboards. Strong alternative to Swin-V2-L.",
        search_space={
            "lr":              {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "drop_path":       {"type": "float",  "low": 0.0,  "high": 0.4},
            "boundary_weight": {"type": "float",  "low": 0.5,  "high": 3.0},
            "layer_decay":     {"type": "float",  "low": 0.70, "high": 0.95},
        },
    ),
    Combo(
        key="06_internimage_l_upernet",
        name="InternImage-L (DCNv3) + UPerNet",
        adapter="src.ga.adapters.combo_06_internimage_l_upernet.InternImageLUPerNetAdapter",
        tier="B",
        notes="Deformable conv backbone. Architecturally different from ViT family — GA diversity.",
        search_space={
            "lr":         {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "ohem_ratio": {"type": "float",  "low": 0.3,  "high": 0.8},
            "aux_weight": {"type": "float",  "low": 0.2,  "high": 0.6},
        },
    ),
    Combo(
        key="07_swinv2_l_m2f",
        name="Swin-V2-L + Mask2Former + deep supervision",
        adapter="src.ga.adapters.combo_07_swinv2_l_m2f.SwinV2LMask2FormerAdapter",
        tier="A",
        notes="Proven top-tier. Already in project as predict_mask2former_detectron2_swin_v2_l.py.",
        search_space={
            "lr":               {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "layer_decay":      {"type": "float",  "low": 0.70, "high": 0.95},
            "boundary_weight":  {"type": "float",  "low": 0.5,  "high": 3.0},
            "deep_sup_layers":  {"type": "choice", "values": [3, 6, 9]},
        },
    ),
    Combo(
        key="08_convnextv2_l_upernet",
        name="ConvNeXt-V2-L (FCMAE pretrained) + UPerNet",
        adapter="src.ga.adapters.combo_08_convnextv2_l_upernet.ConvNeXtV2LUPerNetAdapter",
        tier="B",
        notes="Pure CNN alternative, FCMAE self-supervised pretraining.",
        search_space={
            "lr":             {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "tversky_alpha":  {"type": "float",  "low": 0.3,  "high": 0.7},
            "tversky_beta":   {"type": "float",  "low": 0.3,  "high": 0.7},
            "drop_path":      {"type": "float",  "low": 0.0,  "high": 0.4},
        },
    ),
    Combo(
        key="09_segformer_b5_premium",
        name="SegFormer-B5 (project PREMIUM trainer)",
        adapter="src.ga.adapters.combo_09_segformer_b5_premium.SegFormerB5PremiumAdapter",
        tier="A",
        notes="Reuses src/training/train_SegFormerB5_PREMIUM.py via subprocess.",
        search_space={
            "lr":              {"type": "float",  "low": 3e-5, "high": 1e-4, "log": True},
            "ohem_ratio":      {"type": "float",  "low": 0.5,  "high": 0.9},
            "edge_weight":     {"type": "float",  "low": 1.0,  "high": 4.0},
            "label_smoothing": {"type": "float",  "low": 0.0,  "high": 0.2},
        },
    ),
    Combo(
        key="10_m2f_segformer_b5",
        name="Mask2Former + SegFormer-B5 backbone (project config, bugs fixed)",
        adapter="src.ga.adapters.combo_10_m2f_segformer_b5.Mask2FormerSegFormerB5Adapter",
        tier="B",
        notes="Reuses src/training/train_Mask2Former.py via subprocess. Retraining required (CRITICAL_ISSUES_FOUND).",
        search_space={
            "lr":                  {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "backbone_lr_mult":    {"type": "float",  "low": 0.01, "high": 0.5, "log": True},
            "class_weight_fence":  {"type": "float",  "low": 5.0,  "high": 25.0},
            "deep_sup_layers":     {"type": "choice", "values": [3, 6, 9]},
        },
    ),
    Combo(
        key="11_unetpp_b7",
        name="UNet++ EfficientNet-B7 (project flagship trainer)",
        adapter="src.ga.adapters.combo_11_unetpp_b7.UNetPPB7Adapter",
        tier="A",
        notes="Reuses src/training/train_UNetPlusPlus.py. Currently unused — deployed ONNX is B1 variant.",
        search_space={
            "lr":                {"type": "float",  "low": 1e-4, "high": 1e-3, "log": True},
            "enc_lr_mult":       {"type": "float",  "low": 0.05, "high": 0.3,  "log": True},
            "focal_weight":      {"type": "float",  "low": 1.0,  "high": 4.0},
            "dice_weight":       {"type": "float",  "low": 1.0,  "high": 3.0},
            "boundary_weight":   {"type": "float",  "low": 0.5,  "high": 3.0},
            "lovasz_weight":     {"type": "float",  "low": 0.5,  "high": 3.0},
            "tversky_weight":    {"type": "float",  "low": 0.3,  "high": 2.0},
            "ssim_weight":       {"type": "float",  "low": 0.1,  "high": 1.5},
            "warmup_epochs":     {"type": "int",    "low": 5,    "high": 30},
        },
    ),
    Combo(
        key="12_modnet_matting",
        name="MODNet / RVM matting specialist",
        adapter="src.ga.adapters.combo_12_modnet_matting.ModNetMattingAdapter",
        tier="A",
        notes="Matting, not segmentation. Produces continuous alpha — best for branch/wire edges.",
        search_space={
            "trimap_dilation": {"type": "int",    "low": 5,   "high": 25},
            "refiner_depth":   {"type": "choice", "values": [3, 5, 7]},
            "se_ratio":        {"type": "float",  "low": 0.125, "high": 0.5},
            "lr":              {"type": "float",  "low": 1e-4, "high": 1e-3, "log": True},
        },
    ),
    Combo(
        key="13_beitv2_l_m2f",
        name="BEiT-V2-L + Mask2Former",
        adapter="src.ga.adapters.combo_13_beitv2_l_m2f.BeitV2LMask2FormerAdapter",
        tier="B",
        notes="MIM-pretrained ViT variant for GA diversity.",
        search_space={
            "lr":         {"type": "float",  "low": 1e-5, "high": 3e-4, "log": True},
            "drop_path":  {"type": "float",  "low": 0.0,  "high": 0.4},
            "layer_decay":{"type": "float",  "low": 0.70, "high": 0.95},
        },
    ),
    Combo(
        key="14_ensemble_dino_sam2_segformer",
        name="Ensemble: DINOv2 + SAM 2 + SegFormer-B5 (weighted avg + CRF)",
        adapter="src.ga.adapters.combo_14_ensemble_dino_sam2_segformer.EnsembleAdapter",
        tier="A",
        notes="Requires combos 01, 03, 09 to have been trained first. Averages masks + DenseCRF.",
        search_space={
            "alpha_dino":    {"type": "float", "low": 0.1, "high": 0.7},
            "alpha_sam2":    {"type": "float", "low": 0.1, "high": 0.7},
            "alpha_b5":      {"type": "float", "low": 0.1, "high": 0.7},
            "crf_iter":      {"type": "int",   "low": 3,   "high": 15},
            "crf_pos_sxy":   {"type": "float", "low": 1.0, "high": 5.0},
            "crf_bilateral":{"type":  "float", "low": 5.0, "high": 20.0},
        },
    ),
    Combo(
        key="15_cascade_sam2_segformer",
        name="Cascade: SAM 2 (coarse) → SegFormer-B5 (refiner)",
        adapter="src.ga.adapters.combo_15_cascade_sam2_segformer.CascadeSam2SegFormerAdapter",
        tier="A",
        notes="Two-stage: SAM 2 prior → B5 refinement with consistency loss.",
        search_space={
            "sam_box_margin":     {"type": "int",   "low": 0,    "high": 30},
            "consistency_weight": {"type": "float", "low": 0.1,  "high": 2.0},
            "refiner_lr":         {"type": "float", "low": 1e-5, "high": 2e-4, "log": True},
        },
    ),
    # === Additional combos (16–18) ===
    Combo(
        key="16_sam_hq",
        name="SAM-HQ (high-quality mask head)",
        adapter="src.ga.adapters.combo_16_sam_hq.SamHQAdapter",
        tier="A",
        notes="SAM variant with HQ output token — specifically designed for sharp edges.",
        search_space={
            "hq_token_weight":  {"type": "float", "low": 0.5,  "high": 3.0},
            "lr":               {"type": "float", "low": 1e-5, "high": 2e-4, "log": True},
            "freeze_prompt_enc":{"type": "bool"},
        },
    ),
    Combo(
        key="17_dinov2_l_unetpp_decoder",
        name="DINOv2-L (frozen) + UNet++ decoder (best encoder × best decoder)",
        adapter="src.ga.adapters.combo_17_dinov2_l_unetpp_decoder.DinoV2LUNetPPAdapter",
        tier="A",
        notes="Decoder ablation for #1: does UNet++ decoder beat M2F on DINOv2 features?",
        search_space={
            "decoder_lr":      {"type": "float", "low": 1e-5, "high": 5e-4, "log": True},
            "decoder_channels":{"type": "choice", "values": [(256,128,64,32,16),(512,256,128,64,32)]},
            "attention_type":  {"type": "choice", "values": ["scse", "cbam", None]},
        },
    ),
    Combo(
        key="18_segnext_l_mscan",
        name="SegNeXt-L (MSCAN backbone)",
        adapter="src.ga.adapters.combo_18_segnext_l.SegNeXtLAdapter",
        tier="B",
        notes="Multi-scale attention conv backbone. Recent strong SOTA, architectural diversity.",
        search_space={
            "lr":              {"type": "float", "low": 1e-5, "high": 3e-4, "log": True},
            "ham_channels":    {"type": "choice", "values": [256, 512]},
            "drop_path":       {"type": "float", "low": 0.0,  "high": 0.4},
        },
    ),
]


def get_combo(key: str) -> Combo:
    for c in COMBOS:
        if c.key == key:
            return c
    raise KeyError(f"Unknown combo key: {key}. Available: {[c.key for c in COMBOS]}")


def all_keys() -> list[str]:
    return [c.key for c in COMBOS]


def keys_by_tier(tier: str) -> list[str]:
    return [c.key for c in COMBOS if c.tier == tier]


def get_full_search_space(key: str) -> dict[str, dict[str, Any]]:
    """Combo-specific search space + universal params."""
    combo = get_combo(key)
    merged = dict(UNIVERSAL_SEARCH_SPACE)
    merged.update(combo.search_space)
    return merged
