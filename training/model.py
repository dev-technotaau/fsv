"""training/model.py — DINOv2/v3 backbone + Mask2Former-style decoder + refinement head.

Architecture:
    image (B, 3, H, W)
        |
        +--> DINO backbone (ViT, returns sequence of patch tokens)
        |       outputs: (B, N_patches+1, dim)
        |       patch stride = 14 (DINOv2) or 16 (DINOv3)
        |
        +--> ViT-to-FPN adapter (reshape patches to spatial grid; build 4-scale FPN)
        |       outputs: {res1: B,C,H/4,W/4   res2: B,C,H/8,W/8
        |                 res3: B,C,H/16,W/16  res4: B,C,H/32,W/32}
        |
        +--> Pixel decoder (lightweight FPN-style upsample with skips)
        |       outputs: pixel_features (B, C, H/4, W/4)
        |                multi_scale_features (3 maps for transformer decoder)
        |
        +--> Mask2Former-style transformer decoder
        |       N learnable queries cross-attend to multi-scale features
        |       outputs: query_embeddings (B, num_queries, C)
        |
        +--> Mask head: dot product of (queries, pixel_features) -> per-query masks
        |       For binary seg: take max-confidence query (or sum) -> single mask logit
        |
        +--> Optional learnable refinement head: small UNet over [image, coarse_mask]
        |       outputs: refined_mask (B, 1, H, W)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Backbone wrapper around DINOv2 / DINOv3 (HuggingFace transformers)
# ══════════════════════════════════════════════════════════════════════

class DinoBackbone(nn.Module):
    """Wraps DINOv2 or DINOv3 ViT models from HF transformers.

    Supported names (examples):
        DINOv2: facebook/dinov2-{small,base,large,giant}
        DINOv3: facebook/dinov3-{vits16,vitb16,vitl16,vith16plus,vit7b16}
                -pretrain-lvd1689m

    Patch size is auto-detected from the model config (DINOv2 = 14, DINOv3 = 16).

    Returns the sequence of last-hidden-state patch tokens reshaped back to a
    spatial (B, C, H', W') grid.

    NOTE: DINOv3 weights are gated on Hugging Face. Run `huggingface-cli login`
    once and accept the license at the model's HF page before first use.
    """

    def __init__(self, name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                 freeze_first_n_layers: int = 0,
                 drop_path_rate: float = 0.1,
                 multi_block_n: int = 1,
                 aggregation_type: str = "weighted_sum",
                 return_global_tokens: bool = True) -> None:
        """multi_block_n: how many of the LAST transformer blocks to fuse for
            dense features. 1 = original behavior (last block only). 4 is the
            DINOv3 paper's recommendation for dense tasks.
        aggregation_type: 'sum' | 'concat_proj' | 'weighted_sum' (learnable).
        return_global_tokens: if True, the forward returns CLS + register tokens
            alongside spatial features so the decoder can attend to global context.
        """
        super().__init__()
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers") from e

        try:
            cfg = AutoConfig.from_pretrained(name)
        except Exception as e:
            self._raise_friendly_load_error(name, e, stage="config")
        # drop_path_rate isn't a standard kwarg for all variants; try to set
        try:
            cfg.drop_path_rate = drop_path_rate
        except Exception:
            pass

        try:
            self.model = AutoModel.from_pretrained(name, config=cfg)
        except Exception as e:
            self._raise_friendly_load_error(name, e, stage="weights")
        self.dim = int(getattr(self.model.config, "hidden_size", 1024))
        self.name = name

        # Auto-detect patch size from config (14 for DINOv2, 16 for DINOv3)
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))

        # DINOv3 has 4 register tokens between CLS and patch tokens; DINOv2 has 0.
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0)
        )

        # Multi-block aggregation setup (DINOv3 paper recommends fusing the last
        # 4 blocks for dense prediction tasks — meaningfully better than just
        # the final block's output).
        self.multi_block_n = max(1, int(multi_block_n))
        self.aggregation_type = aggregation_type
        self.return_global_tokens = return_global_tokens
        self._aggregate_proj: Optional[nn.Module] = None
        self._aggregate_weights: Optional[nn.Parameter] = None
        if self.multi_block_n > 1:
            if aggregation_type == "concat_proj":
                # Concat along channel dim then project back to self.dim
                self._aggregate_proj = nn.Linear(self.dim * self.multi_block_n,
                                                   self.dim)
            elif aggregation_type == "weighted_sum":
                # Learnable per-layer scalar weights, softmax-normalized
                self._aggregate_weights = nn.Parameter(
                    torch.zeros(self.multi_block_n)
                )

        # Optionally freeze the first N transformer blocks
        if freeze_first_n_layers > 0:
            self._freeze_first_n_layers(freeze_first_n_layers)

    @staticmethod
    def _raise_friendly_load_error(name: str, e: Exception, stage: str) -> None:
        """Convert opaque HF errors (401 gated, 404 wrong name, network) into
        actionable messages. The DINOv3 family is gated, which is by far the
        most common first-run failure."""
        msg = str(e)
        is_gated = ("401" in msg or "gated" in msg.lower() or "Forbidden" in msg
                    or "access" in msg.lower() and "denied" in msg.lower())
        is_404 = ("404" in msg or "not found" in msg.lower()
                  or "does not exist" in msg.lower())
        prefix = f"[backbone:{stage}] failed to load '{name}': "
        if is_gated and "dinov3" in name.lower():
            raise RuntimeError(
                prefix + "this DINOv3 weight is gated. Run the following ONCE:\n"
                "    huggingface-cli login\n"
                f"  then visit https://huggingface.co/{name}\n"
                "  and click 'Agree and access repository'. Then retry training.\n"
                f"  Original error: {type(e).__name__}: {msg[:160]}"
            ) from e
        if is_404:
            raise RuntimeError(
                prefix + f"model id not found on HuggingFace.\n"
                "  Check spelling — DINOv3 names look like\n"
                "  'facebook/dinov3-vith16plus-pretrain-lvd1689m'.\n"
                f"  Original error: {type(e).__name__}: {msg[:160]}"
            ) from e
        # Re-raise as-is for unknown errors (preserves the original chain).
        raise e

    def _get_layers(self):
        """Return the encoder's transformer-layer ModuleList. DINOv2 calls it
        `layer`, DINOv3 calls it `layers`. Returns None if neither found."""
        return (getattr(self.model.encoder, "layer", None)
                or getattr(self.model.encoder, "layers", None))

    def _freeze_first_n_layers(self, n: int) -> int:
        """Freeze patch embedding + first N transformer blocks.
        Returns how many layers were actually frozen (0 if structure not found)."""
        # Freeze patch embedding always
        for p in self.model.embeddings.parameters():
            p.requires_grad = False
        layers = self._get_layers()
        if layers is None:
            return 0
        frozen = 0
        for i, layer in enumerate(layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False
                frozen += 1
        return frozen

    def enable_gradient_checkpointing(self) -> bool:
        """Activate HF's built-in gradient checkpointing on the backbone.
        Halves activation memory for the backbone at ~25% slower bwd. Critical
        for training H+ at 1024px. Returns True on success."""
        try:
            self.model.gradient_checkpointing_enable()
            return True
        except (AttributeError, ValueError):
            return False

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W).

        Returns:
            If `multi_block_n == 1` and `return_global_tokens == False`:
                Tensor (B, C, H', W')  (back-compatible path)
            Otherwise:
                dict with:
                    'spatial':   (B, C, H', W')   fused patch tokens
                    'cls':       (B, C)           CLS token (None if absent)
                    'registers': (B, R, C)        register tokens (None if R == 0)
        """
        B, _, H, W = x.shape
        ps = self.patch_size
        Hp = (H + ps - 1) // ps * ps
        Wp = (W + ps - 1) // ps * ps
        if (Hp, Wp) != (H, W):
            x = F.pad(x, (0, Wp - W, 0, Hp - H))

        need_hidden = self.multi_block_n > 1
        out = self.model(pixel_values=x, return_dict=True,
                          output_hidden_states=need_hidden)

        skip = 1 + self.num_register_tokens
        if need_hidden:
            # hidden_states is a tuple of (B, T, C), one per layer + initial embed
            # Take the LAST `multi_block_n` blocks (most semantic features)
            hs = out.hidden_states[-self.multi_block_n:]
            patches_per_layer = [h[:, skip:, :] for h in hs]
            if self.aggregation_type == "sum":
                fused = sum(patches_per_layer)
            elif self.aggregation_type == "weighted_sum":
                # softmax-normalized learnable weights — sum to 1, no extra scale
                w = torch.softmax(self._aggregate_weights, dim=0)
                fused = sum(p * w[i] for i, p in enumerate(patches_per_layer))
            elif self.aggregation_type == "concat_proj":
                cat = torch.cat(patches_per_layer, dim=-1)        # (B, T-skip, C*N)
                fused = self._aggregate_proj(cat)                  # (B, T-skip, C)
            else:
                raise ValueError(
                    f"Unknown aggregation_type: {self.aggregation_type}"
                )
            patch_seq = fused                                       # (B, T-skip, C)
            cls_token = (out.hidden_states[-1][:, 0, :]
                          if self.return_global_tokens else None)
            register_tokens = (out.hidden_states[-1][:, 1:1 + self.num_register_tokens, :]
                                if (self.return_global_tokens
                                    and self.num_register_tokens > 0) else None)
        else:
            patch_seq = out.last_hidden_state[:, skip:, :]          # (B, T-skip, C)
            cls_token = (out.last_hidden_state[:, 0, :]
                          if self.return_global_tokens else None)
            register_tokens = (
                out.last_hidden_state[:, 1:1 + self.num_register_tokens, :]
                if (self.return_global_tokens and self.num_register_tokens > 0)
                else None
            )

        n_patches = patch_seq.shape[1]
        h_feat = Hp // ps
        w_feat = Wp // ps
        if h_feat * w_feat != n_patches:
            h_feat = w_feat = int(math.sqrt(n_patches))
        spatial = (patch_seq.transpose(1, 2).contiguous()
                   .view(B, self.dim, h_feat, w_feat))

        # Back-compat: when no multi-block AND no global tokens, return just the
        # tensor (so old call sites keep working).
        if self.multi_block_n == 1 and not self.return_global_tokens:
            return spatial
        return {"spatial": spatial, "cls": cls_token, "registers": register_tokens}


# Back-compat alias
DinoV2Backbone = DinoBackbone


# ══════════════════════════════════════════════════════════════════════
# DINOv3 backbone WITH ViT-Adapter (Chen et al., ICLR 2023)
# ══════════════════════════════════════════════════════════════════════

class DinoBackboneWithAdapter(nn.Module):
    """DINOv3 backbone wrapped with the ViT-Adapter recipe — produces a clean
    multi-scale feature pyramid {res2, res3, res4, res5} suitable for any dense
    prediction head, replacing the homemade ViTToFPN adapter.

    The DINOv3 ViT processes patch tokens in parallel with a small CNN
    (SpatialPriorModule). At chosen interaction indexes (default: 4 stages,
    evenly spaced across ViT depth), the two streams exchange information via
    deformable cross-attention:
      - Extractor: ViT tokens refine the multi-scale pyramid
      - Injector:  multi-scale pyramid refines ViT tokens

    Forward output:
      dict with:
        'multi_scale': {res2, res3, res4, res5}  high-quality 4-scale pyramid
                       at strides (4, 8, 16, 32). Goes directly into the
                       downstream pixel decoder.
        'spatial':     (B, C, H/16, W/16)  ViT spatial features (back-compat)
        'cls':         (B, C)              CLS token  | None if disabled
        'registers':   (B, R, C)           register tokens  | None if R==0

    Backbone weights are KEPT (still pretrained DINOv3-H+, still fine-tuned).
    The SPM + interaction modules are NEW (random init, learn from scratch).
    """

    def __init__(self, name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                 freeze_first_n_layers: int = 0,
                 drop_path_rate: float = 0.1,
                 multi_block_n: int = 4,
                 aggregation_type: str = "weighted_sum",
                 return_global_tokens: bool = True,
                 # ── ViT-Adapter knobs ──
                 adapter_inplanes: int = 64,
                 adapter_n_heads: int = 8,
                 adapter_n_points: int = 4,
                 adapter_n_interactions: int = 4,
                 adapter_init_values: float = 0.0,
                 adapter_ffn_ratio: float = 0.25) -> None:
        super().__init__()
        # Reuse DinoBackbone's construction: backbone weights, multi-block agg
        # config, register-token bookkeeping. We then bypass its forward and
        # drive the encoder layer-by-layer ourselves.
        self.bb = DinoBackbone(
            name=name,
            freeze_first_n_layers=freeze_first_n_layers,
            drop_path_rate=drop_path_rate,
            multi_block_n=multi_block_n,
            aggregation_type=aggregation_type,
            return_global_tokens=return_global_tokens,
        )
        self.dim = self.bb.dim
        self.patch_size = self.bb.patch_size
        self.num_register_tokens = self.bb.num_register_tokens
        self.name = name
        self.return_global_tokens = return_global_tokens
        self.multi_block_n = self.bb.multi_block_n
        self.aggregation_type = self.bb.aggregation_type

        # ── ViT-Adapter modules ──
        # Lazy import to avoid hard dependency for non-adapter code paths.
        from training.vit_adapter import (
            SpatialPriorModule, InteractionBlock,
            flatten_pyramid, unflatten_pyramid,
            _build_reference_points, _build_vit_reference_points,
        )
        self._flatten_pyramid = flatten_pyramid
        self._unflatten_pyramid = unflatten_pyramid
        self._build_pyr_ref = _build_reference_points
        self._build_vit_ref = _build_vit_reference_points

        self.spm = SpatialPriorModule(in_channels=3, embed_dim=self.dim,
                                        inplanes=adapter_inplanes)

        # Locate the encoder's transformer-block ModuleList. The HF DINOv3 model
        # stores them under `m.model.layer` (32 blocks for H+).
        self._n_blocks = len(self._encoder_layers())
        if adapter_n_interactions < 1 or adapter_n_interactions > self._n_blocks:
            raise ValueError(
                f"adapter_n_interactions={adapter_n_interactions} out of range "
                f"for {self._n_blocks}-block backbone"
            )
        # Distribute interactions evenly across depth (last block of each
        # quartile by default). Indexes here are 0-based block positions AFTER
        # which the interaction runs.
        step = self._n_blocks // adapter_n_interactions
        self.interaction_indexes = [
            (i + 1) * step - 1 for i in range(adapter_n_interactions)
        ]
        self.interactions = nn.ModuleList([
            InteractionBlock(
                embed_dim=self.dim, num_heads=adapter_n_heads,
                n_points=adapter_n_points, init_values=adapter_init_values,
                ffn_ratio=adapter_ffn_ratio,
            ) for _ in range(adapter_n_interactions)
        ])

        # Final projection: ViT spatial features (after norm) at stride 16
        # are added to the pyramid's c3 (also stride 16) — fuses both streams
        # at the matched scale. Light norm to keep magnitudes consistent.
        self.norm_c3_fuse = nn.LayerNorm(self.dim)

    # ── HF backbone access helpers ──────────────────────────────────────

    def _encoder_layers(self):
        """Locate the transformer-block ModuleList in the HF DINOv3 model.
        DINOv3 uses `m.model.layer` ; DINOv2 used `m.encoder.layer` ."""
        m = self.bb.model
        # DINOv3
        if hasattr(m, "model") and hasattr(m.model, "layer"):
            return m.model.layer
        # Fallback for DINOv2
        if hasattr(m, "encoder"):
            for attr in ("layer", "layers"):
                layers = getattr(m.encoder, attr, None)
                if layers is not None:
                    return layers
        raise AttributeError(
            "Could not locate transformer-block ModuleList on backbone "
            f"{type(m).__name__}"
        )

    def enable_gradient_checkpointing(self) -> bool:
        """Forward to the inner backbone."""
        return self.bb.enable_gradient_checkpointing()

    # ── Manual forward with adapter interactions ──────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        ps = self.patch_size
        # Pad image to a multiple of patch_size (HF DINOv3 will otherwise crash)
        Hp = (H + ps - 1) // ps * ps
        Wp = (W + ps - 1) // ps * ps
        if (Hp, Wp) != (H, W):
            x = F.pad(x, (0, Wp - W, 0, Hp - H))

        # 1. Spatial Prior Module (parallel CNN) — strides (4, 8, 16, 32)
        c1, c2, c3, c4 = self.spm(x)   # all (B, dim, H_l, W_l)

        # 2. Run DINOv3 embeddings + RoPE (matches HF's internal preamble)
        embed = self.bb.model.embeddings(pixel_values=x)
        rope_pos = self.bb.model.rope_embeddings(pixel_values=x)
        skip = 1 + self.num_register_tokens   # CLS + R register tokens

        # ViT spatial grid
        H_p = Hp // ps
        W_p = Wp // ps

        # Pre-compute reference points on the right device/dtype (rebuilt per
        # forward because the spatial shapes depend on the input size).
        device = x.device
        ref_dtype = torch.float32
        # Pyramid-side ref points (extractor queries → 1 ViT level)
        # When extracting, queries are pyramid tokens (c2/c3/c4). Each query
        # has 1 reference point that we bias toward its own normalized location
        # in the ViT grid.
        pyr_shapes = [(c2.shape[-2], c2.shape[-1]),
                       (c3.shape[-2], c3.shape[-1]),
                       (c4.shape[-2], c4.shape[-1])]
        pyr_ref_for_extract = self._build_pyr_ref(pyr_shapes, device, ref_dtype
                                                     ).unsqueeze(0).expand(B, -1, -1, -1)
        # Extractor queries attend to the SINGLE ViT level, so collapse the
        # n_levels dim from 3 to 1 by taking only the first column (all 3 are
        # identical for self-position references).
        pyr_ref_for_extract = pyr_ref_for_extract[..., :1, :]   # (B, N_pyr, 1, 2)

        # ViT-side ref points (injector queries → 3 pyramid levels)
        vit_ref_for_inject = self._build_vit_ref(H_p, W_p, n_levels=3,
                                                    device=device, dtype=ref_dtype
                                                    ).unsqueeze(0).expand(B, -1, -1, -1)

        # Flatten pyramid to a single token sequence for the interactions
        pyr_tokens, pyr_shapes_kept = self._flatten_pyramid(c2, c3, c4)

        # 3. Iterate ViT blocks, interleaving with InteractionBlocks
        layers = self._encoder_layers()
        # For multi-block aggregation we collect intermediate hidden states
        collect_hidden = self.multi_block_n > 1
        collected_hidden: list[torch.Tensor] = []
        ic_idx = 0
        for li, layer in enumerate(layers):
            embed = layer(embed, position_embeddings=rope_pos)
            if collect_hidden:
                collected_hidden.append(embed)
            if li in self.interaction_indexes:
                # Split CLS+register from patch tokens
                vit_patches = embed[:, skip:]
                vit_patches, pyr_tokens = self.interactions[ic_idx](
                    vit_patches=vit_patches,
                    pyramid_tokens=pyr_tokens,
                    vit_shape=(H_p, W_p),
                    pyramid_shapes=pyr_shapes_kept,
                    vit_reference=vit_ref_for_inject,
                    pyramid_reference=pyr_ref_for_extract,
                )
                # Reassemble full token sequence
                embed = torch.cat([embed[:, :skip], vit_patches], dim=1)
                ic_idx += 1

        # 4. Final norm
        embed = self.bb.model.norm(embed)

        # 5. Multi-block aggregation — re-run the same logic as DinoBackbone
        # but on our manually-collected hidden states.
        if collect_hidden and self.multi_block_n > 1:
            hs = collected_hidden[-self.multi_block_n:]
            patches_per_layer = [h[:, skip:, :] for h in hs]
            if self.aggregation_type == "sum":
                fused = sum(patches_per_layer)
            elif self.aggregation_type == "weighted_sum":
                w = torch.softmax(self.bb._aggregate_weights, dim=0)
                fused = sum(p * w[i] for i, p in enumerate(patches_per_layer))
            elif self.aggregation_type == "concat_proj":
                cat = torch.cat(patches_per_layer, dim=-1)
                fused = self.bb._aggregate_proj(cat)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation_type}")
            patch_seq = fused
        else:
            patch_seq = embed[:, skip:, :]

        # CLS + register tokens come from the FINAL layer's output (after norm)
        cls_token = embed[:, 0, :] if self.return_global_tokens else None
        register_tokens = (embed[:, 1:1 + self.num_register_tokens, :]
                            if (self.return_global_tokens
                                and self.num_register_tokens > 0) else None)

        # 6. Reshape ViT patch tokens to spatial grid (B, C, H_p, W_p)
        spatial = (patch_seq.transpose(1, 2).contiguous()
                    .view(B, self.dim, H_p, W_p))

        # 7. Unflatten the (now-refined) pyramid back to per-scale tensors
        c2_ref, c3_ref, c4_ref = self._unflatten_pyramid(pyr_tokens,
                                                            pyr_shapes_kept)

        # 8. Fuse ViT spatial (stride 16) into c3 (also stride 16) via add.
        #    This brings together the global-reasoning ViT features and the
        #    local-detail-rich pyramid at the same scale.
        # Norm + residual to keep magnitudes balanced.
        spatial_norm = (self.norm_c3_fuse(spatial.flatten(2).transpose(1, 2))
                        ).transpose(1, 2).view_as(spatial)
        if c3_ref.shape[-2:] != spatial_norm.shape[-2:]:
            spatial_norm = F.interpolate(spatial_norm, size=c3_ref.shape[-2:],
                                          mode="bilinear", align_corners=False)
        c3_fused = c3_ref + spatial_norm

        return {
            # Multi-scale pyramid output for the downstream pixel decoder.
            # Channels: all `self.dim` (1280 for H+).
            "multi_scale": {
                "res2": c1,         # stride 4
                "res3": c2_ref,     # stride 8
                "res4": c3_fused,   # stride 16  (ViT + pyramid)
                "res5": c4_ref,     # stride 32
            },
            # Back-compat fields — current pipeline may consume these too
            "spatial": spatial,
            "cls": cls_token,
            "registers": register_tokens,
        }


# ══════════════════════════════════════════════════════════════════════
# ViT-to-FPN adapter — turn single-scale ViT features into 4-scale pyramid
# ══════════════════════════════════════════════════════════════════════

class ViTToFPN(nn.Module):
    """Convert ViT (single-scale) features into a 4-scale FPN.

    Roughly follows ViTDet (Li et al., 2022): the same ViT features are
    reshaped + sampled at 4 different strides via simple deconv/conv ops.
    Cheap and effective.
    """
    def __init__(self, in_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        # res4 (coarsest) — direct downsample 2x from ViT stride 14 -> ~28
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        # res3 = ViT scale (~stride 14)
        # res2 = upsample 2x (stride 7)
        # res1 = upsample 4x (stride ~3-4)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def forward(self, vit_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.proj(vit_feats)
        return {
            "res2": self.up4(c),     # ~stride 4
            "res3": self.up2(c),     # ~stride 7
            "res4": c,               # ~stride 14
            "res5": self.down2(c),   # ~stride 28
        }


# ══════════════════════════════════════════════════════════════════════
# Pixel decoder — FPN-style upsampling for high-res mask features
# ══════════════════════════════════════════════════════════════════════

class PixelDecoder(nn.Module):
    """Lightweight FPN that upsamples the multi-scale features to a single
    high-resolution feature map (used as per-pixel mask embedding)."""
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for _ in range(4)
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(32, dim),
                nn.GELU(),
            ) for _ in range(4)
        ])

    def forward(self, fpn: dict[str, torch.Tensor]) -> torch.Tensor:
        # Top-down pathway: start from coarsest, progressively add upsampled features
        f5 = self.lateral[3](fpn["res5"])
        f4 = self.lateral[2](fpn["res4"]) + F.interpolate(f5, size=fpn["res4"].shape[-2:], mode="bilinear", align_corners=False)
        f4 = self.smooth[2](f4)
        f3 = self.lateral[1](fpn["res3"]) + F.interpolate(f4, size=fpn["res3"].shape[-2:], mode="bilinear", align_corners=False)
        f3 = self.smooth[1](f3)
        f2 = self.lateral[0](fpn["res2"]) + F.interpolate(f3, size=fpn["res2"].shape[-2:], mode="bilinear", align_corners=False)
        f2 = self.smooth[0](f2)
        return f2   # high-res per-pixel features (B, C, H/4-ish, W/4-ish)


# ══════════════════════════════════════════════════════════════════════
# Multi-Scale Deformable Attention pixel decoder (real M2F path)
# ══════════════════════════════════════════════════════════════════════

class MSDeformPixelDecoder(nn.Module):
    """Mask2Former's actual pixel decoder: stacks N deformable-attention
    encoder layers over flattened multi-scale features.

    Wraps Hugging Face's `Mask2FormerPixelDecoderEncoderLayer` so we don't
    need to compile the custom CUDA op — HF ships a PyTorch fallback path
    that works everywhere (slower than the CUDA kernel but still much better
    than naïve attention thanks to deformable sparse sampling).

    Inputs:
        fpn dict with res2 (high-res), res3, res4, res5 feature maps,
        all at the same channel count.

    Outputs:
        same dict layout but with res3/res4/res5 enhanced by N deformable-
        encoder layers, and res2 = res2_lateral + upsample(enhanced_res3).
    """

    def __init__(self, dim: int = 256, num_layers: int = 6,
                 num_heads: int = 8, n_points: int = 4,
                 feedforward_dim: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        # Lazy import — only fail if MSDeform path is actually used
        from transformers.models.mask2former.modeling_mask2former import (
            Mask2FormerPixelDecoderEncoderLayer,
        )
        from transformers.models.mask2former.configuration_mask2former import (
            Mask2FormerConfig,
        )

        # Synthesize a minimal Mask2FormerConfig the encoder layer expects.
        # n_levels=3 and n_points=4 are HARDCODED inside the HF layer, so we
        # ALWAYS run with exactly 3 input scales (res3, res4, res5).
        m2f_cfg = Mask2FormerConfig(
            feature_size=dim,
            encoder_feedforward_dim=feedforward_dim,
            encoder_layers=num_layers,
            num_attention_heads=num_heads,
            dropout=dropout,
        )
        self.layers = nn.ModuleList([
            Mask2FormerPixelDecoderEncoderLayer(m2f_cfg)
            for _ in range(num_layers)
        ])
        self.dim = dim
        self.num_heads = num_heads

        # Per-scale level embedding (added to features so the deformable
        # attention can disambiguate which scale a token came from).
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.level_embed, std=0.02)

        # High-res output: lateral conv from res2 + upsample of enhanced res3
        self.res2_lateral = nn.Conv2d(dim, dim, 1)
        self.res2_smooth = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )

    @staticmethod
    def _build_2d_sin_pe(C: int, H: int, W: int, device, dtype) -> torch.Tensor:
        """Sinusoidal 2D positional encoding for one feature map.
        Returns (1, H*W, C) — flat to be added to the token sequence."""
        pe = torch.zeros(C, H, W, device=device, dtype=dtype)
        d_half = max(1, C // 4)
        y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
        x = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)
        div = torch.exp(torch.arange(0, d_half, dtype=dtype, device=device)
                        * -(math.log(10000.0) / d_half))
        pe[0:d_half, :, :] = torch.sin(y * div.view(-1, 1, 1))
        pe[d_half:2 * d_half, :, :] = torch.cos(y * div.view(-1, 1, 1))
        pe[2 * d_half:3 * d_half, :, :] = torch.sin(x * div.view(-1, 1, 1))
        pe[3 * d_half:4 * d_half, :, :] = torch.cos(x * div.view(-1, 1, 1))
        return pe.flatten(1).transpose(0, 1).unsqueeze(0)         # (1, H*W, C)

    @staticmethod
    def _get_reference_points(spatial_shapes_list, B: int, device, dtype):
        """Build per-token reference points in normalized [0,1] coords.
        Returns (B, N_total, n_levels, 2)."""
        ref_points_list = []
        for H, W in spatial_shapes_list:
            y = torch.linspace(0.5 / H, 1 - 0.5 / H, H, dtype=dtype, device=device)
            x = torch.linspace(0.5 / W, 1 - 0.5 / W, W, dtype=dtype, device=device)
            ry, rx = torch.meshgrid(y, x, indexing="ij")
            ref = torch.stack([rx.flatten(), ry.flatten()], dim=-1)  # (H*W, 2)
            ref_points_list.append(ref)
        ref = torch.cat(ref_points_list, dim=0)                   # (N_total, 2)
        ref = ref[None, :, None, :].expand(B, -1, len(spatial_shapes_list), -1)
        return ref.contiguous()

    def forward(self, fpn: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Use res3 (finest of the 3 transformer-fed scales), res4, res5
        scale_keys = ["res3", "res4", "res5"]
        feats = [fpn[k] for k in scale_keys]
        B = feats[0].shape[0]
        device, dtype = feats[0].device, feats[0].dtype

        # Flatten + concatenate, building auxiliary index tensors
        flat_seq = []
        pe_seq = []
        spatial_shapes_list = []
        for li, f in enumerate(feats):
            _, _, H, W = f.shape
            spatial_shapes_list.append((H, W))
            f_flat = f.flatten(2).transpose(1, 2)                  # (B, H*W, C)
            f_flat = f_flat + self.level_embed[li].view(1, 1, -1)
            flat_seq.append(f_flat)
            pe_seq.append(self._build_2d_sin_pe(self.dim, H, W, device, dtype))
        flat = torch.cat(flat_seq, dim=1)                          # (B, N_total, C)
        pe = torch.cat(pe_seq, dim=1).to(dtype=dtype)              # (1, N_total, C)
        pe = pe.expand(B, -1, -1)

        spatial_shapes_t = torch.tensor(spatial_shapes_list, dtype=torch.long,
                                          device=device)
        level_start_index = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            spatial_shapes_t.prod(1).cumsum(0)[:-1],
        ])
        ref_points = self._get_reference_points(spatial_shapes_list, B, device, dtype)

        # Run through the deformable encoder layers
        x = flat
        for layer in self.layers:
            out = layer(
                hidden_states=x,
                attention_mask=None,
                position_embeddings=pe,
                reference_points=ref_points,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                output_attentions=False,
            )
            # HF encoder layers return a tuple; first element is hidden states
            x = out[0] if isinstance(out, tuple) else out

        # Unflatten back to per-scale spatial maps
        out_per_scale = {}
        cursor = 0
        for li, (H, W) in enumerate(spatial_shapes_list):
            n = H * W
            tok = x[:, cursor:cursor + n, :]
            spatial = tok.transpose(1, 2).reshape(B, self.dim, H, W)
            out_per_scale[scale_keys[li]] = spatial.contiguous()
            cursor += n

        # High-res res2 output: lateral from raw res2 + upsample(enhanced res3)
        r2_lat = self.res2_lateral(fpn["res2"])
        r3_up = F.interpolate(out_per_scale["res3"], size=r2_lat.shape[-2:],
                                mode="bilinear", align_corners=False)
        res2_out = self.res2_smooth(r2_lat + r3_up)

        return {
            "res2": res2_out,
            "res3": out_per_scale["res3"],
            "res4": out_per_scale["res4"],
            "res5": out_per_scale["res5"],
        }


# ══════════════════════════════════════════════════════════════════════
# Mask2Former-style transformer decoder
# ══════════════════════════════════════════════════════════════════════

class TransformerDecoderLayer(nn.Module):
    """One layer of cross-attention (queries -> features) + self-attention + FFN.

    Supports Mask2Former-style MASKED cross-attention: when an attn_mask is
    provided, each query only attends to memory positions where the mask is
    True (or non -inf). This is the defining feature of Mask2Former — it
    forces queries to focus on their predicted region, which sharpens
    convergence and improves boundary quality.
    """
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0,
                 ff_mult: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor,
                 attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q: (B, Q, C). mem: (B, S, C).
        attn_mask: (B*n_heads, Q, S) bool — True positions are MASKED (no attn).
                    None = full attention.
        """
        # Cross-attention: queries attend to memory (flat features), optionally masked
        attn_out = self.cross_attn(self.norm1(q), mem, mem, attn_mask=attn_mask,
                                     need_weights=False)[0]
        q = q + self.dropout(attn_out)
        # Self-attention among queries (no mask)
        q = q + self.dropout(self.self_attn(self.norm2(q), self.norm2(q),
                                                self.norm2(q),
                                                need_weights=False)[0])
        # FFN
        q = q + self.dropout(self.ff(self.norm3(q)))
        return q


class Mask2FormerLikeDecoder(nn.Module):
    """Mask2Former-style decoder with all 5 enhancements:

      1. Per-layer mask prediction (drives masked attention + deep supervision)
      2. Masked cross-attention (queries only attend to their predicted region)
      3. Global tokens (CLS + register tokens appended as always-attended memory)
      4. Deep supervision (every layer's mask is returned; loss is applied to all)
      5. Iterative refinement (mask predictions from layer N drive attention at N+1)
    """
    def __init__(self, dim: int = 256, num_queries: int = 16,
                 num_layers: int = 6, dropout: float = 0.0,
                 use_masked_attention: bool = True,
                 use_global_tokens: bool = True,
                 mask_attention_threshold: float = 0.5,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, dropout=dropout, n_heads=n_heads)
            for _ in range(num_layers)
        ])
        # Per-query class score (binary: just one logit per query)
        self.class_head = nn.Linear(dim, 1)
        self.use_masked_attention = use_masked_attention
        self.use_global_tokens = use_global_tokens
        self.mask_attention_threshold = mask_attention_threshold
        self.n_heads = n_heads
        # LayerNorm for global tokens — keeps CLS/register-token activations on
        # the same scale as projected pixel features (which were normalized by
        # the pixel decoder's GroupNorm path).
        self.global_token_norm = nn.LayerNorm(dim) if use_global_tokens else None

    def _scale_pos_enc(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Sinusoidal 2D positional encoding, one per spatial position."""
        B, C, H, W = feat_map.shape
        pe = torch.zeros(C, H, W, device=feat_map.device, dtype=feat_map.dtype)
        d_half = C // 4
        y = torch.arange(H, device=feat_map.device, dtype=feat_map.dtype).unsqueeze(1)
        x = torch.arange(W, device=feat_map.device, dtype=feat_map.dtype).unsqueeze(0)
        div = torch.exp(torch.arange(0, d_half, dtype=feat_map.dtype, device=feat_map.device)
                        * -(math.log(10000.0) / max(d_half, 1)))
        pe[0:d_half, :, :] = torch.sin(y * div.view(-1, 1, 1))
        pe[d_half:2 * d_half, :, :] = torch.cos(y * div.view(-1, 1, 1))
        pe[2 * d_half:3 * d_half, :, :] = torch.sin(x * div.view(-1, 1, 1))
        pe[3 * d_half:4 * d_half, :, :] = torch.cos(x * div.view(-1, 1, 1))
        return pe.unsqueeze(0).expand(B, -1, -1, -1)

    def _predict_masks(self, q: torch.Tensor,
                        pixel_features: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-query masks from current query embeddings.
        Returns:
            per_query_masks: (B, Q, H, W) raw mask logits per query
            class_scores:    (B, Q)
            mask_logits:     (B, 1, H, W) class-weighted aggregation (final mask)
        """
        per_query_masks = torch.einsum("bqc,bchw->bqhw", q, pixel_features)
        class_scores = self.class_head(q).squeeze(-1)
        weights = F.softmax(class_scores, dim=1)
        mask_logits = (per_query_masks * weights[:, :, None, None]).sum(dim=1,
                                                                          keepdim=True)
        return per_query_masks, class_scores, mask_logits

    def _build_attn_mask(self, per_query_masks: torch.Tensor,
                          target_hw: tuple[int, int],
                          n_global_tokens: int) -> Optional[torch.Tensor]:
        """Convert per-query masks into a Mask2Former-style cross-attention mask.

        per_query_masks: (B, Q, H_pix, W_pix) raw logits at the high-res pixel
                          decoder resolution.
        target_hw:       (H, W) of THIS layer's memory feature map. The mask is
                          downsampled here before flattening.
        n_global_tokens: number of always-attended tokens (CLS + registers)
                          appended at the END of memory.

        Returns:
            attn_mask: (B*n_heads, Q, S) bool — True positions are BLOCKED.
                        None if masking is disabled.
        """
        if not self.use_masked_attention:
            return None
        B, Q, _, _ = per_query_masks.shape
        # Downsample mask logits to the memory feature-map resolution
        m = F.interpolate(per_query_masks.detach().float(), size=target_hw,
                           mode="bilinear", align_corners=False)
        # Threshold in logit space: sigmoid(thr_logit) == self.mask_attention_threshold
        t = self.mask_attention_threshold
        thr = math.log(t / max(1 - t, 1e-6))
        in_region = (m >= thr)                                     # (B, Q, H, W)
        in_region = in_region.flatten(2)                            # (B, Q, S_spatial)

        # Always allow global tokens (CLS + registers)
        if n_global_tokens > 0:
            global_unblocked = torch.ones(B, Q, n_global_tokens, dtype=torch.bool,
                                            device=in_region.device)
            in_region = torch.cat([in_region, global_unblocked], dim=2)

        # SAFETY: if a row has zero allowed positions, allow ALL positions for
        # that row — otherwise softmax(-inf, -inf, ...) produces NaN.
        all_blocked = ~in_region.any(dim=2, keepdim=True)
        in_region = in_region | all_blocked

        # PyTorch MultiheadAttention: True = MASKED OUT. We've been tracking
        # "in_region = True" so flip.
        attn_mask = ~in_region                                      # (B, Q, S)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_mask = attn_mask.reshape(B * self.n_heads, Q, -1)
        return attn_mask

    def forward(self,
                multi_scale_feats: list[torch.Tensor],
                pixel_features: torch.Tensor,
                global_tokens: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        multi_scale_feats: list of (B, C, H_i, W_i) memory maps cycled across layers.
        pixel_features:    (B, C, H_pix, W_pix) high-res per-pixel embeddings (res2).
        global_tokens:     Optional (B, G, C) CLS + register-token embeddings —
                            appended to every layer's memory as always-attended
                            global context (Mask2Former + DINOv3 enhancement).

        Returns:
            mask_logits:      (B, 1, H_pix, W_pix) — FINAL mask (last layer)
            class_scores:     (B, num_queries) — final per-query confidences
            aux_mask_logits:  list of (B, 1, H_pix, W_pix) per-layer masks for
                                deep supervision. The LAST entry == mask_logits.
                                The FIRST entry is the layer-0 prediction
                                (before any decoder layer ran), giving num_layers+1
                                entries total.
        """
        B = pixel_features.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # (B, Q, C)
        q_pos = self.query_pos.unsqueeze(0).expand(B, -1, -1)

        # Pre-compute the global-token memory contribution (constant across layers)
        if (self.use_global_tokens and global_tokens is not None
                and global_tokens.shape[1] > 0):
            gt = self.global_token_norm(global_tokens)
            n_global = gt.shape[1]
        else:
            gt = None
            n_global = 0

        # Step-0 mask prediction (BEFORE first layer): drives layer-0 attention
        # mask. This is the iterative-refinement scheme — layer N's masked
        # attention uses the mask predicted at step N.
        per_query_masks, class_scores, mask_logits = self._predict_masks(
            q, pixel_features
        )
        aux_mask_logits = [mask_logits]

        for li, layer in enumerate(self.layers):
            scale_idx = li % len(multi_scale_feats)
            f = multi_scale_feats[scale_idx]
            pe = self._scale_pos_enc(f)
            mem_spatial = (f + pe).flatten(2).transpose(1, 2)      # (B, S_sp, C)
            if gt is not None:
                mem = torch.cat([mem_spatial, gt], dim=1)         # (B, S_sp+G, C)
            else:
                mem = mem_spatial

            attn_mask = self._build_attn_mask(
                per_query_masks, target_hw=f.shape[-2:],
                n_global_tokens=n_global,
            )

            q = layer(q + q_pos, mem, attn_mask=attn_mask)

            # Predict mask AFTER each layer for deep supervision + next-layer mask
            per_query_masks, class_scores, mask_logits = self._predict_masks(
                q, pixel_features
            )
            aux_mask_logits.append(mask_logits)

        return mask_logits, class_scores, aux_mask_logits


# ══════════════════════════════════════════════════════════════════════
# Optional learnable refinement head — small UNet over [image, coarse_mask]
# ══════════════════════════════════════════════════════════════════════

class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UNet3PlusFusionBlock(nn.Module):
    """One decoder node in UNet3+ — the heart of the full-scale skip design.

    Each decoder node receives features from EVERY encoder scale AND every
    DEEPER decoder node. Each input is:
      1. Projected to a uniform channel count (`cat_channels`) via a 1x1 conv
      2. Resized (max-pool down OR bilinear up) to this decoder node's scale
      3. Concatenated with the others
      4. Fused via a 3x3 conv block to `out_channels`

    This keeps decoder nodes uniform-width and produces sharper boundaries on
    thin/elongated structures (fence pickets) than plain UNet's level-by-level
    skip connections.
    """
    def __init__(self, in_channels_list: list[int],
                 cat_channels: int, out_channels: int) -> None:
        super().__init__()
        # Per-input 1x1 projection + GroupNorm + GELU. Brings every input
        # (encoder OR decoder, any depth) onto a common feature scale.
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, cat_channels, 1),
                nn.GroupNorm(8, cat_channels),
                nn.GELU(),
            ) for in_c in in_channels_list
        ])
        # Concat all inputs → fuse to out_channels via two 3x3 conv blocks.
        total_in = len(in_channels_list) * cat_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: list[torch.Tensor],
                target_hw: tuple[int, int]) -> torch.Tensor:
        """Resize each projected input to target_hw, concat, fuse.
        inputs: list of (B, in_c_i, H_i, W_i) — one per scale.
        target_hw: (H, W) of THIS decoder node's spatial grid."""
        projected = []
        for x, proj in zip(inputs, self.projections):
            y = proj(x)
            if y.shape[-2:] != target_hw:
                # Downsample via max-pool (stride>=2) when going to coarser scale,
                # bilinear up when going to finer scale. Use bilinear for both —
                # it's the right invariant for soft refinement features.
                y = F.interpolate(y, size=target_hw, mode="bilinear",
                                    align_corners=False)
            projected.append(y)
        return self.fuse(torch.cat(projected, dim=1))


class RefinementHead(nn.Module):
    """UNet3+ refinement head with full-scale skip connections.

    Replaces the previous plain UNet refinement (which had simple level-by-level
    skip connections). UNet3+ (Huang et al., 2020) was specifically designed for
    thin / scattered structures — exactly the failure mode of fence pickets at
    distance and through occlusion gaps.

    Architecture (with `base_channels=64`, `num_blocks=4`):
      - Encoder: 5 scales, channels grow 64 → 128 → 256 → 512 → 1024
      - Decoder: 4 nodes (D0..D3), all at uniform `base_channels` width.
        Each Dx receives ALL of:
          * E0..Ex   (encoder, downsampled or identity)
          * Dx+1..D{N-1}  (deeper decoder nodes, upsampled)
          * E{N}     (encoder bottleneck, upsampled)
        → N+1 inputs per decoder node — full-scale fusion at every level.
      - Output: residual mask logits + optional edge logits, both off D0.

    Same public interface as the previous RefinementHead so it's a true
    drop-in: `__init__(base_channels, num_blocks, extra_in_channels, predict_edge)`,
    forward returns `(refined_logits, edge_logits-or-None)`.

    Why this is a strict improvement over plain UNet here:
      - Decoder is LIGHTER on params (uniform 64-channel decoder beats
        plain UNet's deep 1024+512→512 fusion blocks).
      - Full-scale fusion adds MORE multi-scale information per decoder node,
        which is what actually drives boundary quality on thin pickets.
      - Activations are slightly larger due to the cross-scale projections
        (~+30% during refinement pass), but the refinement is only ~3% of
        total model FLOPs so the wall-clock impact is small.
    """

    def __init__(self, base_channels: int = 64, num_blocks: int = 4,
                 extra_in_channels: int = 0, predict_edge: bool = False,
                 use_full_scale_ds: bool = True,
                 use_cgm: bool = True,
                 predict_distance: bool = True,
                 use_pointrend_module: bool = True,
                 pointrend_n_uncertain: int = 4096,
                 pointrend_hidden: int = 128) -> None:
        super().__init__()
        c = base_channels
        N = int(num_blocks)
        in_ch = 3 + 1 + max(0, int(extra_in_channels))   # image + coarse + extras
        self.cat_channels = c
        self.N = N
        self.use_full_scale_ds = bool(use_full_scale_ds)
        self.use_cgm = bool(use_cgm)
        self.predict_distance = bool(predict_distance)
        self.use_pointrend_module = bool(use_pointrend_module)
        self.pointrend_n_uncertain = int(pointrend_n_uncertain)

        # ── Encoder (same shape as plain UNet — kept for compatibility with
        #     existing checkpoint loading + the same DINOv3-fed input contract).
        self.in_conv = _ConvBlock(in_ch, c)
        encoder_channels = [c * (2 ** i) for i in range(N + 1)]
        self.down = nn.ModuleList([
            _ConvBlock(encoder_channels[i], encoder_channels[i + 1])
            for i in range(N)
        ])
        # UNet3+ paper uses MaxPool for the encoder downsample.
        self.pool = nn.MaxPool2d(2)

        # ── Decoder: N nodes (D0..D{N-1}), each with N+1 full-scale inputs.
        #    For Dx, inputs in order: [E0, E1, ..., Ex, D{x+1}, ..., D{N-1}, EN].
        self.decoders = nn.ModuleList()
        for x in range(N):
            in_channels_list: list[int] = []
            # Encoder side: E0..Ex (everything at this scale or coarser)
            for i in range(x + 1):
                in_channels_list.append(encoder_channels[i])
            # Decoder side: D{x+1}..D{N-1} (deeper decoder nodes; uniform width c)
            for _ in range(x + 1, N):
                in_channels_list.append(c)
            # Encoder bottleneck EN (always)
            in_channels_list.append(encoder_channels[N])

            self.decoders.append(_UNet3PlusFusionBlock(
                in_channels_list=in_channels_list,
                cat_channels=c,
                out_channels=c,
            ))

        # ── Output heads off D0 (highest-resolution decoder)
        self.out = nn.Conv2d(c, 1, 1)
        self.edge_head: Optional[nn.Conv2d] = (
            nn.Conv2d(c, 1, 1) if predict_edge else None
        )

        # ── Full-scale Deep Supervision (UNet3+ paper §3.2).
        # Side heads off D1..D{N-1} (D0 is already the main `out` head).
        # Each side head outputs a single-channel logit at its decoder scale;
        # callers upsample to image resolution before BCE/Dice loss.
        self.fds_heads: Optional[nn.ModuleList] = None
        if self.use_full_scale_ds and N >= 2:
            self.fds_heads = nn.ModuleList([
                nn.Conv2d(c, 1, 1) for _ in range(N - 1)   # for D1..D{N-1}
            ])

        # ── Classification-Guided Module (UNet3+ paper §3.3).
        # Binary classifier off the encoder bottleneck E_N: "does this image
        # contain ANY foreground (fence) pixels?". At INFERENCE we gate the
        # final mask logits by log(p_cgm) so negative-image false positives
        # collapse to zero. During training the gate is NOT applied — the
        # CGM is supervised independently via BCE on the per-image fence/non-
        # fence label.
        self.cgm: Optional[nn.Module] = None
        if self.use_cgm:
            self.cgm = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(encoder_channels[N], encoder_channels[N] // 4),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(encoder_channels[N] // 4, 1),
            )

        # ── Boundary Distance Regression head (off D0).
        # Predicts a per-pixel signed distance to the nearest GT boundary
        # (negative inside fence, positive outside, zero at boundary, all
        # clipped to ±50 px during loss). Sub-pixel boundary localization
        # signal that complements the BCE/Dice mask losses.
        self.distance_head: Optional[nn.Conv2d] = (
            nn.Conv2d(c, 1, 1) if self.predict_distance else None
        )

        # ── Forward-time PointRend MLP module (Kirillov et al., 2020).
        # At each forward (train + inference), we identify the top-K most
        # uncertain pixels (|sigmoid(refined_logit) - 0.5| smallest), then
        # pass [feature_at_pixel, coarse_logit_at_pixel] through a small MLP
        # to predict a refined logit at THAT pixel. Sub-pixel boundary
        # refinement on top of UNet3+'s output.
        self.pointrend_mlp: Optional[nn.Module] = None
        if self.use_pointrend_module:
            self.pointrend_mlp = nn.Sequential(
                nn.Linear(c + 1, pointrend_hidden),
                nn.GELU(),
                nn.Linear(pointrend_hidden, pointrend_hidden),
                nn.GELU(),
                nn.Linear(pointrend_hidden, 1),
            )

    def forward(self, image: torch.Tensor, coarse_mask: torch.Tensor,
                extras: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor,
                           Optional[torch.Tensor],
                           Optional[list[torch.Tensor]],
                           Optional[torch.Tensor],
                           Optional[torch.Tensor]]:
        """image: (B, 3, H, W). coarse_mask: (B, 1, H, W) sigmoid-space.
        extras: Optional (B, K, H, W) extra channels.
        Returns 5-tuple:
          refined_logits         — (B, 1, H, W) main refined mask (with PointRend
                                    sub-pixel refinement at uncertain pixels +
                                    CGM gate at inference)
          edge_logits            — (B, 1, H, W) edge-head output | None
          fds_aux_logits         — list[(B, 1, H, W)] FDS side heads | None
          cgm_logit              — (B, 1) per-image classifier logit | None
          boundary_distance_logits — (B, 1, H, W) signed-distance prediction
                                      to nearest fence boundary | None
        """
        if extras is not None:
            x = torch.cat([image, coarse_mask, extras], dim=1)
        else:
            x = torch.cat([image, coarse_mask], dim=1)

        # Encoder forward — collect E0..EN
        e: list[torch.Tensor] = [self.in_conv(x)]
        for d in self.down:
            e.append(d(self.pool(e[-1])))

        # Decoder forward — build deepest-first so deeper Dx are available
        # when shallower ones consume them. Each Dx outputs at the spatial
        # scale of Ex.
        decoders_out: list[Optional[torch.Tensor]] = [None] * self.N
        for x in reversed(range(self.N)):
            target_hw = (e[x].shape[-2], e[x].shape[-1])
            inputs: list[torch.Tensor] = []
            # E0..Ex (encoder, will be down/identity-resampled in fusion block)
            for i in range(x + 1):
                inputs.append(e[i])
            # D{x+1}..D{N-1} (decoder, will be upsampled)
            for x_other in range(x + 1, self.N):
                # Already computed because we iterate in reverse.
                d_out = decoders_out[x_other]
                assert d_out is not None
                inputs.append(d_out)
            # EN (encoder bottleneck, will be upsampled)
            inputs.append(e[self.N])
            decoders_out[x] = self.decoders[x](inputs, target_hw)

        # D0 is at full image resolution → produce mask + edge
        out0 = decoders_out[0]
        assert out0 is not None
        residual = self.out(out0)
        edge_logits = self.edge_head(out0) if self.edge_head is not None else None

        # FDS — side heads off D1..D{N-1}, upsample to image resolution
        fds_aux_logits: Optional[list[torch.Tensor]] = None
        if self.fds_heads is not None:
            fds_aux_logits = []
            H_img, W_img = image.shape[-2:]
            for i, head in enumerate(self.fds_heads):
                d_idx = i + 1                     # heads correspond to D1..D{N-1}
                d_feat = decoders_out[d_idx]
                assert d_feat is not None
                aux = head(d_feat)
                if aux.shape[-2:] != (H_img, W_img):
                    aux = F.interpolate(aux, size=(H_img, W_img),
                                          mode="bilinear", align_corners=False)
                fds_aux_logits.append(aux)

        # CGM — pooled classifier off the encoder bottleneck E_N
        cgm_logit: Optional[torch.Tensor] = None
        if self.cgm is not None:
            cgm_logit = self.cgm(e[self.N])              # (B, 1)

        # Residual added to coarse logits in pre-sigmoid logit space.
        coarse_logit = torch.logit(coarse_mask.clamp(1e-6, 1 - 1e-6))
        refined_logits = coarse_logit + residual

        # Boundary Distance Regression head — off the same D0 features.
        # Trained against the GT signed-distance map (computed in the loss).
        boundary_distance_logits: Optional[torch.Tensor] = None
        if self.distance_head is not None:
            boundary_distance_logits = self.distance_head(out0)

        # Forward-time PointRend refinement on the most-uncertain pixels.
        # Find top-K uncertain pixels (sigmoid output closest to 0.5), gather
        # their D0 features + current refined logits, pass through MLP for a
        # sub-pixel-accurate refined logit, scatter back. Operates in train
        # AND inference modes — during training the MLP gets gradient via the
        # downstream BCE/Dice loss on `refined_logits`.
        if self.pointrend_mlp is not None:
            refined_logits = self._apply_pointrend(refined_logits, out0)

        # CGM gating at INFERENCE: bias the logits by log(p_cgm) per image.
        # This way sigmoid(refined_logits + log(p_cgm)) ≈ p_seg * p_cgm,
        # i.e. negative images get their predictions multiplicatively suppressed.
        # During training we do NOT gate — let the seg path learn freely and
        # CGM be supervised independently via BCE.
        if (cgm_logit is not None) and (not self.training):
            p_cgm = torch.sigmoid(cgm_logit).clamp(1e-6, 1.0)   # (B, 1)
            log_gate = torch.log(p_cgm).view(-1, 1, 1, 1)        # broadcast
            refined_logits = refined_logits + log_gate

        return (refined_logits, edge_logits, fds_aux_logits, cgm_logit,
                boundary_distance_logits)

    def _apply_pointrend(self, refined_logits: torch.Tensor,
                          d0_features: torch.Tensor) -> torch.Tensor:
        """Sub-pixel PointRend refinement on the top-K most-uncertain pixels.

        Identifies pixels where |sigmoid(refined_logit) - 0.5| is smallest
        (= max ambiguity), gathers their D0 features + current logit, runs
        them through self.pointrend_mlp, and scatters the output back into
        refined_logits at those positions.

        K is min(self.pointrend_n_uncertain, H*W). At low resolution every
        pixel may be touched; at high res only the boundary band is touched.
        """
        assert self.pointrend_mlp is not None
        B, _, H, W = refined_logits.shape
        K = min(self.pointrend_n_uncertain, H * W)

        # Uncertainty score (no grad) — flatten and pick top-K per image
        with torch.no_grad():
            uncertainty = -(torch.sigmoid(refined_logits.detach()) - 0.5).abs()
            u_flat = uncertainty.view(B, -1)
            _, idx_flat = torch.topk(u_flat, k=K, dim=1)            # (B, K)

        # Gather D0 features at those points: (B, c, H*W) → gather → (B, c, K)
        c = d0_features.shape[1]
        feat_flat = d0_features.view(B, c, -1)
        gather_idx = idx_flat.unsqueeze(1).expand(-1, c, -1)         # (B, c, K)
        feat_at_pts = torch.gather(feat_flat, 2, gather_idx)         # (B, c, K)

        # Current logit at those points: (B, 1, H*W) → gather → (B, 1, K)
        logit_flat = refined_logits.view(B, 1, -1)
        logit_at_pts = torch.gather(logit_flat, 2, idx_flat.unsqueeze(1))  # (B,1,K)

        # MLP input: concat features + current logit → (B, K, c+1)
        mlp_in = torch.cat([feat_at_pts, logit_at_pts], dim=1).transpose(1, 2)
        new_logits = self.pointrend_mlp(mlp_in)                      # (B, K, 1)

        # Scatter the refined logits back into the spatial map.
        # Under bf16 autocast, the MLP output can be fp32 while the destination
        # is bf16 — scatter requires matching dtypes, so cast the source.
        # Gradient at scattered positions flows through the MLP only; gradient
        # at un-scattered positions flows through the original refined_logits.
        out_flat = refined_logits.view(B, 1, -1).clone()
        new_logits_t = new_logits.transpose(1, 2).to(dtype=out_flat.dtype)
        out_flat.scatter_(2, idx_flat.unsqueeze(1), new_logits_t)
        return out_flat.view_as(refined_logits)


# ══════════════════════════════════════════════════════════════════════
# Top-level model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelOutputs:
    mask_logits: torch.Tensor                 # (B, 1, H, W)  — final mask
    refined_logits: Optional[torch.Tensor] = None    # (B, 1, H, W)  — if refinement head on
    class_scores: Optional[torch.Tensor] = None      # (B, Q) — for inspection
    # Deep-supervision: list of (B, 1, H, W) per-layer mask logits from the
    # M2F decoder (incl. step-0). Last entry is the same as mask_logits.
    aux_logits: Optional[list[torch.Tensor]] = None
    # Edge head from refinement (Sobel-supervised)
    edge_logits: Optional[torch.Tensor] = None
    # Per-iteration refined logits from iterative refinement (deep supervision
    # on the refinement loop). Last entry == refined_logits.
    refined_iter_logits: Optional[list[torch.Tensor]] = None
    # UNet3+ Full-scale Deep Supervision side heads off D1..D{N-1} of the
    # refinement head. Already upsampled to image resolution. Trained against
    # the same GT mask via BCE+Dice with a small per-head weight.
    refined_fds_logits: Optional[list[torch.Tensor]] = None
    # UNet3+ Classification-Guided Module: per-image binary fence/non-fence
    # logit. (B, 1). Used for BCE supervision against `class == pos` and to
    # gate the refined output at inference (negative images → empty mask).
    cgm_logit: Optional[torch.Tensor] = None
    # Boundary Distance Regression head: per-pixel signed distance to the
    # nearest fence boundary (negative inside fence, positive outside, 0 on
    # boundary). Trained with L1 loss against the GT signed-distance map
    # computed in CombinedLoss via scipy distance transform.
    boundary_distance_logits: Optional[torch.Tensor] = None


class FenceSegmentationModel(nn.Module):
    """End-to-end model: backbone -> FPN adapter -> pixel decoder -> M2F decoder
    -> mask head (-> optional refinement head)."""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.cfg = model_cfg
        # Backbone — DinoBackbone OR DinoBackboneWithAdapter depending on config.
        # When use_vit_adapter=True, the backbone produces a ready multi-scale
        # pyramid {res2, res3, res4, res5} via ViT-Adapter and we BYPASS ViTToFPN.
        self.use_vit_adapter = bool(getattr(model_cfg, "use_vit_adapter", False))
        if self.use_vit_adapter:
            self.backbone = DinoBackboneWithAdapter(
                name=model_cfg.backbone_name,
                freeze_first_n_layers=model_cfg.backbone_freeze_first_n_layers,
                drop_path_rate=model_cfg.backbone_drop_path_rate,
                multi_block_n=getattr(model_cfg, "multi_block_n", 1),
                aggregation_type=getattr(model_cfg, "aggregation_type", "weighted_sum"),
                return_global_tokens=getattr(model_cfg, "use_global_tokens", True),
                adapter_inplanes=int(getattr(model_cfg, "vit_adapter_inplanes", 64)),
                adapter_n_heads=int(getattr(model_cfg, "vit_adapter_n_heads", 8)),
                adapter_n_points=int(getattr(model_cfg, "vit_adapter_n_points", 4)),
                adapter_n_interactions=int(getattr(model_cfg, "vit_adapter_n_interactions", 4)),
                adapter_init_values=float(getattr(model_cfg, "vit_adapter_init_values", 0.0)),
                adapter_ffn_ratio=float(getattr(model_cfg, "vit_adapter_ffn_ratio", 0.25)),
            )
            # Project the adapter's 4 scales (each at backbone dim) down to
            # decoder_dim with cheap 1x1 convs — replaces ViTToFPN entirely.
            self.adapter = None
            self.adapter_proj = nn.ModuleDict({
                k: nn.Conv2d(self.backbone.dim, model_cfg.decoder_dim, 1)
                for k in ("res2", "res3", "res4", "res5")
            })
        else:
            self.backbone = DinoBackbone(
                name=model_cfg.backbone_name,
                freeze_first_n_layers=model_cfg.backbone_freeze_first_n_layers,
                drop_path_rate=model_cfg.backbone_drop_path_rate,
                multi_block_n=getattr(model_cfg, "multi_block_n", 1),
                aggregation_type=getattr(model_cfg, "aggregation_type", "weighted_sum"),
                return_global_tokens=getattr(model_cfg, "use_global_tokens", True),
            )
            self.adapter = ViTToFPN(self.backbone.dim, out_dim=model_cfg.decoder_dim)
            self.adapter_proj = None

        # Pixel decoder: simple FPN OR real Mask2Former MSDeformAttn encoder
        pd_type = getattr(model_cfg, "pixel_decoder_type", "fpn")
        if pd_type == "msdeform":
            self.pixel_decoder = MSDeformPixelDecoder(
                dim=model_cfg.decoder_dim,
                num_layers=getattr(model_cfg, "pixel_decoder_layers", 6),
                num_heads=getattr(model_cfg, "pixel_decoder_heads", 8),
                feedforward_dim=getattr(model_cfg, "pixel_decoder_ffn_dim", 1024),
                dropout=model_cfg.decoder_dropout,
            )
        elif pd_type == "fpn":
            self.pixel_decoder = PixelDecoder(dim=model_cfg.decoder_dim)
        else:
            raise ValueError(f"Unknown pixel_decoder_type: {pd_type}")
        self._pixel_decoder_type = pd_type

        # Project backbone-dim global tokens (CLS + registers) into decoder_dim
        # so they're compatible with the cross-attention memory space.
        use_global = getattr(model_cfg, "use_global_tokens", True)
        if use_global:
            self.global_token_proj = nn.Linear(self.backbone.dim,
                                                 model_cfg.decoder_dim)
        else:
            self.global_token_proj = None

        self.transformer_decoder = Mask2FormerLikeDecoder(
            dim=model_cfg.decoder_dim,
            num_queries=model_cfg.decoder_num_queries,
            num_layers=model_cfg.decoder_num_layers,
            dropout=model_cfg.decoder_dropout,
            use_masked_attention=getattr(model_cfg, "use_masked_attention", True),
            use_global_tokens=use_global,
            mask_attention_threshold=getattr(model_cfg,
                                                "mask_attention_threshold", 0.5),
        )
        # Refinement head (with optional decoder-feature injection + edge head)
        self.refinement_use_decoder_features = bool(getattr(
            model_cfg, "refinement_use_decoder_features", True))
        self.refinement_decoder_feature_channels = int(getattr(
            model_cfg, "refinement_decoder_feature_channels", 64))
        self.refinement_use_edge_head = bool(getattr(
            model_cfg, "refinement_use_edge_head", True))
        self.refinement_iterations = max(1, int(getattr(
            model_cfg, "refinement_iterations", 1)))

        # Depth / position cues for the refinement head:
        #   - y_coord: free, normalized vertical position channel
        #   - depth:   frozen DPT/MiDaS teacher, 1 extra channel
        self.refinement_use_y_coord = bool(getattr(
            model_cfg, "refinement_use_y_coord", True))
        self.refinement_use_depth = bool(getattr(
            model_cfg, "refinement_use_depth", False))
        self.depth_model_name = str(getattr(
            model_cfg, "depth_model_name", "Intel/dpt-hybrid-midas"))

        # Lazy-load DPT (frozen) only if depth path is enabled. Keeps memory
        # off the GPU and avoids the import/HF-cache hit when not used.
        self.depth_model: Optional[nn.Module] = None
        if self.refinement_use_depth:
            self.depth_model = self._build_frozen_depth_model(self.depth_model_name)
            # ImageNet (DINOv3) and DPT (typically (0.5, 0.5, 0.5)) use
            # different normalizations. We store both as buffers so the
            # un-then-re-normalization happens on whatever device the model is on.
            self.register_buffer("_imagenet_mean",
                                 torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_imagenet_std",
                                 torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.register_buffer("_dpt_mean",
                                 torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
            self.register_buffer("_dpt_std",
                                 torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        if model_cfg.use_refinement_head:
            extra_ch = (self.refinement_decoder_feature_channels
                        if self.refinement_use_decoder_features else 0)
            extra_ch += (1 if self.refinement_use_y_coord else 0)
            extra_ch += (1 if self.refinement_use_depth else 0)
            self.refinement = RefinementHead(
                model_cfg.refinement_channels,
                model_cfg.refinement_num_blocks,
                extra_in_channels=extra_ch,
                predict_edge=self.refinement_use_edge_head,
                use_full_scale_ds=bool(getattr(
                    model_cfg, "refinement_use_full_scale_ds", True)),
                use_cgm=bool(getattr(
                    model_cfg, "refinement_use_cgm", True)),
                predict_distance=bool(getattr(
                    model_cfg, "refinement_use_distance", True)),
                use_pointrend_module=bool(getattr(
                    model_cfg, "refinement_use_pointrend_module", True)),
                pointrend_n_uncertain=int(getattr(
                    model_cfg, "refinement_pointrend_n_uncertain", 4096)),
                pointrend_hidden=int(getattr(
                    model_cfg, "refinement_pointrend_hidden", 128)),
            )
            # 1x1 projection: pixel decoder features (decoder_dim) -> compact K
            # so we don't blow up the refinement input by 512+ channels.
            if self.refinement_use_decoder_features:
                self.refinement_feature_proj = nn.Conv2d(
                    model_cfg.decoder_dim,
                    self.refinement_decoder_feature_channels, 1,
                )
            else:
                self.refinement_feature_proj = None
        else:
            self.refinement = None
            self.refinement_feature_proj = None

    @staticmethod
    def _build_frozen_depth_model(name: str) -> nn.Module:
        """Load a HuggingFace depth-estimation model (e.g. DPT) and freeze it.
        All parameters get requires_grad=False; the model goes into eval()
        mode so dropout/BN behave deterministically; and `forward` is wrapped
        to enforce the no-grad contract at training time."""
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as e:
            raise ImportError(
                "DPT depth requires `transformers` installed."
            ) from e
        try:
            mdl = AutoModelForDepthEstimation.from_pretrained(name)
        except Exception as e:
            raise RuntimeError(
                f"[depth] failed to load '{name}': {type(e).__name__}: "
                f"{str(e)[:160]}.\nIf this is a gated HF model, run "
                f"`huggingface-cli login` first."
            ) from e
        for p in mdl.parameters():
            p.requires_grad = False
        mdl.eval()
        return mdl

    def _build_y_coord(self, B: int, H: int, W: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a (B, 1, H, W) tensor with normalized y-position [0, 1].
        Top of image = 0, bottom = 1. Constant per image, content-independent.
        Captures the strong correlation between vertical position and depth
        in outdoor fence scenes (sky on top, ground on bottom)."""
        y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        y_map = y.view(1, 1, H, 1).expand(B, 1, H, W)
        return y_map

    @torch.no_grad()
    def _compute_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Predict (B, 1, H, W) normalized [0, 1] depth from the input image.

        Image enters with ImageNet (DINOv3) normalization. We un-normalize back
        to [0, 1], then re-normalize for DPT, then forward the frozen DPT, then
        upsample to image resolution and per-image min-max normalize.

        Frozen, no_grad — DPT contributes no gradient, no optimizer state.
        """
        assert self.depth_model is not None
        # 1. Un-normalize from ImageNet space back to [0, 1]
        img01 = image * self._imagenet_std + self._imagenet_mean
        # 2. Re-normalize for DPT (mean=std=0.5)
        img_dpt = (img01 - self._dpt_mean) / self._dpt_std
        # 3. Resize to DPT-friendly resolution (it natively likes 384;
        #    we accept anything multiple of 32). For speed, downscale.
        target = 384
        H, W = img_dpt.shape[-2:]
        if (H, W) != (target, target):
            img_dpt_ds = F.interpolate(img_dpt, size=(target, target),
                                         mode="bilinear", align_corners=False)
        else:
            img_dpt_ds = img_dpt
        # 4. Forward through DPT — outputs `predicted_depth: (B, H_d, W_d)`
        out = self.depth_model(pixel_values=img_dpt_ds)
        depth_lo = out.predicted_depth                                 # (B, H_d, W_d)
        if depth_lo.dim() == 3:
            depth_lo = depth_lo.unsqueeze(1)                            # (B, 1, ...)
        # 5. Upsample to image resolution
        depth = F.interpolate(depth_lo, size=(H, W), mode="bilinear",
                               align_corners=False)
        # 6. Per-image min-max normalize to [0, 1]
        flat = depth.flatten(2)
        d_min = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
        d_max = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth.detach().to(dtype=image.dtype)

    def forward(self, image: torch.Tensor) -> ModelOutputs:
        B, _, H, W = image.shape
        # 1. Backbone forward
        bb_out = self.backbone(image)
        if isinstance(bb_out, dict):
            vit_feats = bb_out["spatial"]
            cls_tok = bb_out.get("cls")
            reg_tok = bb_out.get("registers")
        else:
            vit_feats = bb_out
            cls_tok = None
            reg_tok = None

        # 2. Build the FPN (4-scale dict).
        #    - With ViT-Adapter: backbone already returned a multi-scale pyramid.
        #      Just project each scale (backbone_dim → decoder_dim).
        #    - Without: run our homemade ViTToFPN on the single-scale ViT output.
        if self.use_vit_adapter:
            ms = bb_out["multi_scale"]      # {res2, res3, res4, res5} at backbone dim
            fpn = {k: self.adapter_proj[k](v) for k, v in ms.items()}
        else:
            fpn = self.adapter(vit_feats)
        # 3. Pixel decoder
        #    - "fpn":      returns a single (B, C, H/4, W/4) high-res tensor
        #    - "msdeform": returns dict with enhanced res2/res3/res4/res5
        pd_out = self.pixel_decoder(fpn)
        if self._pixel_decoder_type == "msdeform":
            # Use enhanced features for both transformer-decoder memory AND
            # high-res pixel features.
            pix = pd_out["res2"]
            ms = [pd_out["res5"], pd_out["res4"], pd_out["res3"]]
        else:
            pix = pd_out                                           # (B, C, H/4, W/4)
            # Cross-attention memory comes from raw FPN levels
            ms = [fpn["res5"], fpn["res4"], fpn["res3"]]

        # 4. Build global token memory for the decoder cross-attention.
        global_tokens = None
        if self.global_token_proj is not None:
            parts = []
            if cls_tok is not None:
                parts.append(cls_tok.unsqueeze(1))                # (B, 1, C_bb)
            if reg_tok is not None and reg_tok.shape[1] > 0:
                parts.append(reg_tok)                              # (B, R, C_bb)
            if parts:
                gt_bb = torch.cat(parts, dim=1)                   # (B, G, C_bb)
                global_tokens = self.global_token_proj(gt_bb)     # (B, G, C_dec)

        # 5. Transformer decoder — returns final mask + per-layer aux masks
        coarse_logits, class_scores, aux_layer_logits = self.transformer_decoder(
            ms, pix, global_tokens=global_tokens,
        )

        # Upsample coarse + aux logits to input resolution
        coarse_logits_up = F.interpolate(coarse_logits, size=(H, W),
                                            mode="bilinear", align_corners=False)
        aux_up: list[torch.Tensor] = []
        for al in aux_layer_logits:
            aux_up.append(F.interpolate(al, size=(H, W), mode="bilinear",
                                          align_corners=False))

        outputs = ModelOutputs(
            mask_logits=coarse_logits_up,
            class_scores=class_scores,
            aux_logits=aux_up,
        )

        # 6. Optional refinement head — with decoder feature injection (A),
        #    edge head (B), iterative refinement (C), y-coord cue, depth cue.
        if self.refinement is not None:
            with torch.no_grad():
                # Detach coarse so refinement gradients don't flow back into
                # the main decoder. Same property for the decoder features.
                coarse_prob = torch.sigmoid(coarse_logits_up).detach()
                extra_parts: list[torch.Tensor] = []
                if self.refinement_feature_proj is not None:
                    # Project decoder dim -> compact K, detach, upsample to image res.
                    extras_lo = self.refinement_feature_proj(pix.detach())
                    extra_parts.append(F.interpolate(
                        extras_lo, size=(H, W),
                        mode="bilinear", align_corners=False,
                    ))
                if self.refinement_use_y_coord:
                    extra_parts.append(self._build_y_coord(
                        B, H, W, image.device, image.dtype,
                    ))
                if self.refinement_use_depth and self.depth_model is not None:
                    extra_parts.append(self._compute_depth(image))
                extras = (torch.cat(extra_parts, dim=1)
                          if extra_parts else None)

            current = coarse_prob
            iter_logits: list[torch.Tensor] = []
            edge_last: Optional[torch.Tensor] = None
            fds_last: Optional[list[torch.Tensor]] = None
            cgm_last: Optional[torch.Tensor] = None
            dist_last: Optional[torch.Tensor] = None
            for it in range(self.refinement_iterations):
                (refined_logits, edge_logits, fds_aux,
                 cgm_logit, dist_logits) = self.refinement(
                    image, current, extras=extras,
                )
                iter_logits.append(refined_logits)
                edge_last = edge_logits
                fds_last = fds_aux       # only the LAST iter's FDS heads supervise
                cgm_last = cgm_logit     # CGM is per-image, last iter is fine
                dist_last = dist_logits  # last iter's distance prediction
                # Detach between iterations so gradient depth stays bounded
                # (each iteration's gradients only flow through ITS own UNet
                # pass, not back through earlier ones — keeps memory + stable).
                if it < self.refinement_iterations - 1:
                    current = torch.sigmoid(refined_logits).detach()

            outputs.refined_logits = refined_logits
            outputs.edge_logits = edge_last
            outputs.refined_iter_logits = iter_logits if len(iter_logits) > 1 else None
            outputs.refined_fds_logits = fds_last
            outputs.cgm_logit = cgm_last
            outputs.boundary_distance_logits = dist_last

        return outputs

    def inference(self, image: torch.Tensor, use_refined: bool = True) -> torch.Tensor:
        """Inference helper: returns the final mask probabilities (B, 1, H, W)."""
        out = self.forward(image)
        logits = out.refined_logits if (use_refined and out.refined_logits is not None) else out.mask_logits
        return torch.sigmoid(logits)

    @property
    def patch_size(self) -> int:
        """The backbone's ViT patch size (14 for DINOv2, 16 for DINOv3).
        Used by collators / TTA to snap multi-scale dims to a valid stride."""
        return int(getattr(self.backbone, "patch_size", 14))

    def enable_gradient_checkpointing(self) -> bool:
        return self.backbone.enable_gradient_checkpointing()


def build_model(model_cfg) -> FenceSegmentationModel:
    """Factory used by train.py."""
    return FenceSegmentationModel(model_cfg)
