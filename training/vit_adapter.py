"""training/vit_adapter.py — ViT-Adapter modules (Chen et al., ICLR 2023).

Adds dense-prediction-friendly multi-scale spatial features to a plain ViT
backbone via three components:

  1. SpatialPriorModule (SPM) — a small CNN that runs in PARALLEL to the ViT
     and produces multi-scale features (c1@stride4, c2@stride8, c3@stride16,
     c4@stride32). These are concatenated as a "spatial pyramid" token
     sequence for the bidirectional interactions.

  2. Injector — at every interaction point, the multi-scale spatial pyramid
     is INJECTED into the ViT patch tokens via deformable cross-attention.
     This gives the ViT access to fine-grained spatial detail it normally
     loses at patchification (16x16 patches → coarse 14x14 grid for 224^2).

  3. Extractor — at the same interaction points, the ViT patch tokens are
     EXTRACTED back into the spatial pyramid via single-scale deformable
     cross-attention. The pyramid c2/c3/c4 are refined by ViT's global
     reasoning at every interaction.

Net effect: the backbone preserves DINOv3-H+'s pretrained ViT weights but
also produces a high-quality 4-scale feature pyramid {res2, res3, res4, res5}
suitable for any dense-prediction head — replacing the simple ViTToFPN
adapter we had before.

Compared to plain ViT + simple adapter: +1-3% mIoU on dense prediction tasks
(per the paper), at ~+30-60M params and ~+15-20% forward time.

Reference: github.com/czczup/ViT-Adapter
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Multi-Scale Deformable Cross-Attention
# ══════════════════════════════════════════════════════════════════════

def _ms_deform_attn(value: torch.Tensor,
                     spatial_shapes: list[tuple[int, int]],
                     sampling_locations: torch.Tensor,
                     attention_weights: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch MSDeformAttn (Zhu et al., 2020).

    Args:
      value:               (B, N_total, n_heads, head_dim)
                            flattened-and-concatenated value tokens across all levels
      spatial_shapes:      list of (H_l, W_l) for each level
      sampling_locations:  (B, N_query, n_heads, n_levels, n_points, 2)
                            normalized [0,1] xy coords (x first)
      attention_weights:   (B, N_query, n_heads, n_levels, n_points)
                            softmax-normalized over (n_levels, n_points)

    Returns: (B, N_query, n_heads * head_dim)
    """
    B, _, n_heads, head_dim = value.shape
    _, N_query, _, n_levels, n_points, _ = sampling_locations.shape

    # Split value back to per-level
    value_lvls = []
    cursor = 0
    for H_l, W_l in spatial_shapes:
        lvl = value[:, cursor:cursor + H_l * W_l]   # (B, H*W, n_heads, head_dim)
        cursor += H_l * W_l
        # Reshape to (B*n_heads, head_dim, H, W) for grid_sample
        lvl = lvl.permute(0, 2, 3, 1).reshape(B * n_heads, head_dim, H_l, W_l)
        value_lvls.append(lvl)

    # grid_sample expects coords in [-1, 1]; shape (N, H_out, W_out, 2)
    # We treat (n_query, n_points) as a 2D grid for sampling.
    sampled_lvls = []
    for lvl_id, (H_l, W_l) in enumerate(spatial_shapes):
        # (B, N_query, n_heads, n_points, 2)
        lvl_loc = sampling_locations[:, :, :, lvl_id]
        # → (B*n_heads, N_query, n_points, 2)
        lvl_loc = lvl_loc.permute(0, 2, 1, 3, 4).reshape(B * n_heads, N_query, n_points, 2)
        # grid_sample expects normalized [-1, 1]
        lvl_loc = 2.0 * lvl_loc - 1.0
        # (B*n_heads, head_dim, N_query, n_points)
        sampled = F.grid_sample(value_lvls[lvl_id], lvl_loc,
                                  mode="bilinear", padding_mode="zeros",
                                  align_corners=False)
        sampled_lvls.append(sampled)
    # Stack levels: (B*n_heads, head_dim, N_query, n_levels, n_points)
    sampled = torch.stack(sampled_lvls, dim=-2)

    # Apply attention weights and sum over (n_levels, n_points)
    # attention_weights: (B, N_query, n_heads, n_levels, n_points)
    # → (B*n_heads, 1, N_query, n_levels, n_points)
    aw = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        B * n_heads, 1, N_query, n_levels, n_points,
    )
    out = (sampled * aw).sum(dim=(-1, -2))   # (B*n_heads, head_dim, N_query)

    # → (B, N_query, n_heads * head_dim)
    out = out.view(B, n_heads, head_dim, N_query).permute(0, 3, 1, 2)
    return out.reshape(B, N_query, n_heads * head_dim)


class MSDeformCrossAttn(nn.Module):
    """Multi-Scale Deformable Cross-Attention.

    Used by both Injector (Q=ViT tokens, KV=multi-scale pyramid) and
    Extractor (Q=multi-scale pyramid, KV=ViT single-scale features).

    For each query, predicts `n_heads * n_levels * n_points` 2D sampling
    offsets relative to a reference point, plus softmax attention weights.
    Gathers the value at those locations and aggregates.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 n_levels: int = 1, n_points: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points
        head_dim = embed_dim // num_heads

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.head_dim = head_dim

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Uniform offset init around the reference point — matches the original
        # MSDeformAttn paper init for stable training.
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]
                      ).view(self.num_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, value: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: list[tuple[int, int]]) -> torch.Tensor:
        """Args:
          query:             (B, N_q, C)
          value:             (B, N_v, C) — flattened+concatenated across all levels
          reference_points:  (B, N_q, n_levels, 2) in [0,1]
          spatial_shapes:    list of (H_l, W_l) per level. Sum of H_l*W_l == N_v.
        """
        B, N_q, _ = query.shape
        N_v = value.shape[1]
        assert N_v == sum(H * W for H, W in spatial_shapes), \
            "value token count does not match sum of spatial_shapes"

        # Project value, reshape to (B, N_v, n_heads, head_dim)
        v = self.value_proj(value).view(B, N_v, self.num_heads, self.head_dim)

        # Predict sampling offsets and attention weights from query
        # offsets: (B, N_q, n_heads, n_levels, n_points, 2)
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.n_levels, self.n_points, 2,
        )
        # weights: (B, N_q, n_heads, n_levels * n_points) → softmax → reshape
        attn_w = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.n_levels * self.n_points,
        )
        attn_w = F.softmax(attn_w, dim=-1).view(
            B, N_q, self.num_heads, self.n_levels, self.n_points,
        )

        # Sampling locations = reference_points + offsets / spatial_shape
        # reference_points: (B, N_q, n_levels, 2) → broadcast to (B, N_q, 1, n_levels, 1, 2)
        # offset_normalizer per level: (n_levels, 2) — divide offset by W or H to normalize.
        offset_normalizer = torch.tensor(
            [[W, H] for H, W in spatial_shapes],
            device=offsets.device, dtype=offsets.dtype,
        )                                   # (n_levels, 2)
        sampling_loc = (
            reference_points[:, :, None, :, None, :]
            + offsets / offset_normalizer[None, None, None, :, None, :]
        )                                   # (B, N_q, n_heads, n_levels, n_points, 2)

        out = _ms_deform_attn(v, spatial_shapes, sampling_loc, attn_w)
        return self.output_proj(out)


def _build_reference_points(spatial_shapes: list[tuple[int, int]],
                              device: torch.device,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build reference points for queries that come FROM the multi-scale pyramid
    (i.e., extractor queries). Each token gets one reference point per level
    proportional to its position. Returns (N_q_total, n_levels, 2)."""
    ref_per_lvl = []
    for H, W in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype) / H,
            torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype) / W,
            indexing="ij",
        )
        ref = torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], -1)  # (H*W, 2)
        ref_per_lvl.append(ref)
    ref = torch.cat(ref_per_lvl, dim=0)               # (N_total, 2)
    # Replicate across levels
    n_levels = len(spatial_shapes)
    ref = ref.unsqueeze(1).expand(-1, n_levels, -1)   # (N_total, n_levels, 2)
    return ref.contiguous()


def _build_vit_reference_points(H_p: int, W_p: int,
                                  n_levels: int,
                                  device: torch.device,
                                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build reference points for ViT patch tokens (the queries during injection).
    Each ViT token attends to ALL n_levels of the spatial pyramid.
    Returns (H_p*W_p, n_levels, 2)."""
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_p - 0.5, H_p, device=device, dtype=dtype) / H_p,
        torch.linspace(0.5, W_p - 0.5, W_p, device=device, dtype=dtype) / W_p,
        indexing="ij",
    )
    ref = torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], -1)   # (N, 2)
    return ref.unsqueeze(1).expand(-1, n_levels, -1).contiguous()    # (N, n_levels, 2)


# ══════════════════════════════════════════════════════════════════════
# Spatial Prior Module — small CNN producing multi-scale features
# ══════════════════════════════════════════════════════════════════════

class SpatialPriorModule(nn.Module):
    """Small ResNet-like CNN that runs in PARALLEL to the ViT, producing a
    multi-scale feature pyramid (c1@stride4, c2@stride8, c3@stride16,
    c4@stride32). All outputs are projected to `embed_dim` channels so they
    interact with ViT tokens at the same width.

    Channel progression: 3 → inplanes → 2*inplanes → 4*inplanes → 4*inplanes
    Final 1x1 projections to embed_dim.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 1280,
                 inplanes: int = 64) -> None:
        super().__init__()
        # Stem: image → stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, 3, 2, 1, bias=False),
            nn.GroupNorm(8, inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inplanes), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        # stride 4 → 8 (c2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, 2, 1, bias=False),
            nn.GroupNorm(8, 2 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 8 → 16 (c3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, 2, 1, bias=False),
            nn.GroupNorm(8, 4 * inplanes), nn.ReLU(inplace=True),
        )
        # stride 16 → 32 (c4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, 2, 1, bias=False),
            nn.GroupNorm(8, 4 * inplanes), nn.ReLU(inplace=True),
        )
        # 1x1 projections to embed_dim
        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1)         # c1 stride 4
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, 1)     # c2 stride 8
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, 1)     # c3 stride 16
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, 1)     # c4 stride 32

    def forward(self, x: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (c1, c2, c3, c4) at strides (4, 8, 16, 32), all in `embed_dim`."""
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return self.fc1(c1), self.fc2(c2), self.fc3(c3), self.fc4(c4)


# ══════════════════════════════════════════════════════════════════════
# Injector — multi-scale pyramid → ViT tokens (cross-attention)
# ══════════════════════════════════════════════════════════════════════

class Injector(nn.Module):
    """ViT patch tokens (queries) attend to the multi-scale spatial pyramid
    (keys/values, 3 levels: c2/c3/c4). The attended feature is added back to
    the ViT tokens (residual), gated by a small LayerScale-like factor that
    starts near zero so the ViT is undisturbed at init."""

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 n_points: int = 4, init_values: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = MSDeformCrossAttn(embed_dim, num_heads,
                                        n_levels=3, n_points=n_points)
        # Per-channel learnable scale (init near zero so injection is gentle)
        self.gamma = nn.Parameter(init_values * torch.ones(embed_dim))

    def forward(self, vit_tokens: torch.Tensor,
                pyramid_tokens: torch.Tensor,
                vit_reference: torch.Tensor,
                pyramid_shapes: list[tuple[int, int]]) -> torch.Tensor:
        """vit_tokens:      (B, N_vit, C) — ViT PATCH tokens only (no CLS/register)
        pyramid_tokens:    (B, N_pyr, C) — flattened+concatenated c2/c3/c4
        vit_reference:     (B, N_vit, 3, 2) — ref points for each ViT query × 3 levels
        pyramid_shapes:    [(H2,W2), (H3,W3), (H4,W4)]
        """
        attn_out = self.attn(
            query=self.norm_q(vit_tokens),
            value=self.norm_kv(pyramid_tokens),
            reference_points=vit_reference,
            spatial_shapes=pyramid_shapes,
        )
        return vit_tokens + self.gamma * attn_out


# ══════════════════════════════════════════════════════════════════════
# Extractor — ViT tokens → multi-scale pyramid (cross-attention) + FFN
# ══════════════════════════════════════════════════════════════════════

class Extractor(nn.Module):
    """Multi-scale pyramid tokens (queries) attend to ViT patch tokens (single
    scale, keys/values). Outputs are added back to the pyramid (residual) +
    a small FFN block to mix channels."""

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 n_points: int = 4, ffn_ratio: float = 0.25,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = MSDeformCrossAttn(embed_dim, num_heads,
                                        n_levels=1, n_points=n_points)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        ffn_dim = max(16, int(embed_dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, pyramid_tokens: torch.Tensor,
                vit_tokens: torch.Tensor,
                pyramid_reference: torch.Tensor,
                vit_shape: tuple[int, int]) -> torch.Tensor:
        """pyramid_tokens:     (B, N_pyr, C) — flattened c2/c3/c4
        vit_tokens:           (B, N_vit, C) — ViT PATCH tokens at single scale
        pyramid_reference:    (B, N_pyr, 1, 2) — ref point per query for the 1 ViT level
        vit_shape:            (H_p, W_p) — ViT patch grid size
        """
        attn_out = self.attn(
            query=self.norm_q(pyramid_tokens),
            value=self.norm_kv(vit_tokens),
            reference_points=pyramid_reference,
            spatial_shapes=[vit_shape],
        )
        pyramid_tokens = pyramid_tokens + attn_out
        pyramid_tokens = pyramid_tokens + self.ffn(self.norm_ffn(pyramid_tokens))
        return pyramid_tokens


# ══════════════════════════════════════════════════════════════════════
# Interaction Block — one (extractor + injector) pair
# ══════════════════════════════════════════════════════════════════════

class InteractionBlock(nn.Module):
    """One stage of bidirectional ViT ↔ pyramid interaction. Performed BETWEEN
    consecutive ViT blocks at chosen indexes (typically the last block of each
    quartile of the ViT depth).

    Order: Extractor (ViT → pyramid) FIRST, then Injector (pyramid → ViT)
    — this matches the official ViT-Adapter implementation (extractor first
    so the pyramid sees the just-updated ViT features before re-injecting)."""

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 n_points: int = 4, init_values: float = 0.0,
                 ffn_ratio: float = 0.25, dropout: float = 0.0) -> None:
        super().__init__()
        self.extractor = Extractor(embed_dim, num_heads, n_points,
                                     ffn_ratio=ffn_ratio, dropout=dropout)
        self.injector = Injector(embed_dim, num_heads, n_points,
                                   init_values=init_values)

    def forward(self, vit_patches: torch.Tensor,
                pyramid_tokens: torch.Tensor,
                vit_shape: tuple[int, int],
                pyramid_shapes: list[tuple[int, int]],
                vit_reference: torch.Tensor,
                pyramid_reference: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns updated (vit_patches, pyramid_tokens)."""
        # Step 1: extract ViT → pyramid
        pyramid_tokens = self.extractor(
            pyramid_tokens, vit_patches,
            pyramid_reference=pyramid_reference,
            vit_shape=vit_shape,
        )
        # Step 2: inject pyramid → ViT
        vit_patches = self.injector(
            vit_patches, pyramid_tokens,
            vit_reference=vit_reference,
            pyramid_shapes=pyramid_shapes,
        )
        return vit_patches, pyramid_tokens


# ══════════════════════════════════════════════════════════════════════
# Helpers for flatten/unflatten the multi-scale pyramid
# ══════════════════════════════════════════════════════════════════════

def flatten_pyramid(c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor,
                     ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Each c_l: (B, C, H_l, W_l). Returns:
      flat: (B, sum(H_l*W_l), C) — concat order c2, c3, c4
      shapes: [(H2,W2), (H3,W3), (H4,W4)]
    """
    flats = []
    shapes = []
    for c in [c2, c3, c4]:
        _, _, H, W = c.shape
        shapes.append((H, W))
        flats.append(c.flatten(2).transpose(1, 2))   # (B, H*W, C)
    return torch.cat(flats, dim=1), shapes


def unflatten_pyramid(flat: torch.Tensor,
                       shapes: list[tuple[int, int]],
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of flatten_pyramid. flat: (B, N_total, C) → (c2, c3, c4)."""
    B, _, C = flat.shape
    out: list[torch.Tensor] = []
    cursor = 0
    for H, W in shapes:
        n = H * W
        tok = flat[:, cursor:cursor + n]            # (B, H*W, C)
        spatial = tok.transpose(1, 2).reshape(B, C, H, W)
        out.append(spatial.contiguous())
        cursor += n
    return out[0], out[1], out[2]
