"""
Transformer bottleneck — the core reasoning module.

Operates entirely at 32×32 latent resolution (1,024 tokens).
Uses **cross-attention only** (warped frame attends to anchor) —
no self-attention on warped frame tokens.  Action conditioning
is injected via AdaLN (Adaptive Layer Normalisation).
"""
import torch
import torch.nn as nn
from torch import Tensor


class _TransformerLayer(nn.Module):
    """Single transformer layer: AdaLN → cross-attn → FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_model,
            vdim=d_model,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        z_warp: Tensor,       # (B, N, D)
        z_anchor: Tensor,     # (B, N, D)  — already has pos encoding added
        scale: Tensor,        # (B, D)
        shift: Tensor,        # (B, D)
    ) -> Tensor:
        # --- AdaLN on warped tokens ---
        z_normed = self.norm1(z_warp)
        # scale/shift are (B, D) → broadcast over sequence dim
        z_adaln = z_normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # --- Cross-attention: warp queries attend to anchor keys/values ---
        attn_out, _ = self.cross_attn(
            query=z_adaln,
            key=z_anchor,
            value=z_anchor,
        )
        z_warp = z_warp + attn_out

        # --- Feed-forward ---
        z_warp = z_warp + self.ffn(self.norm2(z_warp))

        return z_warp


class TransformerBottleneck(nn.Module):
    """Multi-layer cross-attention transformer at latent resolution.

    z_warp tokens attend to z_anchor tokens (which carry spherical
    positional encoding).  Action conditioning enters through AdaLN
    scale/shift parameters on each layer.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Project from latent channels to d_model and back
        self.proj_in_warp = nn.Linear(latent_channels, d_model)
        self.proj_in_anchor = nn.Linear(latent_channels, d_model)
        self.proj_out = nn.Linear(d_model, latent_channels)

        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        z_warp: Tensor,          # (B, 4, 32, 32)
        z_anchor: Tensor,        # (B, 4, 32, 32)
        pos_enc: Tensor,         # (B, 1024, d_model)
        adaln_params: list,      # [(scale, shift), ...] per layer
    ) -> Tensor:                 # (B, 4, 32, 32)
        B, C, H, W = z_warp.shape

        # Flatten spatial → sequence: (B, C, H, W) → (B, H*W, C)
        z_warp_flat = z_warp.permute(0, 2, 3, 1).reshape(B, H * W, C)
        z_anchor_flat = z_anchor.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project to d_model
        z_warp_seq = self.proj_in_warp(z_warp_flat)      # (B, N, D)
        z_anchor_seq = self.proj_in_anchor(z_anchor_flat) # (B, N, D)

        # Add spherical positional encoding to anchor tokens only
        z_anchor_pos = z_anchor_seq + pos_enc             # (B, N, D)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            scale, shift = adaln_params[i]
            z_warp_seq = layer(z_warp_seq, z_anchor_pos, scale, shift)

        # Project back to latent channels and reshape
        z_out = self.proj_out(z_warp_seq)                 # (B, N, C)
        z_out = z_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return z_out
