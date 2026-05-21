"""
Spherical sinusoidal positional encoding for anchor crops.

Encodes per-pixel world-space (azimuth, elevation) coordinates into
sinusoidal embeddings that are added to anchor latent tokens before
cross-attention.  Standard 2D grid encodings are insufficient because
the anchor crop is a rectilinear projection of a sphere — the mapping
from pixel to angular position is non-linear.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def downsample_pos_map(pos_map: Tensor, size: int) -> Tensor:
    """Bilinear downsample of angular position map to match latent resolution.

    Args:
        pos_map: (B, H, W, 2)  float32 [azimuth_rad, elevation_rad]
        size:    target spatial size (e.g. 32)

    Returns:
        (B, size, size, 2) float32
    """
    # F.interpolate expects (B, C, H, W)
    x = pos_map.permute(0, 3, 1, 2)                              # (B, 2, H, W)
    x = F.interpolate(x, size=(size, size), mode='bilinear',
                       align_corners=False)                        # (B, 2, s, s)
    return x.permute(0, 2, 3, 1)                                  # (B, s, s, 2)


class SphericalSinusoidal(nn.Module):
    """Parameter-free spherical sinusoidal positional encoding.

    For each pixel position (az, el), produces a ``d_model``-dim embedding
    using the standard sinusoidal formula with geometric frequency spacing,
    applied independently to azimuth and elevation, then concatenated.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        half_d = d_model // 2

        # Precompute inverse frequency vector: 1 / 10000^(2i/d)
        # Shape: (half_d // 2,)  — pairs of sin/cos per angular dimension
        freq_indices = torch.arange(0, half_d, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (freq_indices / half_d))
        self.register_buffer('inv_freq', inv_freq)                 # (half_d // 2,)

    def forward(self, pos_map: Tensor) -> Tensor:
        """
        Args:
            pos_map: (B, H, W, 2) float32 — [azimuth_rad, elevation_rad]

        Returns:
            (B, H*W, d_model) float32 — positional embeddings
        """
        B, H, W, _ = pos_map.shape
        az = pos_map[..., 0].reshape(B, H * W)                    # (B, N)
        el = pos_map[..., 1].reshape(B, H * W)                    # (B, N)

        # (B, N, 1) @ (1, F) → (B, N, F) where F = half_d // 2
        inv_freq = self.inv_freq.to(pos_map.device, pos_map.dtype)
        az_scaled = az.unsqueeze(-1) * inv_freq.unsqueeze(0)      # (B, N, F)
        el_scaled = el.unsqueeze(-1) * inv_freq.unsqueeze(0)      # (B, N, F)

        # Interleave sin/cos for each angular dimension, then concatenate
        az_enc = torch.cat([az_scaled.sin(), az_scaled.cos()], dim=-1)  # (B, N, half_d)
        el_enc = torch.cat([el_scaled.sin(), el_scaled.cos()], dim=-1)  # (B, N, half_d)

        return torch.cat([az_enc, el_enc], dim=-1)                 # (B, N, d_model)
