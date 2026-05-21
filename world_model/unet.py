"""
Thin UNet backbone — convolutional encoder/decoder surrounding the
transformer bottleneck.

Operates entirely at 32×32 latent resolution (no spatial down/upsampling).
The encoder expands channels from latent_channels (4) → 512, the decoder
contracts back.  The transformer bottleneck replaces traditional UNet
middle blocks.

**No skip connections** — the model should only deviate from the warped
frame where the anchor contradicts it, not preserve fine spatial detail
across the entire image.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from world_model.transformer import TransformerBottleneck

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Standard pre-activation residual block.

    Conv 3×3 → GroupNorm(8) → SiLU → Conv 3×3 → GroupNorm(8) → residual add.
    Uses a 1×1 shortcut projection when input/output channels differ.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act_out(h + self.shortcut(x))


class ThinUNet(nn.Module):
    """Thin convolutional backbone wrapping the transformer bottleneck.

    All processing happens at the 32×32 latent spatial resolution.
    The encoder increases channel depth through residual blocks,
    the transformer performs cross-attention reasoning, and the
    decoder projects back to the original latent channel count.

    Architecture:
        Encoder: 4 → 64 → 128 → 256 → 512 (all at 32×32)
        Bottleneck: TransformerBottleneck (d_model=512)
        Decoder: 512 → 256 → 128 → 64 → 4 (all at 32×32)
    """

    def __init__(
        self,
        latent_channels: int = 4,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        # --- Encoder ---
        self.enc_conv_in = nn.Conv2d(latent_channels, 64, 3, padding=1)
        self.enc_block1 = ResBlock(64, 128)
        self.enc_block2 = ResBlock(128, 256)
        self.enc_block3 = ResBlock(256, 512)

        # --- Transformer bottleneck ---
        self.bottleneck = TransformerBottleneck(
            latent_channels=512,     # channel dim entering transformer
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        # --- Decoder ---
        self.dec_block1 = ResBlock(512, 256)
        self.dec_block2 = ResBlock(256, 128)
        self.dec_block3 = ResBlock(128, 64)
        self.dec_conv_out = nn.Conv2d(64, latent_channels, 3, padding=1)

    def forward(
        self,
        z_warp: Tensor,        # (B, 4, 32, 32)
        z_anchor: Tensor,      # (B, 4, 32, 32)
        pos_enc: Tensor,       # (B, 1024, d_model)
        adaln_params: list,    # [(scale, shift), ...] per layer
    ) -> Tensor:               # (B, 4, 32, 32)

        # --- Encoder: channel expansion ---
        h = self.enc_conv_in(z_warp)           # (B,  64, 32, 32)
        h = self.enc_block1(h)                 # (B, 128, 32, 32)
        h = self.enc_block2(h)                 # (B, 256, 32, 32)
        h_enc = self.enc_block3(h)             # (B, 512, 32, 32)

        # Encode anchor through the same channel expansion path
        a = self.enc_conv_in(z_anchor)
        a = self.enc_block1(a)
        a = self.enc_block2(a)
        a_enc = self.enc_block3(a)             # (B, 512, 32, 32)

        # --- Transformer bottleneck ---
        h_mid = self.bottleneck(h_enc, a_enc, pos_enc, adaln_params)  # (B, 512, 32, 32)

        # Residual connection around the bottleneck
        h_mid = h_mid + h_enc

        # --- Decoder: channel contraction ---
        h = self.dec_block1(h_mid)             # (B, 256, 32, 32)
        h = self.dec_block2(h)                 # (B, 128, 32, 32)
        h = self.dec_block3(h)                 # (B,  64, 32, 32)
        z_out = self.dec_conv_out(h)           # (B,   4, 32, 32)

        # Residual learning: predict the *correction* to z_warp
        return z_out + z_warp

    def load_pretrained_partial(
        self,
        sd_unet_state_dict: dict,
    ) -> tuple[list[str], list[str]]:
        """Selectively load weights from an SD 1.5 UNet state dict.

        Matches keys by shape where architecturally compatible.
        Returns (loaded_keys, skipped_keys).
        """
        own_state = self.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []

        # Build a map from shape → list of SD keys for fuzzy matching
        sd_by_shape: dict[tuple, list[tuple[str, Tensor]]] = {}
        for k, v in sd_unet_state_dict.items():
            sd_by_shape.setdefault(v.shape, []).append((k, v))

        for own_key, own_param in own_state.items():
            # Skip bottleneck (transformer) — our custom architecture
            if 'bottleneck' in own_key:
                skipped.append(own_key)
                continue

            # Try exact name match first (unlikely but cheap)
            if own_key in sd_unet_state_dict:
                sd_val = sd_unet_state_dict[own_key]
                if sd_val.shape == own_param.shape:
                    own_state[own_key] = sd_val
                    loaded.append(own_key)
                    continue

            # Try shape-based fuzzy match (ResBlock convs, group norms)
            candidates = sd_by_shape.get(own_param.shape, [])
            if len(candidates) == 1:
                own_state[own_key] = candidates[0][1]
                loaded.append(f"{own_key} ← {candidates[0][0]}")
            else:
                skipped.append(own_key)

        self.load_state_dict(own_state, strict=False)
        logger.info("Pretrained UNet partial load: %d loaded, %d skipped",
                     len(loaded), len(skipped))
        for k in loaded[:10]:
            logger.debug("  loaded: %s", k)
        for k in skipped[:10]:
            logger.debug("  skipped: %s", k)

        return loaded, skipped
