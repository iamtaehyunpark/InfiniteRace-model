"""
InfiniteRaceWorldModel — full model class composing all components.

Takes three inputs per frame and produces one output:
  Input 1  — warped previous frame   (B, 3, 256, 256) float32 [-1, 1]
  Input 2a — anchor crop             (B, 3, 256, 256) float32 [-1, 1]
  Input 2b — anchor pos_map          (B, 256, 256, 2) float32 radians
  Input 3  — action vector           (B, 3)           float32 [-1, 1]
  Output   — keyframe                (B, 3, 256, 256) float32 [-1, 1]
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from world_model.config import ModelConfig
from world_model.vae import VAE
from world_model.unet import ThinUNet
from world_model.action_mlp import ActionMLP
from world_model.pos_encoding import SphericalSinusoidal, downsample_pos_map
from world_model.lcm import LCMSampler

logger = logging.getLogger(__name__)


class InfiniteRaceWorldModel(nn.Module):
    """Real-time latent diffusion world model for street-view navigation.

    The VAE is **always frozen** — only the UNet, transformer, and action
    MLP have trainable parameters.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # --- Components ---
        self.vae = VAE(checkpoint=config.vae_checkpoint)       # frozen
        self.unet = ThinUNet(
            latent_channels=config.latent_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
        self.action_mlp = ActionMLP(
            d_model=config.d_model,
            n_layers=config.n_layers,
        )
        self.pos_enc = SphericalSinusoidal(d_model=config.d_model)

        # LCM sampler for inference
        self.lcm_sampler = LCMSampler()

    def forward(
        self,
        cue1: Tensor,                   # (B, 3, 256, 256) warped frame
        cue2: Tensor,                   # (B, 3, 256, 256) anchor crop
        pos_map: Tensor,                # (B, 256, 256, 2) radians
        action: Tensor,                 # (B, 3) [-1, 1]
        noise_level: float = 0.0,       # for LCM training
        z_warp_override: Optional[Tensor] = None,  # pre-noised latent
    ) -> Tensor:                        # (B, 3, 256, 256) [-1, 1]
        """Full forward pass — training and single-step inference.

        Args:
            cue1:            warped previous frame
            cue2:            anchor crop
            pos_map:         per-pixel (azimuth, elevation) in radians
            action:          normalised action vector
            noise_level:     noise σ to add to z_warp (0 = none)
            z_warp_override: if provided, use this as z_warp instead of
                             encoding cue1 (used by LCM sampler)

        Returns:
            Predicted keyframe image in [-1, 1].
        """
        # 1. Encode inputs to latent space (VAE frozen)
        with torch.no_grad():
            if z_warp_override is not None:
                z_warp = z_warp_override
            else:
                z_warp = self.vae.encode(cue1)     # (B, 4, 32, 32)
            z_anchor = self.vae.encode(cue2)       # (B, 4, 32, 32)

        # 2. Add noise to z_warp for denoising training
        if noise_level > 0.0 and z_warp_override is None:
            z_warp = z_warp + torch.randn_like(z_warp) * noise_level

        # 3. Spherical positional encoding for anchor
        pos_map_small = downsample_pos_map(
            pos_map, size=self.config.latent_size,
        )                                          # (B, 32, 32, 2)
        pos_enc_out = self.pos_enc(pos_map_small)  # (B, 1024, d_model)

        # 4. Action → AdaLN parameters
        adaln_params = self.action_mlp(action)     # [(scale, shift), ...]

        # 5. UNet forward: encode, bottleneck, decode
        z_out = self.unet(
            z_warp, z_anchor, pos_enc_out, adaln_params,
        )                                          # (B, 4, 32, 32)

        # 6. Decode to pixel space — no torch.no_grad() here so gradients
        # flow back through z_out to the trainable UNet (VAE params are
        # frozen via requires_grad_(False) so no VAE gradients accumulate)
        out = self.vae.decode(z_out)               # (B, 3, 256, 256) [-1,1]

        return out

    @torch.inference_mode()
    def infer(
        self,
        cue1: Tensor,
        cue2: Tensor,
        pos_map: Tensor,
        action: Tensor,
        steps: int = 2,
    ) -> Tensor:
        """Production inference path: LCM multi-step denoising.

        Args:
            cue1:    (B, 3, 256, 256) warped frame
            cue2:    (B, 3, 256, 256) anchor crop
            pos_map: (B, 256, 256, 2) radians
            action:  (B, 3)
            steps:   number of LCM steps (default 2)

        Returns:
            keyframe: (B, 3, 256, 256) float32 [-1, 1]
        """
        return self.lcm_sampler.sample(
            self, cue1, cue2, pos_map, action, steps=steps,
        )

    # -----------------------------------------------------------------
    # Checkpoint management
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trainable weights (excluding frozen VAE)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config,
            'unet': self.unet.state_dict(),
            'action_mlp': self.action_mlp.state_dict(),
        }
        torch.save(state, path)
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "InfiniteRaceWorldModel":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = cls(config)
        model.unet.load_state_dict(ckpt['unet'])
        model.action_mlp.load_state_dict(ckpt['action_mlp'])
        logger.info("Model loaded from %s", path)
        return model.to(device)
