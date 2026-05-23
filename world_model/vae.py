"""
VAE — Shared encoder / decoder wrapping Stable Diffusion 1.5 VAE.

The VAE is **always frozen** during training.  ``encode`` returns the
latent mean directly (no reparameterisation).  ``decode`` maps latents
back to pixel space clamped to [-1, 1].

8× spatial compression: (B, 3, 256, 256) ↔ (B, 4, 32, 32).
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# SD 1.5 VAE scaling factor (latent ← pixel * factor, pixel ← latent / factor)
_SD_SCALING_FACTOR = 0.18215


class VAE(nn.Module):
    """Thin wrapper around the Stable Diffusion VAE (AutoencoderKL).

    All parameters are frozen.  Uses the ``diffusers`` library to load
    pretrained weights from HuggingFace Hub or a local directory.
    """

    def __init__(self, checkpoint: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self._checkpoint = checkpoint
        self._loaded = False

        # Lazy load: weights are pulled on first forward / explicit call
        self.vae: Optional[nn.Module] = None

    # -----------------------------------------------------------------
    # Lazy weight loading
    # -----------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        from diffusers import AutoencoderKL

        logger.info("Loading SD VAE from '%s' …", self._checkpoint)
        self.vae = AutoencoderKL.from_pretrained(self._checkpoint)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self._loaded = True
        logger.info("SD VAE loaded and frozen  (%s parameters)",
                     f"{sum(p.numel() for p in self.vae.parameters()):,}")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def _move_to(self, device: torch.device) -> None:
        """Ensure VAE is on the correct device after lazy load."""
        if self.vae is not None and next(self.vae.parameters()).device != device:
            self.vae = self.vae.to(device)

    def encode(self, x: Tensor) -> Tensor:
        """Encode pixel-space image to latent.

        Args:
            x: (B, 3, H, W) float in [-1, 1]

        Returns:
            z: (B, 4, H//8, W//8) same dtype as input
        """
        self._ensure_loaded()
        self._move_to(x.device)
        # VAE weights stay float32; disable AMP so input dtype matches
        dev = x.device.type if x.device.type in ("cuda",) else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            dist = self.vae.encode(x.float()).latent_dist  # type: ignore[union-attr]
        return (dist.mean * _SD_SCALING_FACTOR).to(x.dtype)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to pixel-space image.

        Args:
            z: (B, 4, H//8, W//8) float

        Returns:
            x: (B, 3, H, W) same dtype as input, clamped to [-1, 1]
        """
        self._ensure_loaded()
        self._move_to(z.device)
        dev = z.device.type if z.device.type in ("cuda",) else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            x = self.vae.decode((z / _SD_SCALING_FACTOR).float()).sample  # type: ignore[union-attr]
        return x.clamp(-1.0, 1.0).to(z.dtype)

    # Allow module to be moved to device before first forward
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self.vae is not None:
            self.vae = self.vae.to(*args, **kwargs)
        return result
