"""
LCM (Latent Consistency Model) — consistency distillation scheduler + sampler.

This makes 2-step inference possible.  Key insight for this use case:
unlike text-to-image LCM (starting from pure noise), here we start from
z_warp (the encoded warped previous frame, already ~85-90% correct)
plus a small amount of noise.  The LCM's job is to find the consistent
clean image in 2 steps from this informed starting point.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _beta_schedule(num_train_timesteps: int = 1000,
                   schedule: str = "scaled_linear") -> Tensor:
    """Compute β schedule matching SD 1.5."""
    if schedule == "scaled_linear":
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5,
            num_train_timesteps, dtype=torch.float64,
        ) ** 2
    elif schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas.float()


class LCMSampler:
    """LCM multi-step sampler for informed-start denoising.

    Instead of starting from pure noise, we start from ``z_warp``
    plus calibrated noise and denoise in ``steps`` consistency steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = num_train_timesteps

        betas = _beta_schedule(num_train_timesteps, beta_schedule)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)   # (T,)

        # σ(t) = sqrt((1 - ᾱ_t) / ᾱ_t)
        self.sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def _get_timesteps(self, steps: int) -> list[int]:
        """Generate LCM timestep schedule.

        For ``steps=2``, returns e.g. [799, 399] — spaced roughly evenly
        across the training schedule, skipping the final clean step.
        """
        step_ratio = self.num_train_timesteps // steps
        timesteps = [
            (self.num_train_timesteps - 1) - i * step_ratio
            for i in range(steps)
        ]
        return [max(0, t) for t in timesteps]

    def _add_noise(self, z: Tensor, t: int) -> Tensor:
        """Add noise to latent at timestep t."""
        acp = self.alphas_cumprod.to(z.device)
        alpha_t = acp[t]
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        noise = torch.randn_like(z)
        return sqrt_alpha * z + sqrt_one_minus * noise

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        cue1: Tensor,       # (B, 3, 256, 256) warped frame
        cue2: Tensor,       # (B, 3, 256, 256) anchor crop
        pos_map: Tensor,    # (B, 256, 256, 2)
        action: Tensor,     # (B, 3)
        steps: int = 2,
    ) -> Tensor:
        """LCM multi-step sampling from informed starting point.

        1. Encode z_warp, add small noise calibrated to first timestep
        2. For each timestep: predict clean x0, re-noise to next timestep
        3. Decode final prediction

        Args:
            model:   InfiniteRaceWorldModel instance
            cue1:    warped previous frame
            cue2:    anchor crop
            pos_map: per-pixel spherical coords
            action:  normalised action vector
            steps:   number of LCM steps (default 2)

        Returns:
            keyframe: (B, 3, 256, 256) float32 [-1, 1]
        """
        device = cue1.device
        acp = self.alphas_cumprod.to(device)
        timesteps = self._get_timesteps(steps)

        # Encode warped frame to latent
        z_warp = model.vae.encode(cue1)          # (B, 4, 32, 32)

        # Start from z_warp + calibrated noise at first timestep
        z_noisy = self._add_noise(z_warp, timesteps[0])

        for i, t in enumerate(timesteps):
            # The model predicts the clean output given noisy input
            noise_level = (1.0 - acp[t]).sqrt().item()
            pred = model.forward(
                cue1, cue2, pos_map, action,
                noise_level=noise_level,
                z_warp_override=z_noisy,
            )

            # Re-encode prediction to latent space
            z_pred = model.vae.encode(pred)

            if i < len(timesteps) - 1:
                # Re-noise to next timestep for consistency step
                next_t = timesteps[i + 1]
                z_noisy = self._add_noise(z_pred, next_t)
            else:
                # Final step: decode clean prediction
                return pred

        return pred   # fallback (unreachable with steps >= 1)


def lcm_consistency_loss(
    model: nn.Module,
    teacher: nn.Module,
    z_warp: Tensor,        # (B, 4, 32, 32)
    cue1: Tensor,          # (B, 3, 256, 256) for encoding context
    cue2: Tensor,          # (B, 3, 256, 256)
    pos_map: Tensor,       # (B, 256, 256, 2)
    action: Tensor,        # (B, 3)
    t1: int,               # current timestep
    t2: int,               # next timestep (t2 < t1)
    alphas_cumprod: Tensor,
) -> Tensor:
    """LCM consistency distillation loss.

    Forces model(noisy_z @ t1) ≈ teacher(noisy_z @ t2) in latent space.
    """
    device = z_warp.device
    acp = alphas_cumprod.to(device)

    # Noise for both timesteps (same noise, different amounts)
    noise = torch.randn_like(z_warp)

    sqrt_alpha_t1 = acp[t1].sqrt()
    sqrt_one_minus_t1 = (1.0 - acp[t1]).sqrt()
    noisy_t1 = sqrt_alpha_t1 * z_warp + sqrt_one_minus_t1 * noise

    sqrt_alpha_t2 = acp[t2].sqrt()
    sqrt_one_minus_t2 = (1.0 - acp[t2]).sqrt()
    noisy_t2 = sqrt_alpha_t2 * z_warp + sqrt_one_minus_t2 * noise

    # Student prediction at t1
    noise_level_t1 = sqrt_one_minus_t1.item()
    pred_student = model(
        cue1, cue2, pos_map, action,
        noise_level=noise_level_t1,
        z_warp_override=noisy_t1,
    )

    # Teacher prediction at t2 (no grad)
    noise_level_t2 = sqrt_one_minus_t2.item()
    with torch.no_grad():
        pred_teacher = teacher(
            cue1, cue2, pos_map, action,
            noise_level=noise_level_t2,
            z_warp_override=noisy_t2,
        )

    # Consistency in latent space
    # z_student needs grad to flow back to student model;
    # z_teacher is detached (teacher is frozen)
    z_student = model.vae.encode(pred_student)

    with torch.no_grad():
        z_teacher = model.vae.encode(pred_teacher)

    loss = F.mse_loss(z_student, z_teacher.detach())
    return loss
