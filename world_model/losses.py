"""
Training losses — masked L1, perceptual (LPIPS), and combined.

The masked L1 loss weights per-pixel error by how much the warp
diverges from ground truth, focusing the model's learning on regions
where the warped frame is wrong rather than reliable areas.
"""
import torch
import torch.nn as nn
from torch import Tensor


def masked_l1_loss(pred: Tensor, target: Tensor, warp: Tensor) -> Tensor:
    """Per-pixel L1 loss weighted by warp error magnitude.

    Regions where warp ≈ target (reliable) get low weight — the model
    is not penalised for staying close to the warp there.  Regions
    where warp ≠ target (errors) get high weight — the model must
    learn to correct them.

    Args:
        pred:   (B, 3, 256, 256)  model prediction
        target: (B, 3, 256, 256)  ground-truth next frame
        warp:   (B, 3, 256, 256)  warped previous frame

    Returns:
        Scalar loss.
    """
    # Per-pixel warp error as importance weight
    warp_err = (warp - target).abs()                           # (B, 3, H, W)
    weight = warp_err / (warp_err.mean() + 1e-6)
    weight = weight.clamp(0.1, 5.0)

    pixel_loss = (pred - target).abs()                         # (B, 3, H, W)
    return (weight * pixel_loss).mean()


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss using a pretrained VGG backbone.

    The ``lpips`` package (Zhang et al., 2018) is used as the backend.
    The VGG network is frozen and used as a feature extractor only.
    """

    def __init__(self):
        super().__init__()
        self._lpips = None   # lazy-loaded

    def _ensure_loaded(self) -> None:
        if self._lpips is not None:
            return
        import lpips
        self._lpips = lpips.LPIPS(net='vgg')
        self._lpips.requires_grad_(False)
        self._lpips.eval()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute LPIPS distance.

        Both inputs in [-1, 1].

        Returns:
            Scalar mean LPIPS distance across the batch.
        """
        self._ensure_loaded()
        # Move LPIPS network to same device if needed
        if next(self._lpips.parameters()).device != pred.device:
            self._lpips = self._lpips.to(pred.device)
        return self._lpips(pred, target).mean()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self._lpips is not None:
            self._lpips = self._lpips.to(*args, **kwargs)
        return result


def training_loss(
    pred: Tensor,
    target: Tensor,
    warp: Tensor,
    lpips_fn: PerceptualLoss,
    step: int,
) -> Tensor:
    """Combined training loss with perceptual weight ramp-up.

    During the first 10k steps only L1 is used (stable reconstruction).
    Perceptual loss ramps linearly from 0 → 0.5 weight over 10k steps.

    Args:
        pred:     model prediction
        target:   ground-truth
        warp:     warped previous frame
        lpips_fn: PerceptualLoss module
        step:     current training step

    Returns:
        Scalar combined loss.
    """
    l1 = masked_l1_loss(pred, target, warp)

    # Ramp up perceptual weight over first 10k steps
    perc_weight = min(0.5, 0.5 * step / 10_000)

    if perc_weight > 0.0:
        perc = lpips_fn(pred, target)
        return l1 + perc_weight * perc

    return l1
