"""
LCM consistency distillation — Level 4 training.

Trains the model to perform accurate 2-step inference by enforcing
consistency: model(noisy_z @ t1) ≈ teacher(noisy_z @ t2) in latent space.

The teacher is an EMA copy of the student, updated after each step.

Usage:
    python -m world_model.distill \
        --checkpoint checkpoints/level3_final.pt \
        --data_dir training_data/ \
        --steps 20000
"""
import argparse
import copy
import itertools
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from world_model.config import DistillConfig, ModelConfig
from world_model.dataset import StreetViewDataset
from world_model.lcm import _beta_schedule, lcm_consistency_loss
from world_model.losses import masked_l1_loss
from world_model.model import InfiniteRaceWorldModel

logger = logging.getLogger(__name__)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LCM consistency distillation.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Level 3 checkpoint to distill from")
    p.add_argument("--data_dir", type=str, default="training_data/")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--skip_steps", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    return p


class _EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay,
                )

    def apply_to(self, model: torch.nn.Module) -> None:
        """Copy shadow params into model (for teacher sync)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


def distill(config: DistillConfig | None = None):
    """LCM consistency distillation loop."""
    if config is None:
        config = DistillConfig()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load student from Level 3 checkpoint
    logger.info("Loading student from '%s'", config.checkpoint)
    student = InfiniteRaceWorldModel.load(config.checkpoint, device=str(device))
    student.train()

    # Create teacher as frozen EMA copy
    teacher = copy.deepcopy(student)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # EMA tracker
    ema = _EMA(student, decay=config.ema_decay)

    # Noise schedule
    betas = _beta_schedule(config.num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # Dataset
    dataset = StreetViewDataset(config.data_dir, split="train")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=config.lr_distill,
    )
    scaler = GradScaler(enabled=config.mixed_precision)

    step = 0
    t0 = time.time()

    for batch in itertools.cycle(loader):
        if step >= config.steps:
            break

        cue1 = batch['cue1'].to(device)
        cue2 = batch['cue2'].to(device)
        pos_map = batch['pos_map'].permute(0, 2, 3, 1).to(device)
        action = batch['action'].to(device)
        target = batch['target'].to(device)

        # Sample timestep pair: t1 > t2
        t1 = torch.randint(
            config.skip_steps, config.num_train_timesteps,
            (1,),
        ).item()
        t2 = t1 - config.skip_steps

        with autocast(enabled=config.mixed_precision):
            # Encode warped frame
            with torch.no_grad():
                z_warp = student.vae.encode(cue1)

            # Consistency loss
            loss_consistency = lcm_consistency_loss(
                model=student,
                teacher=teacher,
                z_warp=z_warp,
                cue1=cue1,
                cue2=cue2,
                pos_map=pos_map,
                action=action,
                t1=t1,
                t2=t2,
                alphas_cumprod=alphas_cumprod,
            )

            # Reconstruction loss to prevent collapse
            pred_student = student(cue1, cue2, pos_map, action)
            loss_recon = masked_l1_loss(pred_student, target, cue1)

            loss = loss_consistency + 0.5 * loss_recon

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # EMA update + sync to teacher
        ema.update(student)
        ema.apply_to(teacher)

        if step % 100 == 0:
            logger.info(
                "step=%d  loss=%.4f  consist=%.4f  recon=%.4f  elapsed=%.1fs",
                step, loss.item(), loss_consistency.item(),
                loss_recon.item(), time.time() - t0,
            )

        if step % 5000 == 0 and step > 0:
            path = Path(config.checkpoint_dir) / f"lcm_step_{step:07d}.pt"
            student.save(str(path))

        step += 1

    # Final save
    final_path = Path(config.checkpoint_dir) / "lcm_final.pt"
    student.save(str(final_path))
    logger.info("Distillation complete — %d steps, saved to %s", step, final_path)


def main():
    args = _make_parser().parse_args()
    config = DistillConfig(
        checkpoint=args.checkpoint,
        lr_distill=args.lr,
        steps=args.steps,
        skip_steps=args.skip_steps,
        ema_decay=args.ema_decay,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    distill(config)


if __name__ == "__main__":
    main()
