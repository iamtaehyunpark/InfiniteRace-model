"""
Training loop — Levels 2 and 3.

Level 2: large driving video pretraining (nuScenes / Waymo)
Level 3: street-view fine-tuning on CueEngine-generated data

Usage:
    python -m world_model.train \
        --data_dir training_data/ \
        --checkpoint_dir checkpoints/ \
        --steps 50000
"""
import argparse
import itertools
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from world_model.config import ModelConfig, TrainConfig
from world_model.dataset import StreetViewDataset
from world_model.losses import PerceptualLoss, training_loss
from world_model.model import InfiniteRaceWorldModel

logger = logging.getLogger(__name__)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train InfiniteRace world model.")
    p.add_argument("--data_dir", type=str, default="training_data/")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--no_wandb", action="store_true")
    return p


def _setup_logging(use_wandb: bool, config: TrainConfig):
    """Initialise wandb or TensorBoard logger."""
    if use_wandb:
        try:
            import wandb
            wandb.init(project="infiniterace-worldmodel", config=vars(config))
            return wandb
        except Exception:
            logger.warning("wandb init failed, falling back to TensorBoard")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(config.checkpoint_dir, "tb_logs"))
        return writer
    except ImportError:
        logger.warning("Neither wandb nor TensorBoard available — logging to stdout only")
        return None


def _log_scalars(log_obj, step: int, metrics: dict):
    """Log scalar metrics to wandb or TensorBoard."""
    if log_obj is None:
        return

    # wandb
    if hasattr(log_obj, 'log'):
        log_obj.log(metrics, step=step)
        return

    # TensorBoard SummaryWriter
    if hasattr(log_obj, 'add_scalar'):
        for k, v in metrics.items():
            log_obj.add_scalar(k, v, step)


def _log_images(log_obj, step: int, images: dict):
    """Log image grids to wandb or TensorBoard."""
    if log_obj is None:
        return

    import torchvision.utils as vutils

    if hasattr(log_obj, 'log'):
        # wandb
        import wandb
        wandb_imgs = {}
        for k, tensor in images.items():
            grid = vutils.make_grid(tensor[:4], nrow=4, normalize=True, value_range=(-1, 1))
            wandb_imgs[k] = wandb.Image(grid.permute(1, 2, 0).cpu().numpy())
        log_obj.log(wandb_imgs, step=step)
    elif hasattr(log_obj, 'add_image'):
        for k, tensor in images.items():
            grid = vutils.make_grid(tensor[:4], nrow=4, normalize=True, value_range=(-1, 1))
            log_obj.add_image(k, grid, step)


def save_checkpoint(
    model: InfiniteRaceWorldModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
):
    """Save training checkpoint."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save({
        'step': step,
        'config': model.config,
        'unet': model.unet.state_dict(),
        'action_mlp': model.action_mlp.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)
    logger.info("Checkpoint saved: %s", path)


def train(config: Optional[TrainConfig] = None):
    """Main training loop."""
    if config is None:
        config = TrainConfig()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # Model
    model = InfiniteRaceWorldModel(config.model).to(device)

    # Freeze VAE
    model.vae.requires_grad_(False)

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
    logger.info("Dataset: %d training samples", len(dataset))

    # Optimizer — only trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.steps)

    # Losses
    lpips_fn = PerceptualLoss().to(device)
    lpips_fn.eval()

    # Mixed precision
    scaler = GradScaler(enabled=config.mixed_precision)

    # Logging
    use_wandb = True
    log_obj = _setup_logging(use_wandb, config)

    # Training loop
    step = 0
    t0 = time.time()

    for batch in itertools.cycle(loader):
        if step >= config.steps:
            break

        cue1 = batch['cue1'].to(device)                        # (B, 3, 256, 256)
        cue2 = batch['cue2'].to(device)                        # (B, 3, 256, 256)
        pos_map = batch['pos_map'].permute(0, 2, 3, 1).to(device)  # (B, 256, 256, 2)
        action = batch['action'].to(device)                    # (B, 3)
        target = batch['target'].to(device)                    # (B, 3, 256, 256)

        with autocast(enabled=config.mixed_precision):
            pred = model(cue1, cue2, pos_map, action)
            # cue1 is the warped previous frame — used as the 'warp' reference
            # for masked_l1_loss weighting (high weight where warp ≠ target)
            loss = training_loss(pred, target, cue1, lpips_fn, step)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                step, loss.item(), lr, elapsed,
            )
            _log_scalars(log_obj, step, {
                'loss/total': loss.item(),
                'lr': lr,
            })

        # Image logging
        if step % 1000 == 0 and step > 0:
            residual = (pred - target).abs() * 3.0
            _log_images(log_obj, step, {
                'cue1_warp': cue1,
                'cue2_anchor': cue2,
                'pred': pred,
                'target': target,
                'residual': residual,
            })

        # Checkpointing
        if step % config.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, config.checkpoint_dir)

        step += 1

    # Final checkpoint
    save_checkpoint(model, optimizer, step, config.checkpoint_dir)
    logger.info("Training complete — %d steps", step)


def main():
    args = _make_parser().parse_args()

    config = TrainConfig(
        model=ModelConfig(),
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    train(config)


if __name__ == "__main__":
    main()
