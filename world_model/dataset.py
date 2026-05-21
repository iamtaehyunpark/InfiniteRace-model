"""
Street-view sequence dataset — loads pre-extracted CueData .npz files.

Each .npz contains one training sample with:
  warped_frame       (256, 256, 3) uint8 BGR  — Cue 1
  anchor_crop        (256, 256, 3) uint8 BGR  — Cue 2a
  anchor_pos_map     (256, 256, 2) float32    — Cue 2b
  action_vector_norm (3,) float32             — Cue 3
  target_frame       (256, 256, 3) uint8 BGR  — ground truth output
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _bgr_to_rgb_norm(bgr: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR (H, W, 3) to float32 RGB (3, H, W) in [-1, 1]."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))       # (3, H, W)


def _file_hash(path: Path) -> int:
    """Deterministic hash of filename for train/val split."""
    return int(hashlib.md5(path.name.encode()).hexdigest(), 16)


class StreetViewDataset(Dataset):
    """PyTorch Dataset for pre-extracted CueData .npz files.

    Args:
        data_dir: directory containing .npz files
        split:    'train' (90%) or 'val' (10%)
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        all_files = sorted(self.data_dir.glob("*.npz"))

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in '{data_dir}'. "
                f"Run generate_dataset.py first."
            )

        # Deterministic 90/10 split by filename hash
        train_files = []
        val_files = []
        for f in all_files:
            if _file_hash(f) % 10 < 9:
                train_files.append(f)
            else:
                val_files.append(f)

        self.files = train_files if split == "train" else val_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = np.load(self.files[idx])

        warped = _bgr_to_rgb_norm(sample['warped_frame'])      # (3, 256, 256)
        anchor = _bgr_to_rgb_norm(sample['anchor_crop'])       # (3, 256, 256)
        target = _bgr_to_rgb_norm(sample['target_frame'])      # (3, 256, 256)
        pos_map = sample['anchor_pos_map']                     # (256, 256, 2)
        action = sample['action_vector_norm']                  # (3,)

        # pos_map: store as (2, 256, 256) for DataLoader collation;
        # model.forward re-permutes to (B, 256, 256, 2)
        pos_map_chw = np.ascontiguousarray(
            pos_map.transpose(2, 0, 1),                        # (2, 256, 256)
        )

        return {
            'cue1':    torch.from_numpy(warped),
            'cue2':    torch.from_numpy(anchor),
            'pos_map': torch.from_numpy(pos_map_chw),
            'action':  torch.from_numpy(action.copy()),
            'target':  torch.from_numpy(target),
        }
