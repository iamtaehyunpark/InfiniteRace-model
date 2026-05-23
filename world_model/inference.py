"""
Real-time inference runner — asynchronous background thread.

Consumes packets from ``wm_queue``, runs 2-step LCM inference,
and pushes keyframes to ``jitter_buffer`` for the rendering pipeline.
"""
import collections
import logging
import threading
from queue import Empty, Queue
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from world_model.model import InfiniteRaceWorldModel

logger = logging.getLogger(__name__)


def normalize(img_bgr: np.ndarray) -> Tensor:
    """Convert uint8 BGR (H, W, 3) to float32 RGB (1, 3, H, W) in [-1, 1]."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return t.unsqueeze(0)                                          # (1, 3, H, W)


def denormalize(tensor: Tensor) -> np.ndarray:
    """Convert float32 RGB (1, 3, H, W) in [-1, 1] to uint8 BGR (H, W, 3)."""
    x = tensor.clamp(-1, 1).cpu().numpy()[0]                       # (3, H, W)
    x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    x = x.transpose(1, 2, 0)                                       # (H, W, 3)
    return x[:, :, ::-1].copy()                                     # RGB → BGR


class JitterBuffer:
    """Thread-safe buffer holding the last N keyframes.

    Provides ``push`` / ``get_two_latest`` for the rendering thread
    to always have the most recent pair of keyframes available for
    interpolation.
    """

    def __init__(self, maxlen: int = 4):
        self._buffer: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, item: dict) -> None:
        with self._lock:
            self._buffer.append(item)

    def get_two_latest(self) -> tuple[Optional[dict], Optional[dict]]:
        """Return the two most recent items, or (None, None) if empty."""
        with self._lock:
            if len(self._buffer) >= 2:
                return self._buffer[-2], self._buffer[-1]
            elif len(self._buffer) == 1:
                return self._buffer[-1], self._buffer[-1]
            return None, None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class WorldModelRunner(threading.Thread):
    """Background daemon thread running LCM inference.

    Continuously polls ``wm_queue`` for packets, runs inference,
    and pushes results to ``jitter_buffer``.
    """

    def __init__(
        self,
        model: InfiniteRaceWorldModel,
        wm_queue: Queue,
        jitter_buffer: JitterBuffer,
        steps: int = 2,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.queue = wm_queue
        self.buffer = jitter_buffer
        self.steps = steps
        self.running = True

    def run(self) -> None:
        self.model.eval()
        device = next(self.model.parameters()).device

        logger.info("WorldModelRunner started on %s (LCM steps=%d)", device, self.steps)

        while self.running:
            try:
                packet = self.queue.get(block=True, timeout=0.1)
            except Empty:
                continue

            if packet is None:
                continue

            try:
                amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
                amp_enabled = device.type in ('cuda', 'mps')
                with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    cue1 = normalize(packet['cue1']).to(device)
                    cue2 = normalize(packet['cue2']).to(device)
                    pos_map = (
                        torch.from_numpy(packet['pos_map'])
                        .unsqueeze(0)
                        .to(device)
                    )                                                   # (1, 256, 256, 2)
                    action = (
                        torch.from_numpy(packet['cue3'])
                        .unsqueeze(0)
                        .to(device)
                    )                                                   # (1, 3)

                    keyframe = self.model.infer(
                        cue1, cue2, pos_map, action, steps=self.steps,
                    )
                    keyframe_bgr = denormalize(keyframe)                # (H, W, 3) uint8 BGR

                self.buffer.push({
                    'frame': keyframe_bgr,
                    'frame_idx': packet['frame_idx'],
                })

            except Exception:
                logger.exception("WorldModelRunner inference error")

        logger.info("WorldModelRunner stopped")
