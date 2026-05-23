"""
Queue interface — Phase 1 integration point.

Bridges CueData from the demo CueEngine to the world model runner.
Drop-in replacement for ``cue_panel.render()`` in ``demo/main.py``.

Integration in demo/main.py replaces:
    cue_panel.render(screen, cue_data)
with:
    world_model_interface.send(cue_data)
"""
import logging
from queue import Full, Queue
from typing import Optional

import numpy as np

from world_model.inference import JitterBuffer, WorldModelRunner, denormalize
from world_model.model import InfiniteRaceWorldModel

logger = logging.getLogger(__name__)


class WorldModelInterface:
    """Bridges CueData from demo CueEngine to world model runner.

    Non-blocking: if the world model is running behind, frames are
    dropped (``put_nowait`` + ``Full`` exception caught).

    Args:
        model_checkpoint: path to trained model checkpoint
        display_surface:  optional pygame Surface to blit output onto
        queue_maxsize:    max packets in flight
        jitter_maxlen:    max keyframes in jitter buffer
        steps:            LCM inference steps (default 2)
    """

    def __init__(
        self,
        model_checkpoint: str,
        display_surface=None,
        queue_maxsize: int = 4,
        jitter_maxlen: int = 4,
        steps: int = 2,
    ):
        self.wm_queue = Queue(maxsize=queue_maxsize)
        self.jitter_buffer = JitterBuffer(maxlen=jitter_maxlen)

        # Load model — pick best available device
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model = InfiniteRaceWorldModel.load(model_checkpoint, device=device)
        logger.info("WorldModelInterface using device: %s", device)

        self.runner = WorldModelRunner(
            model=model,
            wm_queue=self.wm_queue,
            jitter_buffer=self.jitter_buffer,
            steps=steps,
        )
        self.runner.start()

        self.display = display_surface

    def send(self, cue_data) -> None:
        """Called once per game tick in place of cue_panel.render().

        Args:
            cue_data: CueData instance from CueEngine.update()
        """
        if cue_data.warped_frame is None:
            return   # skip until first warp is available

        packet = {
            'cue1':      cue_data.warped_frame,           # (256,256,3) uint8 BGR
            'cue2':      cue_data.anchor_crop,             # (256,256,3) uint8 BGR
            'pos_map':   cue_data.anchor_pos_map,          # (256,256,2) float32
            'cue3':      cue_data.action_vector_norm,      # (3,) float32
            'frame_idx': cue_data.frame_idx,
        }

        try:
            self.wm_queue.put_nowait(packet)
        except Full:
            pass   # drop frame — world model is running behind

        # Optionally display latest keyframe on pygame surface
        if self.display is not None:
            _, kf_latest = self.jitter_buffer.get_two_latest()
            if kf_latest is not None:
                self._blit_keyframe(kf_latest['frame'])

    def _blit_keyframe(self, frame_bgr: np.ndarray) -> None:
        """Blit a keyframe onto the pygame display surface."""
        try:
            import pygame
            # BGR → RGB and transpose for pygame
            rgb = frame_bgr[:, :, ::-1]
            arr = np.ascontiguousarray(rgb.transpose(1, 0, 2))
            surf = pygame.surfarray.make_surface(arr)
            self.display.blit(surf, (0, 0))
        except Exception:
            logger.debug("Failed to blit keyframe", exc_info=True)

    def stop(self) -> None:
        """Clean shutdown of the runner thread."""
        self.runner.running = False
        self.runner.join(timeout=2.0)
        logger.info("WorldModelInterface stopped")
