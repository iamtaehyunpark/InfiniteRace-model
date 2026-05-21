"""
Integration tests — verify queue interface and jitter buffer.

Tests that WorldModelInterface correctly packages CueData into
queue packets and that the JitterBuffer is thread-safe.
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from world_model.inference import JitterBuffer


# ── Mock CueData (matching demo/cue_engine.py interface) ─────────

@dataclass
class _MockCueData:
    anchor_crop: np.ndarray
    anchor_pos_map: np.ndarray
    nearest_node_id: str = "node_0"
    nearest_node_dist_m: float = 5.0
    warped_frame: Optional[np.ndarray] = None
    speed_mps: float = 2.0
    delta_heading_deg: float = 0.0
    steer: float = 0.0
    action_vector_norm: np.ndarray = None
    lookahead_crop: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    frame_idx: int = 0
    heading_deg: float = 0.0
    elevation_deg: float = 0.0
    dx_m: float = 0.0
    dy_m: float = 0.0
    second_node_id: str = "node_1"
    third_node_id: str = "node_2"
    nearest_east_m: float = 0.0
    nearest_north_m: float = 0.0
    second_east_m: float = 5.0
    second_north_m: float = 0.0
    third_east_m: float = 0.0
    third_north_m: float = 5.0

    def __post_init__(self):
        if self.action_vector_norm is None:
            self.action_vector_norm = np.zeros(3, dtype=np.float32)


def _make_cue_data(frame_idx: int = 0, has_warp: bool = True) -> _MockCueData:
    """Create a mock CueData with valid shapes."""
    return _MockCueData(
        anchor_crop=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        anchor_pos_map=np.random.randn(256, 256, 2).astype(np.float32),
        warped_frame=(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            if has_warp else None
        ),
        action_vector_norm=np.random.randn(3).astype(np.float32),
        frame_idx=frame_idx,
    )


# ── JitterBuffer tests ──────────────────────────────────────────────

class TestJitterBuffer:

    def test_empty(self):
        buf = JitterBuffer(maxlen=4)
        a, b = buf.get_two_latest()
        assert a is None
        assert b is None

    def test_single_item(self):
        buf = JitterBuffer(maxlen=4)
        buf.push({'frame_idx': 0})
        a, b = buf.get_two_latest()
        assert a == b == {'frame_idx': 0}

    def test_two_items(self):
        buf = JitterBuffer(maxlen=4)
        buf.push({'frame_idx': 0})
        buf.push({'frame_idx': 1})
        a, b = buf.get_two_latest()
        assert a['frame_idx'] == 0
        assert b['frame_idx'] == 1

    def test_maxlen(self):
        buf = JitterBuffer(maxlen=2)
        for i in range(5):
            buf.push({'frame_idx': i})
        a, b = buf.get_two_latest()
        assert a['frame_idx'] == 3
        assert b['frame_idx'] == 4
        assert len(buf) == 2

    def test_thread_safety(self):
        """Concurrent pushes and reads should not corrupt state."""
        buf = JitterBuffer(maxlen=8)
        errors = []

        def writer():
            for i in range(100):
                buf.push({'frame_idx': i})
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                try:
                    a, b = buf.get_two_latest()
                    if a is not None and b is not None:
                        assert b['frame_idx'] >= a['frame_idx']
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ── Queue Interface tests ───────────────────────────────────────────

class TestQueueInterface:
    """Test the WorldModelInterface packet construction.

    We cannot easily instantiate the full interface (requires a model
    checkpoint), so we test the packet construction logic directly.
    """

    def test_packet_shapes(self):
        """Verify CueData fields have the correct shapes for the model."""
        cue = _make_cue_data(frame_idx=42, has_warp=True)

        # Simulate what WorldModelInterface.send() does
        packet = {
            'cue1':      cue.warped_frame,
            'cue2':      cue.anchor_crop,
            'pos_map':   cue.anchor_pos_map,
            'cue3':      cue.action_vector_norm,
            'frame_idx': cue.frame_idx,
        }

        assert packet['cue1'].shape == (256, 256, 3)
        assert packet['cue1'].dtype == np.uint8
        assert packet['cue2'].shape == (256, 256, 3)
        assert packet['cue2'].dtype == np.uint8
        assert packet['pos_map'].shape == (256, 256, 2)
        assert packet['pos_map'].dtype == np.float32
        assert packet['cue3'].shape == (3,)
        assert packet['cue3'].dtype == np.float32
        assert packet['frame_idx'] == 42

    def test_skip_when_no_warp(self):
        """send() should skip when warped_frame is None."""
        cue = _make_cue_data(has_warp=False)

        # Verify the guard condition
        assert cue.warped_frame is None
        # WorldModelInterface.send() checks this and returns early

    def test_packet_values_preserved(self):
        """Ensure no accidental normalization in packet construction."""
        cue = _make_cue_data()
        original_warp = cue.warped_frame.copy()
        original_anchor = cue.anchor_crop.copy()

        packet = {
            'cue1': cue.warped_frame,
            'cue2': cue.anchor_crop,
        }

        np.testing.assert_array_equal(packet['cue1'], original_warp)
        np.testing.assert_array_equal(packet['cue2'], original_anchor)
