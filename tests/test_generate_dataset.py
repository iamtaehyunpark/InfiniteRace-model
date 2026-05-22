"""
Unit tests for generate_dataset.py.

These tests cover the pure-Python utility functions and data structures
without requiring demo/ or real panorama images.
"""
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from generate_dataset import (
    EARTH_R,
    MAX_STEER,
    MOVE_SPEED,
    DT,
    PlayerState,
    bearing_deg,
    generate_trajectory,
    haversine_m,
    _validate_cue_data,
)


# ---------------------------------------------------------------------------
# Mock CueData and CueEngine — stand-ins for demo.cue_engine
# ---------------------------------------------------------------------------

@dataclass
class MockCueData:
    warped_frame: Optional[np.ndarray]
    anchor_crop: Optional[np.ndarray]
    anchor_pos_map: Optional[np.ndarray]
    action_vector_norm: np.ndarray
    frame_idx: int


class MockCueEngine:
    """Minimal CueEngine replacement that produces synthetic CueData."""

    def __init__(self, nodes=None):
        self._frame_idx = 0
        self._rng = np.random.default_rng(42)

    def update(self, player) -> MockCueData:
        self._frame_idx += 1
        # First frame has no warp (mirrors real CueEngine behaviour)
        warped = (
            None if self._frame_idx == 1
            else self._rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        )
        return MockCueData(
            warped_frame=warped,
            anchor_crop=self._rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_pos_map=self._rng.uniform(-math.pi, math.pi, (256, 256, 2)).astype(np.float32),
            action_vector_norm=self._rng.uniform(-1.0, 1.0, (3,)).astype(np.float32),
            frame_idx=self._frame_idx,
        )


# ---------------------------------------------------------------------------
# PlayerState.step()
# ---------------------------------------------------------------------------

class TestPlayerStateStep:

    def test_heading_east_moves_longitude(self):
        player = PlayerState(lat=37.5665, lon=126.9780, heading_deg=90.0)
        nxt = player.step(turn_deg=0.0)
        # Heading 90° = East → longitude increases, latitude barely changes
        assert nxt.lon > player.lon
        assert abs(nxt.lat - player.lat) < 1e-5

    def test_heading_north_moves_latitude(self):
        player = PlayerState(lat=37.5665, lon=126.9780, heading_deg=0.0)
        nxt = player.step(turn_deg=0.0)
        # Heading 0° = North → latitude increases, longitude barely changes
        assert nxt.lat > player.lat
        assert abs(nxt.lon - player.lon) < 1e-5

    def test_turn_updates_heading(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        nxt = player.step(turn_deg=10.0)
        assert nxt.heading_deg == pytest.approx(10.0, abs=1e-9)
        assert nxt.delta_heading == pytest.approx(10.0, abs=1e-9)

    def test_heading_wraps_at_360(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=350.0)
        nxt = player.step(turn_deg=20.0)
        assert nxt.heading_deg == pytest.approx(10.0, abs=1e-9)

    def test_step_advances_by_correct_distance(self):
        """One step at MOVE_SPEED heading North should move ~(MOVE_SPEED*DT) metres."""
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0, speed_mps=MOVE_SPEED)
        nxt = player.step()
        dist = haversine_m(player.lat, player.lon, nxt.lat, nxt.lon)
        expected = MOVE_SPEED * DT
        assert dist == pytest.approx(expected, rel=1e-3)

    def test_speed_override(self):
        p1 = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        p2 = p1.step(speed=1.5)
        assert p2.speed_mps == pytest.approx(1.5)

    def test_zero_speed_stays_put(self):
        player = PlayerState(lat=37.5, lon=126.9, heading_deg=45.0)
        nxt = player.step(speed=0.0)
        assert nxt.lat == pytest.approx(player.lat, abs=1e-9)
        assert nxt.lon == pytest.approx(player.lon, abs=1e-9)


# ---------------------------------------------------------------------------
# Action vector normalisation
# ---------------------------------------------------------------------------

class TestActionVectorNormalisation:

    def test_steer_within_bounds_for_small_turn(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        for turn in [-MAX_STEER, -5.0, 0.0, 5.0, MAX_STEER]:
            nxt = player.step(turn_deg=turn)
            assert -1.0 <= nxt.steer <= 1.0

    def test_steer_clipped_for_large_turn(self):
        """Turns beyond MAX_STEER clip steer to ±1."""
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        nxt_pos = player.step(turn_deg=MAX_STEER * 3)
        nxt_neg = player.step(turn_deg=-MAX_STEER * 3)
        assert nxt_pos.steer == pytest.approx(1.0)
        assert nxt_neg.steer == pytest.approx(-1.0)

    def test_full_left_steer(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        nxt = player.step(turn_deg=-MAX_STEER)
        assert nxt.steer == pytest.approx(-1.0, abs=1e-6)

    def test_full_right_steer(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        nxt = player.step(turn_deg=MAX_STEER)
        assert nxt.steer == pytest.approx(1.0, abs=1e-6)

    def test_zero_turn_zero_steer(self):
        player = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        nxt = player.step(turn_deg=0.0)
        assert nxt.steer == pytest.approx(0.0, abs=1e-9)
        assert nxt.delta_heading == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Haversine and bearing utilities
# ---------------------------------------------------------------------------

class TestBearingAndHaversine:

    def test_haversine_same_point_is_zero(self):
        assert haversine_m(37.5, 126.9, 37.5, 126.9) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ≈ 341 km
        dist = haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
        assert 335_000 < dist < 345_000

    def test_haversine_symmetric(self):
        d1 = haversine_m(0.0, 0.0, 1.0, 0.0)
        d2 = haversine_m(1.0, 0.0, 0.0, 0.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_haversine_one_degree_latitude_approx(self):
        # 1° latitude ≈ 111,195 m
        dist = haversine_m(0.0, 0.0, 1.0, 0.0)
        assert 111_000 < dist < 111_500

    def test_bearing_north(self):
        # Due North: lat increases, same lon
        b = bearing_deg(0.0, 0.0, 1.0, 0.0)
        assert b == pytest.approx(0.0, abs=0.1)

    def test_bearing_east(self):
        # Due East: same lat, lon increases
        b = bearing_deg(0.0, 0.0, 0.0, 1.0)
        assert b == pytest.approx(90.0, abs=0.1)

    def test_bearing_south(self):
        b = bearing_deg(1.0, 0.0, 0.0, 0.0)
        assert b == pytest.approx(180.0, abs=0.1)

    def test_bearing_west(self):
        b = bearing_deg(0.0, 1.0, 0.0, 0.0)
        assert b == pytest.approx(270.0, abs=0.1)

    def test_bearing_in_range(self):
        for lat1, lon1, lat2, lon2 in [
            (37.0, 126.0, 38.0, 127.0),
            (-10.0, 50.0, 10.0, 30.0),
            (0.0, 179.0, 0.0, -179.0),
        ]:
            b = bearing_deg(lat1, lon1, lat2, lon2)
            assert 0.0 <= b < 360.0


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

class TestTrajectoryLength:

    def test_length_matches_n_steps(self):
        start = PlayerState(lat=37.5665, lon=126.9780, heading_deg=0.0)
        rng = random.Random(42)
        traj = generate_trajectory(start, n_steps=50, rng=rng)
        assert len(traj) == 50

    def test_all_elements_are_player_states(self):
        start = PlayerState(lat=0.0, lon=0.0, heading_deg=90.0)
        rng = random.Random(0)
        traj = generate_trajectory(start, n_steps=30, rng=rng)
        assert all(isinstance(s, PlayerState) for s in traj)

    def test_first_element_is_start(self):
        start = PlayerState(lat=37.5665, lon=126.9780, heading_deg=45.0)
        rng = random.Random(7)
        traj = generate_trajectory(start, n_steps=10, rng=rng)
        assert traj[0] is start

    def test_single_step(self):
        start = PlayerState(lat=1.0, lon=1.0, heading_deg=0.0)
        rng = random.Random(1)
        traj = generate_trajectory(start, n_steps=1, rng=rng)
        assert len(traj) == 1
        assert traj[0] is start

    def test_reproducible_with_same_seed(self):
        start = PlayerState(lat=0.0, lon=0.0, heading_deg=180.0)
        traj1 = generate_trajectory(start, 20, random.Random(99))
        traj2 = generate_trajectory(start, 20, random.Random(99))
        for s1, s2 in zip(traj1, traj2):
            assert s1.lat == s2.lat
            assert s1.lon == s2.lon
            assert s1.heading_deg == s2.heading_deg

    def test_player_moves(self):
        """Player position should change over a 30-step trajectory."""
        start = PlayerState(lat=0.0, lon=0.0, heading_deg=0.0)
        traj = generate_trajectory(start, n_steps=30, rng=random.Random(3))
        assert traj[-1].lat != traj[0].lat or traj[-1].lon != traj[0].lon


# ---------------------------------------------------------------------------
# .npz key/shape contract (uses mock CueEngine — no real panoramas needed)
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "warped_frame_jpg",
    "anchor_crop_jpg",
    "anchor_pos_map",
    "action_vector_norm",
    "target_frame_jpg",
}

EXPECTED_SHAPES = {
    "warped_frame_jpg":       (None,),
    "anchor_crop_jpg":        (None,),
    "anchor_pos_map":     (32, 32, 2),
    "action_vector_norm": (3,),
    "target_frame_jpg":       (None,),
}

EXPECTED_DTYPES = {
    "warped_frame_jpg":       np.uint8,
    "anchor_crop_jpg":        np.uint8,
    "anchor_pos_map":     np.float32,
    "action_vector_norm": np.float32,
    "target_frame_jpg":       np.uint8,
}


class TestNpzKeys:

    def _make_sample(self, tmp_path: Path) -> Path:
        """Run MockCueEngine over a short trajectory and save one valid sample."""
        start = PlayerState(lat=37.5665, lon=126.9780, heading_deg=0.0)
        rng = random.Random(42)
        traj = generate_trajectory(start, n_steps=10, rng=rng)
        engine = MockCueEngine()

        cue_history = [engine.update(p) for p in traj]

        # Find first valid consecutive pair
        for i in range(len(cue_history) - 1):
            cue_t = cue_history[i]
            cue_t1 = cue_history[i + 1]
            if _validate_cue_data(cue_t, f"step {i}") and cue_t1.anchor_crop is not None:
                out = tmp_path / "sample_00000001.npz"
                
                _, warped_jpg = cv2.imencode('.jpg', cue_t.warped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _, anchor_jpg = cv2.imencode('.jpg', cue_t.anchor_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _, target_jpg = cv2.imencode('.jpg', cue_t1.anchor_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                pos_map_32 = cue_t.anchor_pos_map[::8, ::8, :]

                np.savez_compressed(
                    out,
                    warped_frame_jpg=warped_jpg,
                    anchor_crop_jpg=anchor_jpg,
                    anchor_pos_map=pos_map_32,
                    action_vector_norm=cue_t.action_vector_norm,
                    target_frame_jpg=target_jpg,
                )
                return out

        raise RuntimeError("MockCueEngine produced no valid consecutive pair")

    def test_keys_are_exactly_correct(self, tmp_path):
        path = self._make_sample(tmp_path)
        loaded = np.load(path)
        assert set(loaded.keys()) == REQUIRED_KEYS

    def test_no_extra_keys(self, tmp_path):
        path = self._make_sample(tmp_path)
        loaded = np.load(path)
        extra = set(loaded.keys()) - REQUIRED_KEYS
        assert extra == set(), f"Unexpected keys: {extra}"

    def test_shapes_are_correct(self, tmp_path):
        path = self._make_sample(tmp_path)
        loaded = np.load(path)
        for key, expected_shape in EXPECTED_SHAPES.items():
            if expected_shape == (None,):
                assert len(loaded[key].shape) == 1
            else:
                assert loaded[key].shape == expected_shape, (
                    f"Key '{key}': expected shape {expected_shape}, got {loaded[key].shape}"
                )

    def test_dtypes_are_correct(self, tmp_path):
        path = self._make_sample(tmp_path)
        loaded = np.load(path)
        for key, expected_dtype in EXPECTED_DTYPES.items():
            assert loaded[key].dtype == expected_dtype, (
                f"Key '{key}': expected dtype {expected_dtype}, got {loaded[key].dtype}"
            )

    def test_dataset_can_load_sample(self, tmp_path):
        """StreetViewDataset.__getitem__() must succeed on a mock-generated file."""
        path = self._make_sample(tmp_path)
        import sys
        sys.path.insert(0, str(path.parent.parent))
        from world_model.dataset import StreetViewDataset
        import torch
        # Hash-based 90/10 split means one file may land in either split; check both.
        ds_train = StreetViewDataset(str(tmp_path), split="train")
        ds_val   = StreetViewDataset(str(tmp_path), split="val")
        assert len(ds_train) + len(ds_val) == 1, "File must appear in exactly one split"
        ds = ds_train if len(ds_train) > 0 else ds_val
        assert len(ds) > 0
        item = ds[0]
        assert item["cue1"].shape == (3, 256, 256)
        assert item["cue2"].shape == (3, 256, 256)
        assert item["target"].shape == (3, 256, 256)
        assert item["pos_map"].shape == (2, 32, 32)
        assert item["action"].shape == (3,)
        assert item["cue1"].dtype == torch.float32


# ---------------------------------------------------------------------------
# _validate_cue_data
# ---------------------------------------------------------------------------

class TestValidateCueData:

    def test_valid_cue_passes(self):
        rng = np.random.default_rng(0)
        cd = MockCueData(
            warped_frame=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_crop=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_pos_map=rng.uniform(-math.pi, math.pi, (256, 256, 2)).astype(np.float32),
            action_vector_norm=rng.uniform(-1, 1, (3,)).astype(np.float32),
            frame_idx=2,
        )
        assert _validate_cue_data(cd, "test") is True

    def test_none_fails(self):
        assert _validate_cue_data(None, "test") is False

    def test_warped_frame_none_fails(self):
        rng = np.random.default_rng(0)
        cd = MockCueData(
            warped_frame=None,
            anchor_crop=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_pos_map=rng.uniform(-math.pi, math.pi, (256, 256, 2)).astype(np.float32),
            action_vector_norm=rng.uniform(-1, 1, (3,)).astype(np.float32),
            frame_idx=1,
        )
        assert _validate_cue_data(cd, "test") is False

    def test_anchor_crop_none_fails(self):
        rng = np.random.default_rng(0)
        cd = MockCueData(
            warped_frame=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_crop=None,
            anchor_pos_map=rng.uniform(-math.pi, math.pi, (256, 256, 2)).astype(np.float32),
            action_vector_norm=rng.uniform(-1, 1, (3,)).astype(np.float32),
            frame_idx=2,
        )
        assert _validate_cue_data(cd, "test") is False

    def test_wrong_pos_map_shape_fails(self):
        rng = np.random.default_rng(0)
        cd = MockCueData(
            warped_frame=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_crop=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_pos_map=rng.uniform(-math.pi, math.pi, (64, 64, 2)).astype(np.float32),
            action_vector_norm=rng.uniform(-1, 1, (3,)).astype(np.float32),
            frame_idx=2,
        )
        assert _validate_cue_data(cd, "test") is False

    def test_pos_map_none_fails(self):
        rng = np.random.default_rng(0)
        cd = MockCueData(
            warped_frame=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_crop=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            anchor_pos_map=None,
            action_vector_norm=rng.uniform(-1, 1, (3,)).astype(np.float32),
            frame_idx=2,
        )
        assert _validate_cue_data(cd, "test") is False
