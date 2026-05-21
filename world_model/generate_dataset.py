#!/usr/bin/env python3
"""
Generate training dataset by running CueEngine against panorama data.

Usage:
    python -m world_model.generate_dataset \
        --panorama_dir gsv_data/ \
        --output_dir training_data/ \
        --n_samples 50000

Imports CueEngine and loader directly from the demo package.  The script
simulates player trajectories across the node graph and saves .npz files
with all training fields.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate world-model training data from panorama sequences.",
    )
    p.add_argument("--panorama_dir", type=str, required=True,
                   help="Path to GSV panorama data folder (with coordinates.json)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for .npz training samples")
    p.add_argument("--n_samples", type=int, default=50_000,
                   help="Number of training samples to generate")
    p.add_argument("--demo_dir", type=str, default=None,
                   help="Path to demo/ directory (auto-detected if not set)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for trajectory generation")
    return p


class _SimulatedPlayer:
    """Minimal player object matching the interface CueEngine.update() expects."""

    def __init__(self, lat: float, lon: float, heading: float = 0.0):
        self.lat = lat
        self.lon = lon
        self.heading_deg = heading
        self.elevation_deg = 0.0
        self.speed_mps = 0.0
        self.dx_m = 0.0
        self.dy_m = 0.0
        self.delta_heading = 0.0
        self.steer = 0.0
        self._prev_heading = heading

    def step(self, rng: np.random.Generator, nodes, move_speed: float = 3.0):
        """Take one simulation step with random movement."""
        earth_r = 6_371_000.0

        # Random speed and heading change
        self.speed_mps = rng.uniform(0.5, move_speed)
        delta_hdg = rng.uniform(-15.0, 15.0)
        self.heading_deg = (self.heading_deg + delta_hdg) % 360.0
        self.delta_heading = delta_hdg
        self.steer = max(-1.0, min(1.0, delta_hdg / 10.0))
        self.elevation_deg = rng.uniform(-10.0, 10.0)

        # Move forward
        dt = 1.0 / 30.0   # simulate 30 fps
        h_rad = math.radians(self.heading_deg)
        dist = self.speed_mps * dt

        dlat = dist * math.cos(h_rad)
        dlon = dist * math.sin(h_rad)

        self.lat += dlat / earth_r * (180.0 / math.pi)
        cos_lat = math.cos(math.radians(self.lat))
        if cos_lat > 1e-9:
            self.lon += dlon / (earth_r * cos_lat) * (180.0 / math.pi)

        self.dx_m = dlon
        self.dy_m = dlat
        self._prev_heading = self.heading_deg

        # Snap back to nearest node periodically to stay in coverage
        if rng.random() < 0.1 and nodes:
            nearest = min(
                nodes,
                key=lambda n: math.hypot(n.lat - self.lat, n.lon - self.lon),
            )
            self.lat = nearest.lat + rng.uniform(-0.00005, 0.00005)
            self.lon = nearest.lon + rng.uniform(-0.00005, 0.00005)


def main():
    args = _make_parser().parse_args()

    # Locate demo directory
    if args.demo_dir:
        demo_dir = args.demo_dir
    else:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'InfiniteRace-model-input-demo', 'demo'),
            os.path.join(os.path.dirname(__file__), '..', 'demo'),
        ]
        demo_dir = None
        for c in candidates:
            if os.path.isdir(c):
                demo_dir = os.path.abspath(c)
                break
        if demo_dir is None:
            print("Error: Cannot find demo/ directory. Use --demo_dir to specify.")
            sys.exit(1)

    # Add demo dir to path so we can import from it
    sys.path.insert(0, demo_dir)
    from cue_engine import CueEngine
    from loader import load_scene

    # Load scene
    print(f"Loading panoramas from '{args.panorama_dir}' …")
    nodes = load_scene(args.panorama_dir)
    print(f"  {len(nodes)} nodes loaded")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up CueEngine and simulated player
    engine = CueEngine(nodes)
    rng = np.random.default_rng(args.seed)
    player = _SimulatedPlayer(
        lat=nodes[0].lat,
        lon=nodes[0].lon,
        heading=rng.uniform(0, 360),
    )

    print(f"Generating {args.n_samples} samples to '{out_dir}' …")
    saved = 0
    prev_cue = None

    for i in range(args.n_samples + 1):
        # Step the player
        player.step(rng, nodes)

        # Get cue data
        cue = engine.update(player)

        if cue.warped_frame is None:
            prev_cue = cue
            continue

        if prev_cue is not None and prev_cue.warped_frame is not None:
            # Target frame = current anchor crop (what the model should produce
            # given the previous step's cues)
            sample = {
                'warped_frame':       prev_cue.warped_frame,           # (256,256,3)
                'anchor_crop':        prev_cue.anchor_crop,            # (256,256,3)
                'anchor_pos_map':     prev_cue.anchor_pos_map,         # (256,256,2)
                'action_vector_norm': prev_cue.action_vector_norm,     # (3,)
                'target_frame':       cue.anchor_crop,                 # (256,256,3)
            }

            filename = out_dir / f"sample_{saved:07d}.npz"
            np.savez_compressed(str(filename), **sample)
            saved += 1

            if saved % 1000 == 0:
                print(f"  {saved}/{args.n_samples} samples saved")

        prev_cue = cue

        if saved >= args.n_samples:
            break

    print(f"Done — {saved} samples saved to '{out_dir}'")


if __name__ == "__main__":
    main()
