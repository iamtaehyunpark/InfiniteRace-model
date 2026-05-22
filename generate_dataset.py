"""
Generate .npz training samples by running CueEngine over simulated player trajectories.

Each output file contains one training sample with the keys that StreetViewDataset reads:
  warped_frame       (256, 256, 3) uint8 BGR  — Cue 1
  anchor_crop        (256, 256, 3) uint8 BGR  — Cue 2a
  anchor_pos_map     (256, 256, 2) float32    — Cue 2b
  action_vector_norm (3,) float32             — Cue 3
  target_frame       (256, 256, 3) uint8 BGR  — anchor_crop of the next step

The demo/ package (cue_engine.py, loader.py) is imported inside main() so this
file can be imported in tests with a mock CueEngine, without requiring demo/ to
be present.

Usage:
    python generate_dataset.py \\
        --panorama_dir gsv_data/ \\
        --output_dir   training_data/ \\
        --n_samples    20000 \\
        --seed         42 \\
        --resume
"""
import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Physics constants — must match demo/config.py
# ---------------------------------------------------------------------------
EARTH_R    = 6_371_000.0   # metres
MOVE_SPEED = 3.0            # m/s — speed normalisation denominator
TURN_RATE  = 90.0           # degrees/second — max yaw rate
DT         = 1.0 / 30.0    # seconds per simulation step (30 fps)
MAX_STEER  = 10.0           # degrees/frame at full steer (= action[2] = ±1)


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Minimal telemetry object accepted by CueEngine.update().

    Field names match what CueEngine reads from the _Player in demo/main.py
    (derived from DEMO_SPEC.md §8 and §3.2).
    """
    lat: float
    lon: float
    heading_deg: float
    elevation_deg: float = 0.0
    speed_mps: float = MOVE_SPEED
    delta_heading_deg: float = 0.0
    steer: float = 0.0

    def step(self, turn_deg: float = 0.0, speed: Optional[float] = None) -> "PlayerState":
        """Return the next PlayerState after advancing one simulation tick."""
        spd = speed if speed is not None else self.speed_mps
        delta = float(turn_deg)
        new_heading = (self.heading_deg + delta) % 360.0
        heading_rad = math.radians(new_heading)
        lat_rad = math.radians(self.lat)
        dist = spd * DT
        new_lat = self.lat + (dist * math.cos(heading_rad)) / EARTH_R * (180.0 / math.pi)
        new_lon = self.lon + (dist * math.sin(heading_rad)) / (
            EARTH_R * math.cos(lat_rad) + 1e-9
        ) * (180.0 / math.pi)
        steer = float(np.clip(delta / MAX_STEER, -1.0, 1.0))
        return PlayerState(
            lat=new_lat,
            lon=new_lon,
            heading_deg=new_heading,
            elevation_deg=self.elevation_deg,
            speed_mps=spd,
            delta_heading_deg=delta,
            steer=steer,
        )


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS coordinates."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return EARTH_R * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from (lat1, lon1) to (lat2, lon2) in [0, 360)."""
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(
    start: PlayerState,
    n_steps: int,
    rng: random.Random,
) -> list:
    """Simulate a random walk starting from `start` for `n_steps` ticks.

    The player walks forward at MOVE_SPEED m/s with occasional random turns.
    Returns a list of PlayerState of length `n_steps`.
    """
    states: list = [start]
    current = start
    turn_deg = 0.0
    for _ in range(n_steps - 1):
        r = rng.random()
        if r < 0.05:
            # Initiate a new random turn
            turn_deg = rng.uniform(-TURN_RATE * DT, TURN_RATE * DT)
        elif r < 0.15:
            # Straighten out
            turn_deg = 0.0
        current = current.step(turn_deg=turn_deg)
        states.append(current)
    return states


# ---------------------------------------------------------------------------
# CueData validation
# ---------------------------------------------------------------------------

def _validate_cue_data(cue_data, label: str) -> bool:
    """Return True if cue_data is usable as a training input (not target)."""
    if cue_data is None:
        return False
    if getattr(cue_data, "warped_frame", None) is None:
        return False
    if getattr(cue_data, "anchor_crop", None) is None:
        return False
    pm = getattr(cue_data, "anchor_pos_map", None)
    if pm is None:
        print(f"WARNING [{label}]: anchor_pos_map is None — skipping", file=sys.stderr)
        return False
    if getattr(pm, "shape", None) != (256, 256, 2):
        print(
            f"WARNING [{label}]: anchor_pos_map shape {getattr(pm, 'shape', '?')} "
            f"!= (256,256,2) — skipping",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate .npz training samples from CueEngine street-view trajectories."
    )
    p.add_argument(
        "--panorama_dir", type=str, default="gsv_data/",
        help="Directory containing panorama JPEGs and coordinates.json",
    )
    p.add_argument(
        "--output_dir", type=str, default="training_data/",
        help="Output directory for .npz files",
    )
    p.add_argument(
        "--n_samples", type=int, default=20_000,
        help="Total training samples to generate (including any already saved with --resume)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume", action="store_true",
        help="Skip existing .npz files and continue from where generation left off",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _make_parser().parse_args()

    # Deferred import — keeps this module importable in tests without demo/ present
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from demo.loader import load_scene          # type: ignore[import]
        from demo.cue_engine import CueEngine       # type: ignore[import]
    except ImportError as exc:
        print(
            f"\nERROR: Cannot import from demo/: {exc}\n"
            "\n  Fix: Ensure the demo/ directory is present in the repo root.\n"
            "       It must contain cue_engine.py and loader.py.\n"
            "\n  On a cloud instance: upload the full InfiniteRace-model repo,\n"
            "  including the demo/ package, before running this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_count = len(list(out_dir.glob("*.npz"))) if args.resume else 0
    if args.resume and existing_count > 0:
        print(f"Resume mode: {existing_count} existing samples found, will generate more.")
    to_generate = max(0, args.n_samples - existing_count)
    if to_generate == 0:
        print(f"Already have {existing_count} >= {args.n_samples} samples. Nothing to do.")
        return

    # Load panorama scene
    print(f"Loading panoramas from '{args.panorama_dir}' ...")
    nodes = load_scene(args.panorama_dir)
    if not nodes:
        print(
            f"ERROR: No panorama nodes loaded from '{args.panorama_dir}'.\n"
            "       Check --panorama_dir and ensure coordinates.json is present.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Loaded {len(nodes)} panorama nodes.")

    TRAJ_STEPS = 120  # ~4 seconds per trajectory at 30 fps

    saved = 0
    skipped = 0
    sample_counter = existing_count   # global .npz file counter for naming
    t_start = time.time()

    while saved < to_generate:
        # Pick a random starting node and heading
        node = rng.choice(nodes)
        start = PlayerState(
            lat=node.lat,
            lon=node.lon,
            heading_deg=rng.uniform(0.0, 360.0),
        )

        # Fresh CueEngine per trajectory — resets internal warp reference state
        try:
            engine = CueEngine(nodes)
        except Exception as exc:
            print(f"WARNING: CueEngine init failed: {exc}", file=sys.stderr)
            skipped += TRAJ_STEPS
            continue

        traj = generate_trajectory(start, TRAJ_STEPS, rng)

        # Run the full trajectory sequentially so CueEngine state evolves correctly
        cue_history: list = []
        for player in traj:
            try:
                cd = engine.update(player)
                cue_history.append(cd)
            except Exception as exc:
                cue_history.append(None)

        # Extract consecutive pairs (step t → step t+1) as training samples
        for i in range(len(cue_history) - 1):
            if saved >= to_generate:
                break

            cue_t = cue_history[i]
            cue_t1 = cue_history[i + 1]

            # Validate current-step cues (model inputs)
            if not _validate_cue_data(cue_t, f"step {i}"):
                skipped += 1
                continue

            # Validate next-step anchor_crop (ground-truth target)
            if cue_t1 is None or getattr(cue_t1, "anchor_crop", None) is None:
                skipped += 1
                continue

            out_path = out_dir / f"sample_{sample_counter:08d}.npz"
            try:
                np.savez(
                    out_path,
                    warped_frame=cue_t.warped_frame,
                    anchor_crop=cue_t.anchor_crop,
                    anchor_pos_map=cue_t.anchor_pos_map,
                    action_vector_norm=cue_t.action_vector_norm,
                    target_frame=cue_t1.anchor_crop,
                )
            except Exception as exc:
                print(f"WARNING: Failed to save {out_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            saved += 1
            sample_counter += 1

            if saved % 500 == 0:
                elapsed = time.time() - t_start
                rate = saved / elapsed if elapsed > 0 else 0.0
                remaining = (to_generate - saved) / rate if rate > 0 else float("inf")
                total_att = saved + skipped
                yield_pct = 100.0 * saved / total_att if total_att > 0 else 100.0
                print(
                    f"  [{saved:>6d}/{to_generate}]  "
                    f"{rate:.1f} samples/sec  "
                    f"ETA {remaining / 60:.1f} min  "
                    f"yield {yield_pct:.1f}%"
                )

    elapsed = time.time() - t_start
    total_att = saved + skipped
    yield_pct = 100.0 * saved / total_att if total_att > 0 else 100.0
    print(
        f"\nDone: {saved} samples saved to '{args.output_dir}' "
        f"in {elapsed / 60:.1f} min  "
        f"({skipped} skipped, yield {yield_pct:.1f}% of {total_att} attempts)"
    )


if __name__ == "__main__":
    main()
