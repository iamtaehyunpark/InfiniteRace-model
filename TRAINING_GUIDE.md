# InfiniteRace World Model — Training Guide

Complete walkthrough: M2 MacBook Air → Vast.ai RTX 4090 → `lcm_final.pt`

**Current repo state:** `demo/` and `gsv_data/` (30 nodes) are already committed.
Everything below assumes you are working from `iamtaehyunpark/InfiniteRace-model`.

---

## Cost Table

Prices vary by instance. Filter Vast.ai by GPU model and sort by price.

| Stage | Est. Time (RTX 4090) | At $0.21/hr | At $0.40/hr |
|-------|----------------------|-------------|-------------|
| Dataset generation (20k samples) | ~45 min | ~$0.16 | ~$0.30 |
| Level 3 fine-tuning (early stop ~20k steps) | ~2.5 hr | ~$0.53 | ~$1.00 |
| Level 4 distillation (20k steps) | ~2 hr | ~$0.42 | ~$0.80 |
| **Total (with early stopping)** | **~5.25 hr** | **~$1.11** | **~$2.10** |
| Level 3 without early stop (50k steps) | ~6 hr | ~$1.26 | ~$2.40 |

Early stopping is enabled by default in `run_training.sh` (`--early_stop_patience 3000`).
Training stops automatically when val LPIPS stops improving, typically at 15–25k steps.

---

## 1. Local Verification (Before Renting)

### 1.1 Check what's already in the repo

```bash
ls gsv_data/          # 30 panorama nodes (node_000.jpg … node_029.jpg)
ls demo/              # cue_engine.py, loader.py, config.py, etc.
```

30 nodes is enough for a working model. For better generalisation, add more
panoramas to `gsv_data/` before generating the full 20k dataset (see §1.5).

### 1.2 Run the test suite

```bash
python -m pytest tests/ -q
```

Expected: **60 passed, 1 skipped** (GPU speed test skipped on CPU-only Mac).
If any test fails, do not proceed to cloud.

### 1.3 Dry-run dataset generation (20 samples)

```bash
python generate_dataset.py \
    --panorama_dir gsv_data/ \
    --output_dir   /tmp/training_data_test/ \
    --n_samples    20 \
    --seed         42
```

Expected:
```
Loading panoramas from 'gsv_data/' ...
  Loaded 30 panorama nodes.

Done: 20 samples saved to '/tmp/training_data_test/' in 0.1 min  (0 skipped, yield 100.0% of 20 attempts)
```

### 1.4 Verify `.npz` format

```python
import numpy as np, os
files = sorted(os.listdir('/tmp/training_data_test/'))
d = np.load(f'/tmp/training_data_test/{files[0]}')
print(list(d.keys()))
# ['warped_frame', 'anchor_crop', 'anchor_pos_map', 'action_vector_norm', 'target_frame']
assert d['warped_frame'].shape    == (256, 256, 3) and d['warped_frame'].dtype    == 'uint8'
assert d['anchor_pos_map'].shape  == (256, 256, 2) and d['anchor_pos_map'].dtype  == 'float32'
assert d['action_vector_norm'].shape == (3,)        and d['action_vector_norm'].dtype == 'float32'
assert d['target_frame'].shape    == (256, 256, 3)
print("Format OK")
```

### 1.5 (Optional) Expand panorama coverage

30 nodes covering a small area will produce a model that works but may overfit
to that specific location. For a more generalisable model, add more nodes before
uploading. Each node needs a JPEG + an entry in `coordinates.json`:

```json
{
  "id":            "node_030",
  "lat":           37.5670,
  "lon":           126.9785,
  "image":         "node_030.jpg",
  "compass_angle": 0.0
}
```

Acquire panoramas from Mapillary, KartaView, or the Google Street View Static API.
`compass_angle` is the bearing (0=North) that maps to the centre of the image.

After adding nodes, re-run the §1.3 dry run to confirm they load correctly.

### 1.6 Verify demo runs (optional but recommended)

```bash
pip install pygame
python demo/main.py
```

Navigate with WASD + mouse drag. Confirm:
- **Panel 0** (Anchor crop) shows the street-view scene
- **Panel 1** (Warped frame) looks like Panel 0 with a lateral shift
- **Panel 3** (Residual) is sparse — bright only at occlusion edges
- **Panel 4** (Action vector) changes when you move

---

## 2. Cloud Instance Setup (Vast.ai RTX 4090)

### 2.1 Create instance

1. Go to [vast.ai](https://vast.ai) → **Search** tab
2. Filter:
   - GPU: **RTX 4090** (24 GB VRAM)
   - CUDA: **12.x** (12.1 or 12.2 both work)
   - Template: **PyTorch** (Ubuntu 22.04)
   - Storage: **≥ 50 GB** (NVMe preferred)
3. Sort by price, select → **Rent**
4. Wait for **Running** status

Verified working: instance type #9021757 (Iceland, $0.404/hr, CUDA 12.2, 135 GB NVMe).

### 2.2 Connect and clone

```bash
# SSH in (exact command shown in Vast.ai UI → Instances → Connect)
ssh root@<INSTANCE_IP> -p <PORT>

# On the instance — clone directly from GitHub (no upload needed)
cd /workspace
git clone https://github.com/iamtaehyunpark/InfiniteRace-model.git
cd InfiniteRace-model
```

### 2.3 Run setup

```bash
bash setup_cloud.sh
```

This installs all dependencies, pre-downloads the SD VAE weights (~335 MB),
and runs the test suite. Expect **60 passed, 1 skipped** (GPU speed test now runs).
Takes ~8 minutes.

---

## 3. Dataset Generation on Cloud

### 3.1 Start a tmux session first

```bash
tmux new -s dataset
```

**Always use tmux.** An SSH disconnect kills any bare process instantly.

### 3.2 Generate 20,000 samples

```bash
python generate_dataset.py \
    --panorama_dir gsv_data/ \
    --output_dir   training_data/ \
    --n_samples    20000 \
    --seed         42
```

Expected progress (every 500 samples):
```
  [   500/20000]  12.3 samples/sec  ETA 26.2 min  yield 82.1%
  [  1000/20000]  12.5 samples/sec  ETA 25.1 min  yield 81.9%
```

If disconnected, resume without re-generating what's already done:
```bash
tmux attach -t dataset   # reconnect to existing session
# or restart the script with --resume:
python generate_dataset.py --panorama_dir gsv_data/ --output_dir training_data/ \
    --n_samples 20000 --resume
```

### 3.3 Verify output

```bash
ls training_data/ | wc -l   # should be ~20000
python -c "
import numpy as np
d = np.load('training_data/sample_00000001.npz')
print('Keys:', list(d.keys()))
print('action_vector_norm:', d['action_vector_norm'])  # should not be all-zero
"
```

---

## 4. Running Training

### 4.1 Start a tmux training session

```bash
tmux new -s train
```

### 4.2 Run the full pipeline

```bash
bash run_training.sh
```

The pipeline runs automatically through all stages with no manual intervention:

| Stage | What happens |
|-------|-------------|
| 0 | Counts `.npz` files — exits if < 100 |
| 1 | Overfits 10 samples × 500 steps — exits if loss doesn't decrease |
| 2 | Level 2 pretraining (skipped by default, needs nuScenes/Waymo) |
| 3 | Level 3 fine-tuning with **early stopping** (val LPIPS every 1k steps) |
| 4 | POC gate — forward pass + shape assert on `level3_final.pt` |
| 5 | Level 4 LCM distillation |
| 6 | Inference speed benchmark (prints ms/fps, flags if > 50ms) |

### 4.3 Monitoring (in a second tmux pane: `Ctrl-B %`)

```bash
# GPU utilisation — should be ≥ 90% during training
watch -n 5 nvidia-smi

# Live loss
tail -f logs/level3.log | grep -E 'step=|val_lpips|Early stop'

# Disk space
df -h /workspace
```

### 4.4 What to expect — Stage 3 (Level 3)

Training loss and val LPIPS are logged every 1,000 steps.

| Steps | Training loss | Val LPIPS | Status |
|-------|--------------|-----------|--------|
| 1k | ~1.0–1.2 | ~0.45–0.55 | Warming up |
| 5k | ~0.6–0.8 | ~0.35–0.45 | Learning structure |
| 15k | ~0.35–0.5 | ~0.25–0.35 | Converging |
| 25k+ | ~0.3–0.45 | plateauing | Early stop likely fires here |

**Early stopping:** when val LPIPS fails to improve for 3 consecutive checks
(3,000 steps), training stops and `checkpoints/best.pt` is copied to
`checkpoints/level3_final.pt`. You do not need to do anything — the script
handles this automatically.

**Unhealthy signs:**
- Training loss stays flat after step 2k → check data format (run §1.4)
- NaN loss at any point → reduce LR: `LR=5e-5 bash run_training.sh`
- Val LPIPS stuck above 0.5 at step 10k → likely too few diverse panoramas (add more nodes)
- Loss < 0.05 from step 1k → data leakage (warp frame equals target frame — check trajectory generation)

### 4.5 What to expect — Stage 5 (Level 4 distillation)

```
step=0    loss=0.28  consist=0.18  recon=0.20
step=100  loss=0.24  consist=0.14  recon=0.20
```

Healthy: both consistency and reconstruction losses decrease together.
Stop: if consistency loss drops below 0.05 while reconstruction loss stays high
(>0.3) — the model has collapsed. Re-run distillation from `level3_final.pt` with
a lower LR: `python3 -m world_model.distill --lr 1e-6 ...`.

---

## 5. Downloading the Checkpoint

```bash
# From your local machine
scp -P <PORT> root@<INSTANCE_IP>:/workspace/InfiniteRace-model/checkpoints/lcm_final.pt .
```

**Destroy the instance immediately after** — Vast.ai: Instances → your instance → **Destroy**.

Also download `level3_final.pt` if you want to re-run distillation later without
re-training Level 3:
```bash
scp -P <PORT> root@<INSTANCE_IP>:/workspace/InfiniteRace-model/checkpoints/level3_final.pt .
```

---

## 6. Attaching to the Demo

In `demo/main.py`, two changes:

**1. Add import at the top:**
```python
from world_model.queue_interface import WorldModelInterface
```

**2. Before the game loop, initialise the interface:**
```python
wm = WorldModelInterface(model_checkpoint="checkpoints/lcm_final.pt")
```

**3. Inside the game loop, replace:**
```python
cue_panel.render(screen, cue_data)        # remove this line
```
with:
```python
wm.send(cue_data)                         # add this line
```

`WorldModelInterface` runs inference in a background daemon thread. The Pygame
loop is not blocked. Keyframes appear in `wm.jitter_buffer` as they complete.

---

## 7. Troubleshooting

### Output is blurry / low detail

**Cause:** Model converged before the perceptual loss ramp-up completed
(LPIPS weight reaches 0.5 only at step 10k).

**Fix:** More training steps. Re-run Level 3 from the best checkpoint:
```bash
python3 -m world_model.train \
    --data_dir training_data/ \
    --checkpoint_dir checkpoints/ \
    --resume checkpoints/level3_final.pt \
    --steps 20000 \
    --lr 3e-5 \
    --val_every 1000 \
    --early_stop_patience 3000 \
    --no_wandb
```

### Output ignores anchor crop (looks like pure warp)

**Cause:** Cross-attention weights collapsed — anchor latent contributes
near zero to the queries.

**Fix:** Check `anchor_pos_map` range and sinusoidal encoding output:
```python
from world_model.pos_encoding import SphericalSinusoidal
import torch
enc = SphericalSinusoidal(512)
out = enc(torch.randn(1, 32, 32, 2))
print(out.std().item())   # should be 0.5–1.5, not < 0.1
```
If near zero, the pos_map values from `generate_dataset.py` may be out of range.
Run §1.4 and check `anchor_pos_map` min/max — should be in `(-π, π)`.

### Temporal flickering in demo output

**Cause:** Each keyframe predicted independently with high variance.

**Fix:** Increase distillation skip steps for stronger consistency pressure:
```bash
python3 -m world_model.distill \
    --checkpoint checkpoints/level3_final.pt \
    --data_dir   training_data/ \
    --steps      20000 \
    --skip_steps 1000
```

### Action vector has no effect (scene looks same regardless of movement)

**Cause:** ActionMLP hasn't learned to use the action conditioning — common if
all training samples have near-zero action vectors (player mostly stationary).

**Diagnose:**
```python
import numpy as np, glob
actions = np.stack([np.load(f)["action_vector_norm"]
                    for f in glob.glob("training_data/*.npz")[:200]])
print("std:", actions.std(axis=0))   # target: [~0.4, ~0.3, ~0.3]
```
If std < 0.1 on any component, the trajectories aren't diverse enough.
The fix is to regenerate with more varied turns — `TURN_RATE` in `generate_dataset.py`
controls this. With 30 panorama nodes covering a small area the player
frequently hits the edge and reverses, which produces low-diversity actions.
Adding more nodes (§1.5) fixes both issues.

### Val LPIPS not improving past 0.45

**Cause:** Too few panorama nodes — the model overfits to the 30-node area
and cannot generalise to unseen viewpoints in the val split.

**Fix:** Add ≥ 50 more panorama nodes covering different streets, re-generate
the dataset, and re-train.

### Loss diverges (NaN)

**Fix:**
1. Reduce LR: re-run with `--lr 5e-5`
2. Scan for corrupt samples:
   ```bash
   python3 -c "
   import numpy as np, glob, sys
   for f in glob.glob('training_data/*.npz'):
       try:
           d = np.load(f)
           for k in d.files:
               if np.isnan(d[k].astype(float)).any():
                   print('NaN:', f, k); sys.exit(1)
       except Exception as e:
           print('Bad file:', f, e)
   print('All OK')
   "
   ```
3. Delete corrupt files and resume: `python generate_dataset.py ... --resume`

### Inference speed > 50ms on RTX 4090

Run TensorRT / torch.compile optimisation:
```bash
python optimize_tensorrt.py \
    --checkpoint checkpoints/lcm_final.pt \
    --output     checkpoints/model_trt.pt \
    --steps      2
```
Then use `model_trt.pt` in the demo attachment step.
