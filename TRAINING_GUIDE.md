# InfiniteRace World Model — Training Guide

Complete walkthrough: M2 MacBook Air → Vast.ai RTX 4090 → `lcm_final.pt`

---

## Cost Table

| Stage | Est. Time | Cost at $0.21/hr |
|-------|-----------|-----------------|
| Dataset generation (20k samples) | ~45 min | ~$0.16 |
| Level 3 fine-tuning (50k steps, batch 16) | ~5 hr | ~$1.05 |
| Level 4 distillation (20k steps, batch 8) | ~2 hr | ~$0.42 |
| **Total** | **~7.75 hr** | **~$1.63** |

---

## 1. Local Preparation (M2 MacBook Air)

### 1.1 Prepare panorama data (`gsv_data/`)

Each panorama node needs:
- An equirectangular JPEG (any resolution; 4096×2048 recommended)
- A `coordinates.json` file listing all nodes

`coordinates.json` format (array of objects):
```json
[
  {
    "id":            "node_000",
    "lat":           37.5665,
    "lon":           126.9780,
    "image":         "node_000.jpg",
    "compass_angle": 90.0
  },
  ...
]
```

`compass_angle` is the compass bearing (0=North) that maps to the **centre** of
the panorama image.

Acquire panoramas from Mapillary, KartaView, or Google Street View Static API.
Store all files flat in `gsv_data/`.

### 1.2 Verify CueEngine produces valid cues

Run the demo (requires Pygame and the full `demo/` package):
```bash
pip install pygame
python demo/main.py --data_dir gsv_data/
```

Navigate with WASD + mouse drag. Confirm in the panel display:
- **Panel 0** (Anchor crop) shows the street-view scene correctly
- **Panel 1** (Warped frame) shows a laterally-shifted version of Panel 0
- **Panel 3** (Residual) is sparse — bright only at occlusion edges
- **Panel 4** (Action vector) updates when you move

If Panel 0 is blank, check that `coordinates.json` exists and paths are correct.

### 1.3 Dry-run dataset generation (50 samples)

```bash
python generate_dataset.py \
    --panorama_dir gsv_data/ \
    --output_dir   training_data_test/ \
    --n_samples    50 \
    --seed         42
```

Expected output:
```
Loading panoramas from 'gsv_data/' ...
  Loaded N panorama nodes.
  [    50/50]  X.X samples/sec  ETA 0.0 min  yield YY.Y%

Done: 50 samples saved to 'training_data_test/' in 0.X min ...
```

### 1.4 Verify `.npz` format

```python
import numpy as np

d = np.load("training_data_test/sample_00000001.npz")
print(dict(d))
# Expected keys: warped_frame, anchor_crop, anchor_pos_map, action_vector_norm, target_frame

assert d["warped_frame"].shape == (256, 256, 3)
assert d["warped_frame"].dtype == "uint8"
assert d["anchor_pos_map"].shape == (256, 256, 2)
assert d["anchor_pos_map"].dtype == "float32"
assert d["action_vector_norm"].shape == (3,)
print("Format OK")
```

### 1.5 Run shape tests

```bash
python -m pytest tests/ -v
```

All 22 existing tests plus the new `test_generate_dataset.py` tests must pass.
Expected: **30+ passed, 1 skipped** (GPU speed test skipped on CPU-only Mac).

### 1.6 Package for upload

```bash
tar -czf infiniterace_model.tar.gz \
    world_model/ demo/ tests/ \
    generate_dataset.py setup_cloud.sh run_training.sh \
    optimize_tensorrt.py requirements.txt \
    gsv_data/
```

---

## 2. Cloud Instance Setup (Vast.ai RTX 4090)

### 2.1 Create instance

1. Go to [vast.ai](https://vast.ai) → **Search** tab
2. Filter:
   - GPU: RTX 4090 (24 GB VRAM)
   - CUDA: 12.x
   - Template: **PyTorch** (Ubuntu 22.04)
   - Storage: ≥ 50 GB
3. Select cheapest instance (~$0.21/hr) → **Rent**
4. Wait for status to change to **Running**

### 2.2 SSH and upload

```bash
# Find the SSH command in the Vast.ai UI (Instances → Connect)
ssh root@<INSTANCE_IP> -p <PORT>

# Upload the tarball (in a separate terminal)
scp -P <PORT> infiniterace_model.tar.gz root@<INSTANCE_IP>:/workspace/
```

On the instance:
```bash
cd /workspace
tar -xzf infiniterace_model.tar.gz
cd InfiniteRace-model
```

### 2.3 Run setup

```bash
bash setup_cloud.sh
```

This installs all dependencies, pre-downloads the VAE weights, and runs the test
suite. Expect: `22 passed, 1 skipped` (GPU speed test will now pass too).
Takes ~8 minutes.

---

## 3. Dataset Generation on Cloud

### 3.1 Start a tmux session (critical — SSH disconnects kill bare processes)

```bash
tmux new -s dataset
```

### 3.2 Generate 20,000 samples

```bash
python generate_dataset.py \
    --panorama_dir gsv_data/ \
    --output_dir   training_data/ \
    --n_samples    20000 \
    --seed         42
```

Expected progress output every 500 samples:
```
  [   500/20000]  12.3 samples/sec  ETA 26.2 min  yield 82.1%
  [  1000/20000]  12.5 samples/sec  ETA 25.1 min  yield 81.9%
```

If disconnected, reconnect and resume:
```bash
tmux attach -t dataset
# Or restart with --resume:
python generate_dataset.py --panorama_dir gsv_data/ --output_dir training_data/ \
    --n_samples 20000 --resume
```

### 3.3 Verify output

```bash
ls training_data/ | wc -l   # Should be ~20000
python -c "
import numpy as np
d = np.load('training_data/sample_00000001.npz')
print('Keys:', list(d.keys()))
print('warped_frame:', d['warped_frame'].shape, d['warped_frame'].dtype)
print('anchor_pos_map:', d['anchor_pos_map'].shape)
print('action_vector_norm:', d['action_vector_norm'])
"
```

---

## 4. Running Training

### 4.1 Start a tmux training session

```bash
tmux new -s train
```

**Never run training outside tmux** — a dropped SSH connection will kill the
process mid-run and lose all progress.

### 4.2 Run the full pipeline

```bash
bash run_training.sh
```

The pipeline runs automatically through all stages. Monitor with:

```bash
# In a separate pane (Ctrl-B, %) 
nvidia-smi                        # GPU utilisation — should be >90%
tail -f logs/level3.log           # Training loss
```

### 4.3 Healthy vs unhealthy loss curves

**Stage 1 (overfit sanity, 500 steps, 10 samples):**
- Healthy: loss drops sharply from ~1.0 → <0.3 within 200 steps
- Unhealthy: loss stays flat or increases → check data format

**Stage 3 (Level 3, 50k steps):**
- Healthy: loss starts ~0.8–1.2, decreases steadily to ~0.3–0.5 by step 20k,
  then slowly to ~0.2–0.4 by step 50k. Occasional plateaus are normal.
- Unhealthy: NaN loss at any point → reduce learning rate or check for bad samples
- Unhealthy: loss < 0.05 from step 1k → likely data leakage (warp=target)

**Stage 5 (Level 4 distillation, 20k steps):**
- Healthy: consistency loss ~0.1–0.3, reconstruction loss ~0.2–0.4
- Both losses should decrease, with occasional oscillation

### 4.4 Monitoring

```bash
# GPU utilisation and VRAM (should be ~23 GB used on RTX 4090)
watch -n 5 nvidia-smi

# Live training log
tail -f logs/level3.log | grep 'step='

# Disk space (training_data + checkpoints)
df -h /workspace
```

---

## 5. Downloading the Checkpoint

```bash
# From your local machine (replace with actual IP/port from Vast.ai UI)
scp -P <PORT> root@<INSTANCE_IP>:/workspace/InfiniteRace-model/checkpoints/lcm_final.pt .
```

**Terminate the instance immediately after download** — every minute costs money.

On Vast.ai: Instances → your instance → **Destroy**.

---

## 6. Attaching to the Demo

In `demo/main.py`, replace the visualisation call with the world model interface:

```python
# Find this line (roughly):
cue_panel.render(screen, cue_data)

# Replace with:
from world_model.queue_interface import WorldModelInterface
# (initialise once at startup, before the game loop):
wm = WorldModelInterface(model_checkpoint="checkpoints/lcm_final.pt")

# Inside the game loop, replace cue_panel.render with:
wm.send(cue_data)
```

The `WorldModelInterface` runs inference in a background thread so the Pygame
loop is not blocked.

---

## 7. Troubleshooting

### Output is blurry / low detail

**Cause:** Training loss converged too early or perceptual loss weight is too low.

**Fix:** Extend Level 3 training by 20k steps:
```bash
STEPS_L3=70000 bash run_training.sh
```

### Output ignores anchor crop (looks like pure warp)

**Cause:** Cross-attention weights collapsed — anchor latent has near-zero
contribution to queries.

**Fix:** Check that `anchor_pos_map` values are in the range `(-π, π)` and that
the `SphericalSinusoidal` encoding is producing non-trivial embeddings:
```python
from world_model.pos_encoding import SphericalSinusoidal
import torch
enc = SphericalSinusoidal(512)
pm = torch.randn(1, 32, 32, 2)
out = enc(pm)
print(out.std())  # Should be ~0.5–1.5, not near 0
```

### Temporal flickering in demo output

**Cause:** Model is not temporally consistent — each keyframe is predicted
independently with high variance.

**Fix:** In distillation (Stage 5), increase the number of skip steps:
```bash
# Edit world_model/config.py: skip_steps = 1000 (from 500)
# Re-run distillation only:
python3 -m world_model.distill \
    --checkpoint checkpoints/level3_final.pt \
    --data_dir training_data/ \
    --steps 20000 \
    --skip_steps 1000
```

### Action vector has no effect

**Cause:** ActionMLP output is near zero (zero-init is correct at step 0, but
should grow during training). The model has not learned to use action conditioning.

**Fix:** Verify the action vectors in training data span the full range. Check:
```python
import numpy as np, glob
actions = [np.load(f)["action_vector_norm"] for f in glob.glob("training_data/*.npz")[:100]]
actions = np.stack(actions)
print("action mean:", actions.mean(axis=0))   # Should not be all-zero
print("action std:", actions.std(axis=0))     # Should be ~0.3–0.7
```
If std is < 0.1, the generated trajectories are not diverse enough. Re-generate
with a higher `TURN_RATE` or random speed variations in `generate_dataset.py`.

### Loss diverges (NaN)

**Cause:** Learning rate too high, bad batch, or corrupt sample.

**Fix:**
1. Reduce `--lr` to `5e-5` and restart from last good checkpoint
2. Scan for corrupt samples:
   ```bash
   python3 -c "
   import numpy as np, glob, sys
   for f in glob.glob('training_data/*.npz'):
       try:
           d = np.load(f)
           for k in d.files:
               if np.isnan(d[k]).any():
                   print('NaN in', f, k); sys.exit(1)
       except Exception as e:
           print('Bad file:', f, e)
   print('All OK')
   "
   ```
3. If a specific file is corrupt, delete it and re-run with `--resume`
