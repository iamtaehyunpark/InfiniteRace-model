#!/usr/bin/env bash
# run_training.sh — Full training pipeline: Level 3 fine-tune → Level 4 distillation
#
# Stages:
#   0. Dataset check
#   1. Overfit sanity check (10 samples × 500 steps)
#   2. Level 2 pretraining (skipped by default — requires nuScenes/Waymo)
#   3. Level 3 street-view fine-tuning
#   4. POC validation gate (forward-pass check)
#   5. Level 4 LCM consistency distillation
#   6. Inference speed benchmark
#   F. Final summary
#
# Usage:
#   bash run_training.sh
#
# Env overrides (set before running):
#   DATA_DIR    CKPT_DIR    STEPS_L3    STEPS_L4    BATCH_L3    BATCH_L4    SKIP_L2
#
set -euo pipefail

# Set cache directories in /workspace to avoid filling root container partition
export HF_HOME="/workspace/.hf_home"
# Reduce CUDA allocator fragmentation (needed when backpropping through VAE decoder)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Activate virtual environment if it exists
if [ -d "/venv/main" ]; then
    echo "Activating virtual environment at /venv/main ..."
    source /venv/main/bin/activate
elif [ -d "/workspace/venv" ]; then
    echo "Activating virtual environment at /workspace/venv ..."
    source /workspace/venv/bin/activate
fi

# ---------------------------------------------------------------------------
# User-configurable variables
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-training_data/}"
CKPT_DIR="${CKPT_DIR:-checkpoints/}"
STEPS_L3="${STEPS_L3:-50000}"
STEPS_L4="${STEPS_L4:-20000}"
BATCH_L3="${BATCH_L3:-4}"
BATCH_L4="${BATCH_L4:-8}"
SKIP_L2="${SKIP_L2:-1}"   # 1 = skip Level 2 (default); 0 = run Level 2

LOG_DIR="logs/"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "============================================================"
echo "  InfiniteRace World Model — Training Pipeline"
echo "  $(date)"
echo "  DATA_DIR=$DATA_DIR   CKPT_DIR=$CKPT_DIR"
echo "  STEPS_L3=$STEPS_L3  STEPS_L4=$STEPS_L4"
echo "  BATCH_L3=$BATCH_L3  BATCH_L4=$BATCH_L4  SKIP_L2=$SKIP_L2"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0 — Dataset check
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 0: Dataset check ==="
NPZ_COUNT=$(find "$DATA_DIR" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
echo "  Found $NPZ_COUNT .npz files in $DATA_DIR"
if [[ "$NPZ_COUNT" -lt 100 ]]; then
    echo ""
    echo "ERROR: Too few training samples ($NPZ_COUNT < 100)."
    echo ""
    echo "  Generate training data first:"
    echo "    python generate_dataset.py \\"
    echo "        --panorama_dir gsv_data/ \\"
    echo "        --output_dir $DATA_DIR \\"
    echo "        --n_samples 20000"
    echo ""
    exit 1
fi
echo "  Dataset OK: $NPZ_COUNT samples."

# ---------------------------------------------------------------------------
# Stage 1 — Overfit sanity check (10 samples, 500 steps)
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 1: Overfit sanity check ==="

OVERFIT_DATA="/tmp/overfit_data/"
OVERFIT_CKPT="/tmp/overfit_ckpt/"
OVERFIT_LOG="$LOG_DIR/overfit.log"
rm -rf "$OVERFIT_DATA" "$OVERFIT_CKPT"
mkdir -p "$OVERFIT_DATA" "$OVERFIT_CKPT"

# Copy 10 random .npz files (shuf -n avoids SIGPIPE under set -o pipefail)
find "$DATA_DIR" -name "*.npz" | shuf -n 10 | xargs -I{} cp {} "$OVERFIT_DATA"
echo "  Copied 10 samples to $OVERFIT_DATA"

echo "  Training 500 steps on 10 samples ..."
python3 -m world_model.train \
    --data_dir "$OVERFIT_DATA" \
    --checkpoint_dir "$OVERFIT_CKPT" \
    --steps 500 \
    --batch_size 4 \
    --log_every 50 \
    --save_every 500 \
    --no_wandb \
    2>&1 | tee "$OVERFIT_LOG"

# Parse first and last loss from the log
FIRST_LOSS=$(grep -oP 'loss=\K[0-9]+\.[0-9]+' "$OVERFIT_LOG" | head -1)
LAST_LOSS=$(grep -oP 'loss=\K[0-9]+\.[0-9]+' "$OVERFIT_LOG" | tail -1)

echo "  First logged loss: $FIRST_LOSS"
echo "  Last  logged loss: $LAST_LOSS"

# Compare using python (bash float comparison is unreliable)
LOSS_OK=$(python3 -c "
first = float('$FIRST_LOSS')
last  = float('$LAST_LOSS')
print('yes' if last < first else 'no')
")

if [[ "$LOSS_OK" != "yes" ]]; then
    echo ""
    echo "ERROR: Loss did not decrease during overfit check ($FIRST_LOSS → $LAST_LOSS)."
    echo ""
    echo "  Diagnostics:"
    echo "    - Check $OVERFIT_LOG for NaN or very large losses"
    echo "    - Verify data format: python -c \"import numpy as np; d=np.load(ls $OVERFIT_DATA*.npz | head -1); print(d.files)\""
    echo "    - Check GPU memory: nvidia-smi"
    echo ""
    exit 1
fi
echo "  Sanity check PASSED: loss decreased $FIRST_LOSS → $LAST_LOSS"

# ---------------------------------------------------------------------------
# Stage 2 — Level 2 pretraining (skipped by default)
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2: Level 2 pretraining ==="
if [[ "$SKIP_L2" == "1" ]]; then
    echo "  SKIP_L2=1 — skipping Level 2."
    echo "  (Level 2 requires nuScenes/Waymo driving video data.)"
    echo "  To enable: SKIP_L2=0 bash run_training.sh"
    echo "  When enabled, runs world_model.train for ~100k steps on large-scale data."
else
    echo "  Running Level 2 pretraining ..."
    # Level 2 requires nuScenes or Waymo data in DATA_DIR (different format).
    # Replace DATA_DIR below with your Level 2 dataset path.
    python3 -m world_model.train \
        --data_dir "$DATA_DIR" \
        --checkpoint_dir "$CKPT_DIR" \
        --steps 100000 \
        --batch_size "$BATCH_L3" \
        --log_every 100 \
        --save_every 5000 \
        --no_wandb \
        2>&1 | tee "$LOG_DIR/level2.log"
    echo "  Level 2 complete."
fi

# ---------------------------------------------------------------------------
# Stage 3 — Level 3 street-view fine-tuning
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3: Level 3 fine-tuning ($STEPS_L3 steps) ==="
python3 -m world_model.train \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --steps "$STEPS_L3" \
    --batch_size "$BATCH_L3" \
    --log_every 100 \
    --save_every 2000 \
    --val_every 1000 \
    --early_stop_patience 3000 \
    --no_wandb \
    2>&1 | tee "$LOG_DIR/level3.log"

# Prefer best.pt (saved by early stopping) over the latest periodic checkpoint
if [[ -f "$CKPT_DIR/best.pt" ]]; then
    cp "$CKPT_DIR/best.pt" "$CKPT_DIR/level3_final.pt"
    echo "  Level 3 complete (early stopped). Using best val LPIPS checkpoint."
else
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.pt 2>/dev/null | head -1)
    if [[ -z "$LATEST_CKPT" ]]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR after Level 3 training."
        exit 1
    fi
    cp "$LATEST_CKPT" "$CKPT_DIR/level3_final.pt"
    echo "  Level 3 complete (ran to $STEPS_L3 steps, no early stop)."
fi
echo "  Checkpoint: $CKPT_DIR/level3_final.pt"

# ---------------------------------------------------------------------------
# Stage 4 — POC validation gate
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 4: POC validation gate ==="
python3 - <<'PYEOF'
import torch
from world_model.model import InfiniteRaceWorldModel

print("  Loading level3_final.pt ...")
model = InfiniteRaceWorldModel.load("checkpoints/level3_final.pt", device="cuda")
model.eval()

cue1    = torch.randn(1, 3, 256, 256, device="cuda")
cue2    = torch.randn(1, 3, 256, 256, device="cuda")
pos_map = torch.randn(1, 256, 256, 2, device="cuda")
action  = torch.randn(1, 3, device="cuda").clamp(-1, 1)

with torch.inference_mode():
    out = model(cue1, cue2, pos_map, action)

assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"
assert not torch.isnan(out).any(), "Output contains NaN values"
print("  Forward pass OK — output shape:", tuple(out.shape))
PYEOF

echo "  POC validation PASSED."

# ---------------------------------------------------------------------------
# Stage 5 — Level 4 LCM consistency distillation
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 5: Level 4 distillation ($STEPS_L4 steps) ==="
python3 -m world_model.distill \
    --checkpoint "$CKPT_DIR/level3_final.pt" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --steps "$STEPS_L4" \
    --batch_size "$BATCH_L4" \
    2>&1 | tee "$LOG_DIR/level4.log"
echo "  Level 4 distillation complete. Checkpoint: $CKPT_DIR/lcm_final.pt"

# ---------------------------------------------------------------------------
# Stage 6 — Inference speed benchmark
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 6: Inference speed benchmark ==="
python3 - <<'PYEOF'
import time
import torch
from world_model.model import InfiniteRaceWorldModel

TARGET_MS = 50.0

model = InfiniteRaceWorldModel.load("checkpoints/lcm_final.pt", device="cuda")
model = model.half().eval()

cue1    = torch.randn(1, 3, 256, 256, device="cuda", dtype=torch.float16)
cue2    = torch.randn(1, 3, 256, 256, device="cuda", dtype=torch.float16)
pos_map = torch.randn(1, 256, 256, 2, device="cuda", dtype=torch.float16)
action  = torch.randn(1, 3, device="cuda", dtype=torch.float16)

results = {}
for steps in [2, 1]:
    # Warmup
    for _ in range(5):
        model.infer(cue1, cue2, pos_map, action, steps=steps)
    torch.cuda.synchronize()
    # Benchmark
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        model.infer(cue1, cue2, pos_map, action, steps=steps)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000.0
    fps = 1000.0 / ms
    results[steps] = ms
    status = "OK" if ms <= TARGET_MS else "SLOW"
    print(f"  {steps}-step inference: {ms:.1f} ms  ({fps:.1f} fps)  [{status}]")

any_ok = any(ms <= TARGET_MS for ms in results.values())
if not any_ok:
    print()
    print(f"  Neither 1-step nor 2-step inference hit the {TARGET_MS} ms target.")
    print("  Recommendation: run TensorRT/torch.compile optimisation:")
    print("    python optimize_tensorrt.py \\")
    print("        --checkpoint checkpoints/lcm_final.pt \\")
    print("        --output    checkpoints/model_trt.pt \\")
    print("        --steps     2")
else:
    print(f"  Target ({TARGET_MS} ms) achieved.")
PYEOF

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Training complete!"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  Checkpoints in $CKPT_DIR:"
ls -lh "$CKPT_DIR"/*.pt 2>/dev/null || echo "  (no .pt files found)"
echo ""
# Print scp command using current hostname/IP if available
INSTANCE_IP="${SSH_CLIENT%% *}"
if [[ -n "${INSTANCE_IP:-}" ]]; then
    echo "  Download checkpoint:"
    echo "    scp root@$INSTANCE_IP:/workspace/InfiniteRace-model/$CKPT_DIR/lcm_final.pt ."
else
    echo "  Download checkpoint (replace <INSTANCE_IP>):"
    echo "    scp root@<INSTANCE_IP>:/workspace/InfiniteRace-model/$CKPT_DIR/lcm_final.pt ."
fi
echo ""
echo "  IMPORTANT: Terminate your Vast.ai instance now to stop billing."
echo ""
