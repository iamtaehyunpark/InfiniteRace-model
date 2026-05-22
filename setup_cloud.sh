#!/usr/bin/env bash
# setup_cloud.sh — One-shot cloud GPU instance setup (Ubuntu 22.04, CUDA 12.x)
#
# Run once after SSH-ing into a fresh Vast.ai RTX 4090 instance:
#   bash setup_cloud.sh
#
set -euo pipefail

echo "============================================================"
echo "  InfiniteRace World Model — Cloud Setup"
echo "  $(date)"
echo "============================================================"

# Fix DNS nameservers first
echo "Fixing DNS configuration..."
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
echo "nameserver 1.1.1.1" >> /etc/resolv.conf
echo "DNS configuration fixed."

# ---------------------------------------------------------------------------
# Step 1 — System packages
# ---------------------------------------------------------------------------
echo ""
echo "[1/7] Installing system packages ..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux \
    nvtop
echo "      System packages installed."

# ---------------------------------------------------------------------------
# Step 2 — PyTorch 2.3.0 with CUDA 12.1
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Installing PyTorch 2.3.0 (CUDA 12.1) ..."
pip install --quiet \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121
echo "      PyTorch installed."

# ---------------------------------------------------------------------------
# Step 3 — Python requirements
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] Installing Python requirements ..."
pip install --quiet -r requirements.txt
pip install --quiet lpips==0.1.4
echo "      Python packages installed."

# ---------------------------------------------------------------------------
# Step 4 — Pre-download SD VAE weights
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] Pre-downloading Stable Diffusion VAE weights ..."
echo "      (stabilityai/sd-vae-ft-mse — ~335 MB)"
python3 - <<'PYEOF'
from diffusers import AutoencoderKL
print("  Downloading sd-vae-ft-mse ...")
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("  VAE weights cached.")
PYEOF
echo "      VAE weights ready."

# ---------------------------------------------------------------------------
# Step 5 — Create working directories
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] Creating directories ..."
mkdir -p training_data/ checkpoints/ logs/ gsv_data/
echo "      Directories: training_data/ checkpoints/ logs/ gsv_data/"

# ---------------------------------------------------------------------------
# Step 6 — Run shape tests
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Running shape tests ..."
python3 -m pytest tests/ -q --tb=short
echo "      Tests passed."

# ---------------------------------------------------------------------------
# Step 7 — Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Upload panorama data:"
echo "       scp -r gsv_data/ root@<INSTANCE_IP>:/workspace/InfiniteRace-model/gsv_data/"
echo ""
echo "  2. Generate training dataset (in a tmux session):"
echo "       tmux new -s dataset"
echo "       python generate_dataset.py --panorama_dir gsv_data/ --output_dir training_data/ --n_samples 20000"
echo ""
echo "  3. Run training pipeline:"
echo "       tmux new -s train"
echo "       bash run_training.sh"
echo ""
echo "  See TRAINING_GUIDE.md for the complete walkthrough."
echo ""
