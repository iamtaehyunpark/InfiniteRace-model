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
    libgl1 \
    libglib2.0-0 \
    tmux \
    nvtop
echo "      System packages installed."

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 2 — Virtual Environment Setup & PyTorch 2.3.0 with CUDA 12.1
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Setting up virtual environment and PyTorch 2.3.0 (CUDA 12.1) ..."

# Set cache directories in /workspace to avoid filling root container partition
export PIP_CACHE_DIR="/workspace/.pip_cache"
export HF_HOME="/workspace/.hf_home"

if [ -d "/venv/main" ]; then
    echo "Using existing system virtual environment at /venv/main"
    VENV_PATH="/venv/main"
else
    if [ ! -d "/workspace/venv" ]; then
        echo "Creating virtual environment at /workspace/venv ..."
        python3 -m venv /workspace/venv
    fi
    VENV_PATH="/workspace/venv"
fi

# Activate virtualenv
source "$VENV_PATH/bin/activate"

pip install --upgrade pip

if [ "$VENV_PATH" = "/workspace/venv" ]; then
    pip install --quiet \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "      PyTorch installed."
else
    echo "      Using pre-installed PyTorch in system environment."
fi

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
import os
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
