"""
config.py — Centralized Hyperparameters & Memory Management
============================================================
All configurable variables live here so you never have to hunt through
multiple files to tweak paths, learning rates, or memory-critical sizes.

VRAM Strategy (4 GB limit):
    - BATCH_SIZE  = 2       → keeps peak memory well under 4 GB
    - IMAGE_SIZE  = 128     → aggressive spatial crop for the backward pass
    - Encoder     = resnet34 → lightweight yet effective feature extractor
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 1. Paths
# ──────────────────────────────────────────────
# Root of the MMFlood dataset (TorchGeo will look for tiles here)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

# Ensure the data directory exists for local testing
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. Dataset Parameters
# ──────────────────────────────────────────────
INCLUDE_DEM = True          # Concatenate DEM channel after Sentinel-1
INCLUDE_HYDRO = True        # Concatenate Hydrography as last channel
DOWNLOAD_DATA = True     # Set True to auto-download (requires internet)

# Total input channels: SAR VV (1) + SAR VH (1) + DEM (1) + Hydro (1) = 4
IN_CHANNELS = 2 + int(INCLUDE_DEM) + int(INCLUDE_HYDRO)  # 4

# ──────────────────────────────────────────────
# 3. Memory-Critical Sizes  (VRAM Hack)
# ──────────────────────────────────────────────
IMAGE_SIZE = 128            # Spatial crop in pixels (128 or 256 max)
BATCH_SIZE = 2              # Keep ≤ 2 for 4 GB VRAM
NUM_WORKERS = 0             # 0 = main-process loading (safest on Windows)

# Number of random patches the sampler draws per epoch
SAMPLES_PER_EPOCH = 200     # Small for local dev; scale up on cloud

# ──────────────────────────────────────────────
# 4. Model Architecture
# ──────────────────────────────────────────────
ENCODER_NAME = "resnet34"   # Lightweight encoder
ENCODER_WEIGHTS = "imagenet"  # Transfer-learning init
NUM_CLASSES = 1             # Binary mask: water vs. non-water

# ──────────────────────────────────────────────
# 5. Training Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50             # Ignored during fast_dev_run
OPTIMIZER = "adamw"

# Loss: "dice", "bce", or "combined"
LOSS_TYPE = "combined"

# ──────────────────────────────────────────────
# 6. Lightning Trainer Flags
# ──────────────────────────────────────────────
FAST_DEV_RUN = True         # ← Sanity-check mode (1 batch train + val)
ACCELERATOR = "auto"        # "gpu" if CUDA available, else "cpu"
PRECISION = "16-mixed"      # Mixed-precision to halve VRAM usage
LOG_EVERY_N_STEPS = 1

# Mask special value — MMFlood uses 255 for 'no-data' pixels
IGNORE_INDEX = 255
