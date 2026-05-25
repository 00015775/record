"""Hyperparameters + paths for the few-shot pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.mod01_config import DATA_ROOT as _DATA_ROOT_STR  # noqa: E402


def auto_device() -> torch.device:
    """Pick the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str) -> torch.device:
    """Resolve a CLI --device string. 'auto' picks the best available."""
    if name in ("auto", "", None):
        return auto_device()
    return torch.device(name)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path(_DATA_ROOT_STR)
FEWSHOT_DIR    = Path(__file__).resolve().parent
WEIGHTS_DIR    = FEWSHOT_DIR / "pretrained_weights"
PROTO_DB_DIR   = FEWSHOT_DIR / "proto_db"
DEFAULT_PROTO  = PROTO_DB_DIR / "uzsl_global.npz"
ENCODER_CKPT   = FEWSHOT_DIR / "encoder_finetuned.pt"

# ── Skeleton layout (joints we use) ──────────────────────────────────────────
# 6 pose (shoulders/elbows/wrists) + 21 left hand + 21 right hand = 48 joints
POSE_ARM_IDX = [11, 12, 13, 14, 15, 16]   # original MediaPipe pose indices
N_POSE = len(POSE_ARM_IDX)
N_HAND = 21
N_JOINTS = N_POSE + N_HAND + N_HAND        # 48
IN_CHANNELS = 3                            # x, y, z

# Indices in our concatenated joint tensor
POSE_SL  = slice(0, N_POSE)                # 0..5
LH_SL    = slice(N_POSE, N_POSE + N_HAND)  # 6..26
RH_SL    = slice(N_POSE + N_HAND, N_JOINTS) # 27..47

# Symmetric joint indices for horizontal-flip augmentation (swap L/R)
FLIP_MAP = {
    0: 1, 1: 0,    # L sh ↔ R sh
    2: 3, 3: 2,    # L el ↔ R el
    4: 5, 5: 4,    # L wr ↔ R wr
}
# Hands swap entirely
for i in range(N_HAND):
    FLIP_MAP[N_POSE + i] = N_POSE + N_HAND + i
    FLIP_MAP[N_POSE + N_HAND + i] = N_POSE + i

# ── Temporal ─────────────────────────────────────────────────────────────────
T_FIXED = 64

# ── Encoder ──────────────────────────────────────────────────────────────────
EMBED_DIM     = 256          # output dim of the ST-GCN encoder
GCN_KERNEL    = 3            # number of adjacency partitions (identity, inward, outward)
TCN_KERNEL    = 9            # temporal conv kernel size
HIDDEN_DIMS   = (64, 128, 256)  # channel progression through ST-GCN blocks

# ── Episodic training ────────────────────────────────────────────────────────
# Each class has ~4 reps total.  With one rep held out for validation that
# leaves 3 train reps, so K_SHOT + Q_QUERY must be ≤ 3.
N_WAY        = 5         # classes per episode
K_SHOT       = 2         # support examples per class (≤ train reps available)
Q_QUERY      = 1         # query examples per class
EPISODES     = 4000      # total training episodes
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPS   = 200
LOG_EVERY    = 50
EVAL_EVERY   = 500
EVAL_EPISODES = 200      # episodes used per evaluation
DROPOUT      = 0.3

# ── Augmentation (applied during episodic training) ──────────────────────────
AUG_NOISE_SIGMA      = 0.01
AUG_ROTATION_DEG     = 12.0
AUG_SCALE_RANGE      = (0.9, 1.1)
AUG_TRANSLATE_RANGE  = 0.05
AUG_FLIP_PROB        = 0.5
AUG_TIME_WARP_RANGE  = (0.8, 1.2)
AUG_PROB             = 0.5      # per-transform application probability

# ── Distance metric ──────────────────────────────────────────────────────────
DISTANCE      = "cosine"   # "cosine" | "euclidean"
TEMPERATURE   = 10.0       # for cosine: scales similarities before softmax
