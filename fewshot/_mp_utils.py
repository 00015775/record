"""Small MediaPipe helpers shared by live / recognize scripts."""
from __future__ import annotations

import numpy as np


def landmark_array(lm, n: int, with_vis: bool = False) -> np.ndarray:
    """Convert a MediaPipe landmark list to a numpy array. Zeros if missing."""
    out_dim = 4 if with_vis else 3
    arr = np.zeros((n, out_dim), dtype=np.float32)
    if lm is None:
        return arr
    for i, p in enumerate(lm.landmark[:n]):
        if with_vis:
            arr[i] = (p.x, p.y, p.z, getattr(p, "visibility", 0.0))
        else:
            arr[i] = (p.x, p.y, p.z)
    return arr
