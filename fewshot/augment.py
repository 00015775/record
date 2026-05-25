"""Augmentation for (T, V, 3) skeleton sequences during episodic training."""
from __future__ import annotations

import math
import numpy as np

from fewshot import config


def add_noise(j: np.ndarray) -> np.ndarray:
    return j + np.random.randn(*j.shape).astype(np.float32) * config.AUG_NOISE_SIGMA


def rotate_xy(j: np.ndarray) -> np.ndarray:
    angle = math.radians(np.random.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG))
    c, s = math.cos(angle), math.sin(angle)
    out = j.copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def scale_uniform(j: np.ndarray) -> np.ndarray:
    f = np.random.uniform(*config.AUG_SCALE_RANGE)
    return j * f


def translate_xy(j: np.ndarray) -> np.ndarray:
    tx = np.random.uniform(-config.AUG_TRANSLATE_RANGE, config.AUG_TRANSLATE_RANGE)
    ty = np.random.uniform(-config.AUG_TRANSLATE_RANGE, config.AUG_TRANSLATE_RANGE)
    out = j.copy()
    out[..., 0] += tx
    out[..., 1] += ty
    return out


def horizontal_flip(j: np.ndarray) -> np.ndarray:
    """Mirror X and remap joints across the body's vertical axis."""
    out = j.copy()
    out[..., 0] *= -1
    swapped = out.copy()
    for src, dst in config.FLIP_MAP.items():
        swapped[:, dst] = out[:, src]
    return swapped


def time_warp(j: np.ndarray) -> np.ndarray:
    T, V, C = j.shape
    factor = np.random.uniform(*config.AUG_TIME_WARP_RANGE)
    T_warp = max(2, int(round(T * factor)))
    src_idx = np.linspace(0, T - 1, T_warp)
    nearest = np.round(src_idx).astype(int).clip(0, T - 1)
    warped = j[nearest]                                        # (T_warp, V, C)
    # Resample back to T
    out = np.zeros_like(j)
    src = np.linspace(0, T_warp - 1, T_warp)
    dst = np.linspace(0, T_warp - 1, T)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst, src, warped[:, v, c])
    return out.astype(np.float32)


class Augmenter:
    def __call__(self, j: np.ndarray) -> np.ndarray:
        j = add_noise(j)
        if np.random.random() < config.AUG_PROB:
            j = rotate_xy(j)
        if np.random.random() < config.AUG_PROB:
            j = scale_uniform(j)
        if np.random.random() < config.AUG_PROB:
            j = translate_xy(j)
        if np.random.random() < config.AUG_FLIP_PROB:
            j = horizontal_flip(j)
        if np.random.random() < config.AUG_PROB:
            j = time_warp(j)
        return j
