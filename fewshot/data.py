"""Data loading + UzSL → ST-GCN input adapter.

Converts our `keypoints.npz` files (pose 33×4, left_hand 21×3, right_hand 21×3)
into (C=3, T_FIXED, V=48) tensors suitable for the ST-GCN encoder.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fewshot import config


# ── Joint extraction + normalisation ──────────────────────────────────────────

def _extract_joints(npz: dict) -> np.ndarray:
    """Return (T, V=48, 3) joint tensor in MediaPipe coordinate space (raw)."""
    pose = npz["pose"]                     # (T, 33, 4)
    lh   = npz["left_hand"]                # (T, 21, 3)
    rh   = npz["right_hand"]               # (T, 21, 3)
    T = pose.shape[0]

    out = np.zeros((T, config.N_JOINTS, 3), dtype=np.float32)
    out[:, config.POSE_SL] = pose[:, config.POSE_ARM_IDX, :3]
    out[:, config.LH_SL]   = lh
    out[:, config.RH_SL]   = rh
    return out


def _normalise(joints: np.ndarray) -> np.ndarray:
    """Per-frame translate to shoulder midpoint, scale by shoulder distance.

    Frames where shoulders are not detected are left unchanged (zeros stay zero).
    """
    out = joints.copy()
    for t in range(out.shape[0]):
        l_sh = joints[t, 0]    # left shoulder (joint 0)
        r_sh = joints[t, 1]    # right shoulder (joint 1)
        if not (np.any(np.abs(l_sh) > 1e-6) and np.any(np.abs(r_sh) > 1e-6)):
            continue
        origin = (l_sh + r_sh) / 2.0
        scale = float(np.linalg.norm(l_sh - r_sh))
        if scale < 1e-6:
            continue
        out[t] = (out[t] - origin) / scale
    return out


def _resample(seq: np.ndarray, T_out: int = config.T_FIXED) -> np.ndarray:
    """Linearly interpolate (T_in, V, C) → (T_out, V, C)."""
    T_in = seq.shape[0]
    if T_in == T_out:
        return seq
    src = np.linspace(0, T_in - 1, T_in)
    dst = np.linspace(0, T_in - 1, T_out)
    V = seq.shape[1]
    C = seq.shape[2]
    out = np.zeros((T_out, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst, src, seq[:, v, c])
    return out


def preprocess_npz(npz_path: Path) -> np.ndarray:
    """Full pipeline: .npz → (T_FIXED, V, 3) normalised joints."""
    npz = dict(np.load(npz_path))
    joints = _extract_joints(npz)
    joints = _normalise(joints)
    joints = _resample(joints, config.T_FIXED)
    return joints


def preprocess_arrays(pose: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Pipeline for live data (already in memory arrays).

    pose: (T, 33, ≥3), lh: (T, 21, 3), rh: (T, 21, 3).  Returns (T_FIXED, V, 3).
    """
    T = pose.shape[0]
    joints = np.zeros((T, config.N_JOINTS, 3), dtype=np.float32)
    joints[:, config.POSE_SL] = pose[:, config.POSE_ARM_IDX, :3]
    joints[:, config.LH_SL]   = lh[..., :3]
    joints[:, config.RH_SL]   = rh[..., :3]
    joints = _normalise(joints)
    joints = _resample(joints, config.T_FIXED)
    return joints


def to_tensor(joints: np.ndarray) -> torch.Tensor:
    """(T, V, C) numpy → (C, T, V) tensor suitable for the encoder."""
    return torch.from_numpy(joints).permute(2, 0, 1).contiguous().float()


# ── Sample discovery ─────────────────────────────────────────────────────────

def _load_topic_translations() -> dict:
    p = config.DATA_ROOT / "topic_translations.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_signs(topic: str) -> list[dict]:
    p = config.DATA_ROOT / topic / "signs.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data and isinstance(data[0], str):
            return [{"uz": s, "ru": s, "en": s} for s in data]
        return data
    except Exception:
        return []


def list_topics() -> list[str]:
    return sorted(
        p.name for p in config.DATA_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_signers(topic: str) -> list[str]:
    topic_dir = config.DATA_ROOT / topic
    if not topic_dir.exists():
        return []
    return sorted(
        p.name for p in topic_dir.iterdir()
        if p.is_dir() and p.name.startswith("signer")
    )


def collect_all_samples() -> list[dict]:
    """Walk DATA_ROOT and collect every available keypoints.npz with its label.

    Returns list of dicts: {npz, topic, signer, sign_uz, sign_ru, sign_en, rep, label}
    where `label` is the globally unique class key `{topic}/{sign_uz}`.
    """
    samples: list[dict] = []
    for topic in list_topics():
        sign_meta = {s["uz"]: s for s in _load_signs(topic)}
        for signer in list_signers(topic):
            signer_dir = config.DATA_ROOT / topic / signer
            for sign_dir in sorted(signer_dir.iterdir()):
                if not sign_dir.is_dir() or sign_dir.name.startswith("."):
                    continue
                kp_dir = sign_dir / "keypoints"
                if not kp_dir.exists():
                    continue
                meta = sign_meta.get(sign_dir.name, {"uz": sign_dir.name, "ru": "", "en": ""})
                for rep_dir in sorted(kp_dir.iterdir()):
                    npz = rep_dir / "keypoints.npz"
                    if not npz.exists():
                        continue
                    try:
                        rep_idx = int(rep_dir.name.split("-")[1])
                    except Exception:
                        continue
                    samples.append({
                        "npz":     npz,
                        "topic":   topic,
                        "signer":  signer,
                        "sign_uz": meta.get("uz", sign_dir.name),
                        "sign_ru": meta.get("ru", ""),
                        "sign_en": meta.get("en", ""),
                        "rep":     rep_idx,
                        "label":   f"{topic}/{meta.get('uz', sign_dir.name)}",
                    })
    return samples


def group_by_label(samples: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        g[s["label"]].append(s)
    return dict(g)


def split_classes_by_rep(samples: list[dict], val_rep: int):
    """Leave-one-rep-out split.

    Returns (train_samples, val_samples) — every class is present in both
    as long as it has both `rep != val_rep` and `rep == val_rep` available.
    """
    train, val = [], []
    for s in samples:
        (val if s["rep"] == val_rep else train).append(s)
    return train, val


# ── Dataset ──────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    """Generic Dataset that returns (C, T, V) tensor + label string for each sample.

    Used both as the source for episodic sampling and as the prototype-building
    dataset.  Augmentation is delegated to a callable (see augment.py).
    """

    def __init__(self, samples: list[dict], augment: Optional[Callable] = None):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        joints = preprocess_npz(s["npz"])   # (T, V, 3)
        if self.augment is not None:
            joints = self.augment(joints)
        x = to_tensor(joints)               # (3, T, V)
        return x, s["label"]
