"""OpenHands pretrained-weight loader (best-effort).

Tries to download the AUTSL ST-GCN checkpoint and map keys onto our encoder.
If the download or mapping fails, returns False so the training loop can fall
back to scratch initialisation without aborting.

Known weight sources (subject to change):
    https://github.com/AI4Bharat/OpenHands  →  see model zoo / releases page

Manual install:
    Drop a checkpoint file as fewshot/pretrained_weights/stgcn_autsl.pth
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import urllib.error
import urllib.request

import torch
import torch.nn as nn

from fewshot import config


_LOCAL_CANDIDATES = [
    "stgcn_autsl.pth",
    "stgcn.pth",
    "encoder.pth",
]

# Direct-link mirrors that have hosted OpenHands weights in the past.
# These are checked in order; first hit wins. If none works, we silently fall
# back to scratch init.
_REMOTE_MIRRORS: list[str] = [
    # Add manual mirrors here if you have a known URL.
    # e.g. "https://huggingface.co/AI4Bharat/OpenHands-stgcn-autsl/resolve/main/stgcn.pth",
]


def _find_local_checkpoint() -> Optional[Path]:
    for name in _LOCAL_CANDIDATES:
        p = config.WEIGHTS_DIR / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _try_download_remote() -> Optional[Path]:
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.WEIGHTS_DIR / "stgcn_autsl.pth"
    for url in _REMOTE_MIRRORS:
        try:
            print(f"  downloading {url} …")
            urllib.request.urlretrieve(url, target)
            if target.exists() and target.stat().st_size > 0:
                return target
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            print(f"  download failed: {e}")
            continue
    return None


def _map_openhands_to_ours(src_state: dict, dst_state: dict) -> dict:
    """Best-effort key-renaming from OpenHands' ST-GCN to our naming.

    Both implementations follow the canonical Yan-et-al. layout, so most
    weights match by shape if you strip a model-name prefix.  Anything we
    can't map gets silently dropped.
    """
    # Drop prefix like 'model.' or 'encoder.'
    renamed: dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        new_k = k
        for prefix in ("model.", "encoder.", "module."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        renamed[new_k] = v

    # Now intersect with our state dict by exact key + shape match
    loaded: dict[str, torch.Tensor] = {}
    for k, v in dst_state.items():
        if k in renamed and renamed[k].shape == v.shape:
            loaded[k] = renamed[k]
    return loaded


def load_pretrained_into(encoder: nn.Module, verbose: bool = True) -> bool:
    """Try to load AUTSL pretrained weights into `encoder`.

    Returns True on (at least partial) success, False on full fallback.
    """
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = _find_local_checkpoint() or _try_download_remote()
    if ckpt_path is None:
        if verbose:
            print("  no pretrained checkpoint available → scratch init")
        return False

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if verbose:
            print(f"  could not read {ckpt_path}: {e}")
        return False

    src = raw.get("model_state_dict") or raw.get("state_dict") or raw.get("model") or raw
    if not isinstance(src, dict):
        if verbose:
            print(f"  unexpected checkpoint shape in {ckpt_path}")
        return False

    dst = encoder.state_dict()
    mapped = _map_openhands_to_ours(src, dst)
    n_loaded = len(mapped)
    n_total  = len(dst)

    if n_loaded == 0:
        if verbose:
            print(f"  checkpoint at {ckpt_path} matched 0/{n_total} of our params → scratch init")
        return False

    dst.update(mapped)
    encoder.load_state_dict(dst, strict=False)
    if verbose:
        print(f"  pretrained init: loaded {n_loaded}/{n_total} tensors from {ckpt_path.name}")
    return True
