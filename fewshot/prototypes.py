"""Build / load / save the global prototype database.

The database stores one prototype vector per sign label, computed as the mean
of that sign's available rep embeddings.

File format: .npz with three arrays
    labels      : (N,)  object array of class label strings
    prototypes  : (N, D) float32
    counts      : (N,)  int32 — number of reps that contributed to each prototype
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fewshot import config
from fewshot.data import collect_all_samples, group_by_label, preprocess_npz, to_tensor
from fewshot.encoder import build_encoder


def _load_encoder(ckpt_path: Path, device: torch.device):
    enc = build_encoder().to(device).eval()
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("encoder_state", state)
        enc.load_state_dict(sd, strict=False)
        print(f"  loaded encoder ← {ckpt_path}")
    else:
        print(f"  WARNING: no encoder checkpoint at {ckpt_path} — using random init")
    return enc


def build_database(encoder, samples: list[dict], device: torch.device,
                   batch_size: int = 16) -> dict:
    """Encode every sample and compute mean per label.

    Returns dict with keys 'labels' (list[str]), 'prototypes' (np.ndarray),
    'counts' (np.ndarray), 'meta' (dict per label with uz/ru/en/topic).
    """
    by_label = group_by_label(samples)
    labels = sorted(by_label.keys())

    proto_list, count_list = [], []
    meta_per_label: dict[str, dict] = {}

    with torch.no_grad():
        for label in labels:
            items = by_label[label]
            embeds: list[torch.Tensor] = []
            # Encode in batches
            for start in range(0, len(items), batch_size):
                batch_items = items[start:start + batch_size]
                xs = [to_tensor(preprocess_npz(s["npz"])) for s in batch_items]
                x = torch.stack(xs).to(device)
                e = encoder(x).cpu()           # (B, D)
                embeds.append(e)
            all_e = torch.cat(embeds, dim=0)
            mean_e = all_e.mean(dim=0)
            proto_list.append(mean_e.numpy())
            count_list.append(len(items))

            # Per-label metadata (use first sample's info)
            s0 = items[0]
            meta_per_label[label] = {
                "topic":   s0["topic"],
                "sign_uz": s0["sign_uz"],
                "sign_ru": s0["sign_ru"],
                "sign_en": s0["sign_en"],
            }
            print(f"  {label:60s}  reps={len(items):2d}  ‖p‖={float(np.linalg.norm(mean_e.numpy())):.3f}")

    return {
        "labels":     labels,
        "prototypes": np.stack(proto_list).astype(np.float32),
        "counts":     np.array(count_list, dtype=np.int32),
        "meta":       meta_per_label,
    }


def save_database(db: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        labels     = np.array(db["labels"], dtype=object),
        prototypes = db["prototypes"],
        counts     = db["counts"],
        meta       = np.array([json.dumps(db["meta"])], dtype=object),
    )


def load_database(in_path: Path) -> dict:
    data = np.load(in_path, allow_pickle=True)
    return {
        "labels":     list(data["labels"]),
        "prototypes": data["prototypes"],
        "counts":     data["counts"],
        "meta":       json.loads(str(data["meta"][0])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the global prototype database")
    ap.add_argument("--encoder", type=Path, default=config.ENCODER_CKPT,
                    help=f"Encoder checkpoint (default: {config.ENCODER_CKPT})")
    ap.add_argument("--out",     type=Path, default=config.DEFAULT_PROTO,
                    help=f"Output .npz path (default: {config.DEFAULT_PROTO})")
    ap.add_argument("--device",  default="auto",
                    help="cpu | cuda | mps | auto (default: auto)")
    ap.add_argument("--exclude-rep", type=int, default=None,
                    help="Hold out a rep from prototype-building (e.g. 0 for honest test)")
    args = ap.parse_args()

    device = config.resolve_device(args.device)
    print(f"  device: {device}")
    samples = collect_all_samples()
    if args.exclude_rep is not None:
        samples = [s for s in samples if s["rep"] != args.exclude_rep]
        print(f"  excluded rep-{args.exclude_rep} → {len(samples)} samples")

    encoder = _load_encoder(args.encoder, device)
    print(f"\n  building prototypes from {len(samples)} samples …\n")
    db = build_database(encoder, samples, device)
    save_database(db, args.out)
    print(f"\n  saved {len(db['labels'])} prototypes → {args.out}")


if __name__ == "__main__":
    main()
