"""Add a new sign to the prototype DB without retraining the encoder.

Records → keypoints already extracted → this script:
  1. Walks Data_Numpy_Arrays_RSL_UzSL/<topic>/<signer>/<sign>/keypoints/
  2. Encodes every available rep
  3. Stores the mean embedding as the new prototype
  4. Saves the updated DB

Run:
    python -m fewshot.add_sign --topic Alifbo --sign A
    python -m fewshot.add_sign --topic Tanishish --sign Salom --signer signer02
    python -m fewshot.add_sign --topic Sonlar --sign "Bir" --refresh-all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fewshot import config
from fewshot.data import collect_all_samples, preprocess_npz, to_tensor
from fewshot.encoder import build_encoder
from fewshot.prototypes import load_database, save_database, build_database


def _load_encoder(ckpt_path: Path, device: torch.device):
    enc = build_encoder().to(device).eval()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("encoder_state", state)
        enc.load_state_dict(sd, strict=False)
        print(f"  encoder ← {ckpt_path}")
    else:
        print(f"  WARNING: encoder not found at {ckpt_path}")
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic",   required=True)
    ap.add_argument("--sign",    help="Sign name (uz). Required unless --refresh-all is set.")
    ap.add_argument("--signer",  default="signer01")
    ap.add_argument("--db",      type=Path, default=config.DEFAULT_PROTO)
    ap.add_argument("--encoder", type=Path, default=config.ENCODER_CKPT)
    ap.add_argument("--device",  default="auto",
                    help="cpu | cuda | mps | auto (default: auto)")
    ap.add_argument("--refresh-all", action="store_true",
                    help="Rebuild every prototype in the DB from current data")
    args = ap.parse_args()

    device = config.resolve_device(args.device)
    print(f"  device: {device}")
    encoder = _load_encoder(args.encoder, device)

    if args.refresh_all:
        all_samples = collect_all_samples()
        print(f"\n  refreshing all {len({s['label'] for s in all_samples})} prototypes …\n")
        db = build_database(encoder, all_samples, device)
        save_database(db, args.db)
        print(f"\n  saved {len(db['labels'])} prototypes → {args.db}")
        return

    if not args.sign:
        ap.error("--sign is required unless --refresh-all is set")

    # Locate samples for the target sign
    samples = [
        s for s in collect_all_samples()
        if s["topic"] == args.topic and s["signer"] == args.signer and s["sign_uz"] == args.sign
    ]
    if not samples:
        print(f"  no reps found for {args.topic}/{args.signer}/{args.sign}")
        return

    label = f"{args.topic}/{args.sign}"
    print(f"\n  encoding {len(samples)} reps of '{label}' …")

    embeds = []
    with torch.no_grad():
        for s in samples:
            x = to_tensor(preprocess_npz(s["npz"])).unsqueeze(0).to(device)
            e = encoder(x).cpu().numpy()[0]
            embeds.append(e)
    proto = np.mean(np.stack(embeds), axis=0).astype(np.float32)
    print(f"  prototype norm: {float(np.linalg.norm(proto)):.3f}")

    # Update DB (or create new)
    if args.db.exists():
        db = load_database(args.db)
    else:
        db = {"labels": [], "prototypes": np.zeros((0, proto.shape[0]), np.float32),
              "counts": np.zeros((0,), np.int32), "meta": {}}

    if label in db["labels"]:
        idx = db["labels"].index(label)
        db["prototypes"][idx] = proto
        db["counts"][idx] = len(samples)
        action = "updated"
    else:
        db["labels"].append(label)
        db["prototypes"] = np.vstack([db["prototypes"], proto[None, :]])
        db["counts"]     = np.concatenate([db["counts"], [len(samples)]])
        action = "added"

    s0 = samples[0]
    db["meta"][label] = {
        "topic":   s0["topic"],
        "sign_uz": s0["sign_uz"],
        "sign_ru": s0["sign_ru"],
        "sign_en": s0["sign_en"],
    }

    save_database(db, args.db)
    print(f"\n  {action} '{label}' — DB now has {len(db['labels'])} prototypes → {args.db}")


if __name__ == "__main__":
    main()
