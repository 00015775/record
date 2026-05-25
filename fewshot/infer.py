"""Inference CLI: classify a .npz file or a whole topic/signer against the proto DB.

Run:
    # Single file
    python -m fewshot.infer --npz path/to/keypoints.npz --top 5

    # All recorded reps from a topic/signer (top-1 accuracy)
    python -m fewshot.infer --topic Alifbo --signer signer01

    # Evaluate on a held-out rep across the whole dataset
    python -m fewshot.infer --eval-rep 0
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
from fewshot.prototypes import load_database
from fewshot.proto import similarity


def _load_encoder(ckpt_path: Path, device: torch.device):
    enc = build_encoder().to(device).eval()
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("encoder_state", state)
        enc.load_state_dict(sd, strict=False)
    return enc


@torch.no_grad()
def predict(encoder, npz_path: Path, prototypes: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    """Return logits (1, N_classes) for one .npz file."""
    joints = preprocess_npz(npz_path)
    x = to_tensor(joints).unsqueeze(0).to(device)         # (1, 3, T, V)
    e = encoder(x)                                        # (1, D)
    logits = similarity(e, prototypes)                    # (1, N)
    return logits


def topk_labels(logits: torch.Tensor, labels: list[str], k: int = 5):
    probs = torch.softmax(logits, dim=-1)[0]
    top = probs.topk(min(k, probs.numel()))
    return [(labels[i], float(p)) for p, i in zip(top.values.tolist(), top.indices.tolist())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",      type=Path, default=config.DEFAULT_PROTO)
    ap.add_argument("--encoder", type=Path, default=config.ENCODER_CKPT)
    ap.add_argument("--device",  default="auto",
                    help="cpu | cuda | mps | auto (default: auto)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--npz", type=Path, help="Classify a single keypoints.npz")
    mode.add_argument("--topic", help="Evaluate every rep under this topic/signer")
    mode.add_argument("--eval-rep", type=int,
                      help="Evaluate top-1/top-5 on this rep across whole dataset")
    ap.add_argument("--signer", default="signer01")
    ap.add_argument("--top",    type=int, default=5)
    args = ap.parse_args()

    device = config.resolve_device(args.device)
    print(f"  device: {device}")
    db = load_database(args.db)
    labels = db["labels"]
    prototypes = torch.from_numpy(db["prototypes"]).to(device)
    encoder = _load_encoder(args.encoder, device)
    print(f"  loaded {len(labels)} prototypes from {args.db.name}")

    # Single-file inference
    if args.npz:
        logits = predict(encoder, args.npz, prototypes, device)
        for i, (lbl, p) in enumerate(topk_labels(logits, labels, args.top), 1):
            meta = db["meta"].get(lbl, {})
            disp = f"{meta.get('sign_uz', lbl):<14}" if meta else lbl
            print(f"  {i}. {disp}  prob={p*100:5.1f}%   ({lbl})")
        return

    # Topic-wide eval
    samples = collect_all_samples()
    if args.topic:
        samples = [s for s in samples if s["topic"] == args.topic and s["signer"] == args.signer]
        if not samples:
            print(f"no samples for {args.topic}/{args.signer}")
            return
        title = f"{args.topic} / {args.signer}"
    elif args.eval_rep is not None:
        samples = [s for s in samples if s["rep"] == args.eval_rep]
        title = f"rep-{args.eval_rep} (all topics/signers)"
    else:
        print("specify --npz, --topic, or --eval-rep")
        return

    print(f"\n  evaluating {len(samples)} samples on {title}\n")
    top1 = top5 = total = 0
    per_class_correct: dict[str, int] = {}
    per_class_total:   dict[str, int] = {}
    confusions: list[tuple[str, str]] = []
    # cache (sample, logits_cpu) for follow-up analyses
    cached: list[tuple[dict, torch.Tensor]] = []

    for s in samples:
        logits = predict(encoder, s["npz"], prototypes, device)
        cached.append((s, logits.cpu()))
        true = s["label"]
        per_class_total[true] = per_class_total.get(true, 0) + 1
        top = topk_labels(logits, labels, 5)
        top1_correct = (top[0][0] == true)
        top5_correct = any(t[0] == true for t in top)
        top1 += int(top1_correct)
        top5 += int(top5_correct)
        total += 1
        per_class_correct[true] = per_class_correct.get(true, 0) + int(top1_correct)
        if not top1_correct:
            confusions.append((true, top[0][0]))

    print(f"  total: {total}")
    print(f"  top-1: {top1}/{total} = {top1/total*100:5.1f}%")
    print(f"  top-5: {top5}/{total} = {top5/total*100:5.1f}%")

    # Per-topic breakdown (always shown for multi-topic evals)
    topics = sorted({lbl.split("/")[0] for lbl in per_class_total.keys()})
    topic_idx_map = {
        t: [i for i, lbl in enumerate(labels) if lbl.startswith(t + "/")]
        for t in topics
    }

    if len(topics) > 1:
        print(f"\n  per-topic accuracy:")
        print(f"  {'topic':<40s} {'reps':>5} {'top-1':>7} {'top-5':>7} {'within-topic top-1':>20}")
        print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*20}")
        for topic in topics:
            topic_samples = [(s, l) for (s, l) in cached if s["topic"] == topic]
            n = len(topic_samples)
            t1 = t5 = wt1 = 0
            t_idx = topic_idx_map[topic]
            t_idx_t = torch.tensor(t_idx)
            for s, logits in topic_samples:
                true = s["label"]
                # full-DB top-1 / top-5
                top = topk_labels(logits, labels, 5)
                if top[0][0] == true: t1 += 1
                if any(x[0] == true for x in top): t5 += 1
                # within-topic top-1: only score against this topic's prototypes
                within_logits = logits[0, t_idx_t]
                pred_local = within_logits.argmax().item()
                pred_lbl = labels[t_idx[pred_local]]
                if pred_lbl == true: wt1 += 1
            print(f"  {topic:<40s} {n:>5d} {t1/n*100:>6.1f}% {t5/n*100:>6.1f}% {wt1/n*100:>19.1f}%")

    # Most-confused topic pairs
    if confusions:
        from collections import Counter
        pair_counts = Counter((t.split('/')[0], p.split('/')[0]) for t, p in confusions)
        print(f"\n  most-frequent topic-pair confusions (true → predicted):")
        for (t_top, p_top), n in pair_counts.most_common(10):
            same = "  (same topic)" if t_top == p_top else ""
            print(f"    {t_top:35s} → {p_top:35s}  ×{n}{same}")


if __name__ == "__main__":
    main()
