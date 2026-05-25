"""Episodic fine-tuning of the ST-GCN encoder with prototypical loss.

Run:
    python -m fewshot.train
    python -m fewshot.train --val-rep 0 --episodes 6000
    python -m fewshot.train --device cuda --scratch    # skip pretrained init
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fewshot import config
from fewshot.augment import Augmenter
from fewshot.data import collect_all_samples, split_classes_by_rep
from fewshot.encoder import build_encoder
from fewshot.episodes import EpisodeSampler
from fewshot.pretrained import load_pretrained_into
from fewshot.proto import proto_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(encoder, sampler: EpisodeSampler, device: torch.device,
             episodes: int) -> tuple[float, float]:
    encoder.eval()
    total_loss = 0.0
    total_acc  = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            sup_x, sup_y, qry_x, qry_y = sampler.sample_episode()
            sup_x = sup_x.to(device); sup_y = sup_y.to(device)
            qry_x = qry_x.to(device); qry_y = qry_y.to(device)
            sup_e = encoder(sup_x)
            qry_e = encoder(qry_x)
            loss, acc, _ = proto_loss(sup_e, sup_y, qry_e, qry_y, n_way=sampler.n_way)
            total_loss += loss.item()
            total_acc  += acc
    encoder.train()
    return total_loss / episodes, total_acc / episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-rep",  type=int, default=0,
                    help="Rep index held out for evaluation (default: 0)")
    ap.add_argument("--episodes", type=int, default=config.EPISODES)
    ap.add_argument("--lr",       type=float, default=config.LR)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--n-way",    type=int, default=config.N_WAY)
    ap.add_argument("--k-shot",   type=int, default=config.K_SHOT)
    ap.add_argument("--q-query",  type=int, default=config.Q_QUERY)
    ap.add_argument("--device",   default="auto",
                    help="cpu | cuda | mps | auto (default: auto-detects best available)")
    ap.add_argument("--scratch",  action="store_true",
                    help="Skip pretrained-weight load")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--save",     type=Path, default=config.ENCODER_CKPT,
                    help=f"Checkpoint save path (default: {config.ENCODER_CKPT})")
    args = ap.parse_args()

    set_seed(args.seed)
    device = config.resolve_device(args.device)
    print(f"  device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    all_samples = collect_all_samples()
    if not all_samples:
        raise SystemExit(f"No samples under {config.DATA_ROOT}")

    train_samples, val_samples = split_classes_by_rep(all_samples, val_rep=args.val_rep)

    # ── Fit episode sizes to available data ──────────────────────────────────
    from fewshot.data import group_by_label
    train_by = group_by_label(train_samples)
    train_counts = sorted({len(v) for v in train_by.values()})
    min_train_reps = min((len(v) for v in train_by.values()), default=0)
    print(f"  train rep counts: {train_counts}  (min per class: {min_train_reps})")

    needed = args.k_shot + args.q_query
    if min_train_reps < needed:
        max_k = max(1, min_train_reps - args.q_query)
        print(f"  ⚠ requested K={args.k_shot} + Q={args.q_query} = {needed} reps but "
              f"only {min_train_reps} available per class — clamping K_SHOT to {max_k}")
        args.k_shot = max_k
        if args.k_shot + args.q_query > min_train_reps:
            args.q_query = max(1, min_train_reps - args.k_shot)
            print(f"    also clamping Q_QUERY to {args.q_query}")

    train_sampler = EpisodeSampler(
        train_samples,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        augment=Augmenter(),
    )

    # Validation sampler reuses train pool for class diversity, but draws unseen
    # queries from val_samples for each chosen class.
    val_sampler = _build_val_sampler(train_samples, val_samples,
                                     n_way=args.n_way, k_shot=args.k_shot,
                                     q_query=args.q_query)

    print(f"  total samples:   {len(all_samples)}")
    print(f"  train samples:   {len(train_samples)}")
    print(f"  val samples:     {len(val_samples)}")
    print(f"  train classes:   {train_sampler.n_classes}")
    print(f"  val classes:     {val_sampler.n_classes if val_sampler else 0}")
    print(f"  episode:         {args.n_way}-way {args.k_shot}-shot, {args.q_query}-query")

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = build_encoder().to(device)
    if not args.scratch:
        print("\n  attempting pretrained init …")
        load_pretrained_into(encoder)
    else:
        print("\n  --scratch: skipping pretrained init")

    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.episodes - config.WARMUP_EPS, eta_min=args.lr * 0.05
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    t0 = time.time()
    log_loss = log_acc = 0.0
    log_n    = 0

    print(f"\n  starting {args.episodes} episodes\n")
    for ep in range(1, args.episodes + 1):
        sup_x, sup_y, qry_x, qry_y = train_sampler.sample_episode()
        sup_x = sup_x.to(device); sup_y = sup_y.to(device)
        qry_x = qry_x.to(device); qry_y = qry_y.to(device)

        sup_e = encoder(sup_x)
        qry_e = encoder(qry_x)
        loss, acc, _ = proto_loss(sup_e, sup_y, qry_e, qry_y, n_way=args.n_way)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optim.step()

        if ep > config.WARMUP_EPS:
            sched.step()

        log_loss += loss.item(); log_acc += acc; log_n += 1

        if ep % config.LOG_EVERY == 0:
            print(f"  ep {ep:5d}/{args.episodes}  "
                  f"loss={log_loss/log_n:.4f}  acc={log_acc/log_n*100:5.1f}%  "
                  f"lr={optim.param_groups[0]['lr']:.2e}")
            log_loss = log_acc = 0.0; log_n = 0

        if val_sampler is not None and ep % config.EVAL_EVERY == 0:
            v_loss, v_acc = evaluate(encoder, val_sampler, device, config.EVAL_EPISODES)
            print(f"    ── val: loss={v_loss:.4f}  acc={v_acc*100:5.1f}%  "
                  f"(best {best_val_acc*100:.1f}%)")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save({"encoder_state": encoder.state_dict(),
                            "ep": ep, "val_acc": v_acc}, args.save)
                print(f"    ★ saved → {args.save}")

    elapsed = time.time() - t0
    # Always save final checkpoint
    final_path = args.save.with_name(args.save.stem + "_final.pt")
    torch.save({"encoder_state": encoder.state_dict(),
                "ep": args.episodes, "val_acc": best_val_acc}, final_path)
    print(f"\n  done in {elapsed:.0f}s — best val acc {best_val_acc*100:.1f}%")
    print(f"  best  ckpt → {args.save}")
    print(f"  final ckpt → {final_path}")


def _build_val_sampler(train_samples: list[dict], val_samples: list[dict],
                       n_way: int, k_shot: int, q_query: int):
    """Build a sampler that uses train reps for support and val reps for query.

    This matches the deployment scenario: existing reps form prototypes, a new
    rep is the query. If a class has no val rep it is dropped from the sampler.
    """
    if not val_samples:
        return None

    from fewshot.data import group_by_label
    from fewshot.augment import Augmenter

    by_tr = group_by_label(train_samples)
    by_va = group_by_label(val_samples)
    classes = [c for c in by_tr if c in by_va and len(by_tr[c]) >= k_shot]
    if len(classes) < n_way:
        print(f"  ⚠ not enough classes for val sampler ({len(classes)} < {n_way})")
        return None

    sampler = EpisodeSampler.__new__(EpisodeSampler)
    sampler.by_label = {c: by_tr[c] for c in classes}
    sampler.labels   = classes
    sampler.n_way    = n_way
    sampler.k_shot   = k_shot
    sampler.q_query  = q_query
    sampler.augment  = None
    sampler.rng      = random.Random(0)
    sampler._val_pool = {c: by_va[c] for c in classes}

    # Patch sample_episode so queries come from val_pool
    original_load = sampler._load

    def sample_with_val_query(self):
        chosen = self.rng.sample(self.labels, self.n_way)
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for ci, label in enumerate(chosen):
            sup_items = self.rng.sample(self.by_label[label], self.k_shot)
            qry_items = self.rng.sample(
                self._val_pool[label], min(self.q_query, len(self._val_pool[label]))
            )
            for s in sup_items:
                sup_x.append(self._load(s, augment=False)); sup_y.append(ci)
            for s in qry_items:
                qry_x.append(self._load(s, augment=False)); qry_y.append(ci)
        sup_x = torch.stack(sup_x); qry_x = torch.stack(qry_x)
        return sup_x, torch.tensor(sup_y, dtype=torch.long), \
               qry_x, torch.tensor(qry_y, dtype=torch.long)

    sampler.sample_episode = sample_with_val_query.__get__(sampler, EpisodeSampler)
    return sampler


if __name__ == "__main__":
    main()
