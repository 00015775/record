"""Prototypical Network loss + similarity metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fewshot import config


def compute_prototypes(support_embeds: torch.Tensor, support_y: torch.Tensor,
                       n_way: int) -> torch.Tensor:
    """support_embeds: (N*K, D), support_y: (N*K,) → prototypes (N, D)."""
    D = support_embeds.size(-1)
    proto = support_embeds.new_zeros((n_way, D))
    counts = support_embeds.new_zeros((n_way,))
    for cls in range(n_way):
        mask = support_y == cls
        if mask.any():
            proto[cls] = support_embeds[mask].mean(dim=0)
            counts[cls] = mask.sum().float()
    return proto


def similarity(query: torch.Tensor, prototypes: torch.Tensor,
               metric: str = config.DISTANCE,
               temperature: float = config.TEMPERATURE) -> torch.Tensor:
    """Return logits (Q, N) — higher = more similar.

    cosine    : T * cos_sim(q, p)
    euclidean : -||q - p||^2
    """
    if metric == "cosine":
        q = F.normalize(query, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        return temperature * (q @ p.t())
    if metric == "euclidean":
        # (Q, 1, D) - (1, N, D) → (Q, N, D)
        diff = query.unsqueeze(1) - prototypes.unsqueeze(0)
        return -(diff * diff).sum(dim=-1)
    raise ValueError(f"unknown metric: {metric}")


def proto_loss(support_embeds: torch.Tensor, support_y: torch.Tensor,
               query_embeds: torch.Tensor, query_y: torch.Tensor,
               n_way: int):
    """Standard prototypical-network cross-entropy loss + accuracy."""
    proto = compute_prototypes(support_embeds, support_y, n_way)   # (N, D)
    logits = similarity(query_embeds, proto)                       # (Q, N)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc, logits
