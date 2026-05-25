"""Episodic sampler for N-way K-shot prototypical training."""
from __future__ import annotations

from typing import Optional
import random

import numpy as np
import torch

from fewshot import config
from fewshot.data import (
    SkeletonDataset, group_by_label, preprocess_npz, to_tensor,
)
from fewshot.augment import Augmenter


class EpisodeSampler:
    """Samples episodes of (support, query) tensors from a pool of labeled samples.

    Each call to `sample_episode()` returns:
        support_x : (N*K, C, T, V)   tensor
        support_y : (N*K,)           int tensor with per-class index in [0, N)
        query_x   : (N*Q, C, T, V)   tensor
        query_y   : (N*Q,)           int tensor with per-class index in [0, N)
    """

    def __init__(
        self,
        samples: list[dict],
        n_way:  int = config.N_WAY,
        k_shot: int = config.K_SHOT,
        q_query: int = config.Q_QUERY,
        augment: Optional[Augmenter] = None,
        rng: Optional[random.Random] = None,
    ):
        self.by_label = {
            label: items for label, items in group_by_label(samples).items()
            if len(items) >= k_shot + q_query
        }
        self.labels = list(self.by_label.keys())
        self.n_way  = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.augment = augment
        self.rng = rng or random.Random()
        if len(self.labels) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with ≥{k_shot+q_query} reps each, "
                f"got {len(self.labels)}."
            )

    def _load(self, sample: dict, augment: bool) -> torch.Tensor:
        joints = preprocess_npz(sample["npz"])
        if augment and self.augment is not None:
            joints = self.augment(joints)
        return to_tensor(joints)

    def sample_episode(self):
        chosen = self.rng.sample(self.labels, self.n_way)
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for ci, label in enumerate(chosen):
            items = self.by_label[label]
            picks = self.rng.sample(items, self.k_shot + self.q_query)
            for s in picks[:self.k_shot]:
                sup_x.append(self._load(s, augment=True))
                sup_y.append(ci)
            for s in picks[self.k_shot:]:
                qry_x.append(self._load(s, augment=True))
                qry_y.append(ci)
        sup_x = torch.stack(sup_x)
        qry_x = torch.stack(qry_x)
        sup_y = torch.tensor(sup_y, dtype=torch.long)
        qry_y = torch.tensor(qry_y, dtype=torch.long)
        return sup_x, sup_y, qry_x, qry_y

    @property
    def n_classes(self) -> int:
        return len(self.labels)
