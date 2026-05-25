"""ST-GCN encoder for skeleton sequences.

Input  : (B, C=3, T=64, V=48)   — batched skeleton sequences
Output : (B, EMBED_DIM)         — fixed-length embedding per sequence

Architecture follows Yan et al. 2018 "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition" with a 3-partition adjacency (identity,
inward, outward), so we can load OpenHands' AUTSL pretrained weights without
architectural changes.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fewshot import config


# ── Skeleton graph (48 joints) ────────────────────────────────────────────────

def _skeleton_edges() -> List[Tuple[int, int]]:
    """Define the kinematic skeleton edges over our 48-joint layout."""
    edges: List[Tuple[int, int]] = []

    # Pose arm chain (indices 0..5)
    edges += [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]

    # Hand topology (MediaPipe Hand: 21 landmarks)
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),                  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),                  # index
        (5, 9), (9, 10), (10, 11), (11, 12),             # middle
        (9, 13), (13, 14), (14, 15), (15, 16),           # ring
        (13, 17), (17, 18), (18, 19), (19, 20),          # pinky
        (0, 17),                                         # wrist→pinky base
    ]
    # Left hand at offset 6, right hand at offset 27
    for a, b in hand_edges:
        edges.append((6 + a, 6 + b))
        edges.append((27 + a, 27 + b))

    # Bridge pose wrists → hand wrists
    edges.append((4, 6))    # left pose-wrist → left hand-wrist
    edges.append((5, 27))   # right pose-wrist → right hand-wrist

    return edges


def _build_adjacency(num_nodes: int, edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Build 3-partition adjacency: identity, inward (towards body), outward.

    Returns (3, V, V) tensor, each partition normalised by inverse degree.
    """
    # Determine "root" of skeleton as the midpoint between shoulders (joint 0 and 1)
    # We'll use joint 0 (left shoulder) as proxy root for distance computation
    root = 0

    # BFS distance from root
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0

    # Hop distance from root using BFS
    dist = np.full(num_nodes, -1, dtype=np.int32)
    dist[root] = 0
    q = [root]
    while q:
        next_q = []
        for v in q:
            for u in range(num_nodes):
                if A[v, u] > 0 and dist[u] == -1:
                    dist[u] = dist[v] + 1
                    next_q.append(u)
        q = next_q

    # Build 3 partitions
    identity = np.eye(num_nodes, dtype=np.float32)
    inward   = np.zeros_like(A)
    outward  = np.zeros_like(A)
    for a, b in edges:
        # Edge (a,b): if b is closer to root → inward from a, outward from b
        if dist[a] >= 0 and dist[b] >= 0:
            if dist[a] > dist[b]:
                inward[a, b]  = 1.0
                outward[b, a] = 1.0
            elif dist[b] > dist[a]:
                inward[b, a]  = 1.0
                outward[a, b] = 1.0
            else:
                # Same distance → put both as "self-connection-like" (rare)
                inward[a, b]  = 1.0
                outward[a, b] = 1.0

    parts = np.stack([identity, inward, outward], axis=0)   # (3, V, V)

    # Normalise: A_norm = D^-1 A  (degree-normalized)
    for k in range(parts.shape[0]):
        deg = parts[k].sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        parts[k] = parts[k] / deg

    return torch.from_numpy(parts)   # (3, V, V) float32


# ── ST-GCN building blocks ────────────────────────────────────────────────────

class SpatialGCN(nn.Module):
    """Graph conv over V joints, partitioned across 3 adjacency matrices."""

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int = 3):
        super().__init__()
        self.num_partitions = num_partitions
        self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V),  A: (P, V, V)
        n, _, t, v = x.shape
        p = self.num_partitions
        x = self.conv(x)                              # (B, p*Cout, T, V)
        x = x.view(n, p, -1, t, v)                    # (B, P, Cout, T, V)
        x = torch.einsum("bpctv,pvw->bctw", x, A)     # graph conv
        return x.contiguous()


class STGCNBlock(nn.Module):
    """One ST-GCN layer = SpatialGCN + TemporalConv with residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 tcn_kernel: int = config.TCN_KERNEL, dropout: float = 0.0):
        super().__init__()
        self.gcn = SpatialGCN(in_channels, out_channels)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        pad = (tcn_kernel - 1) // 2
        self.tcn = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(tcn_kernel, 1),
                             stride=(stride, 1),
                             padding=(pad, 0))
        self.bn_tcn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=False)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn_gcn(x)
        x = F.relu(x, inplace=True)
        x = self.tcn(x)
        x = self.bn_tcn(x)
        x = self.drop(x)
        x = x + res
        return F.relu(x, inplace=True)


# ── Encoder ───────────────────────────────────────────────────────────────────

class STGCNEncoder(nn.Module):
    """Stack of ST-GCN blocks producing a fixed embedding.

    forward(x) → (B, EMBED_DIM)
    """

    def __init__(self,
                 in_channels: int = config.IN_CHANNELS,
                 hidden_dims: tuple = config.HIDDEN_DIMS,
                 embed_dim:   int = config.EMBED_DIM,
                 dropout:     float = config.DROPOUT):
        super().__init__()
        edges = _skeleton_edges()
        A = _build_adjacency(config.N_JOINTS, edges)   # (P, V, V)
        self.register_buffer("A", A)

        # Input batchnorm over (C, T, V) → flatten as (C*V) → BN1d
        self.data_bn = nn.BatchNorm1d(in_channels * config.N_JOINTS)

        # Build blocks: increase channels, halve time at boundaries
        blocks: list[STGCNBlock] = []
        prev = in_channels
        for i, h in enumerate(hidden_dims):
            stride = 2 if i > 0 else 1
            blocks.append(STGCNBlock(prev, h, stride=stride, dropout=dropout))
            blocks.append(STGCNBlock(h, h, stride=1, dropout=dropout))
            prev = h
        self.blocks = nn.ModuleList(blocks)

        # Global average pool over (T, V) → linear projection
        self.head = nn.Linear(hidden_dims[-1], embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        b, c, t, v = x.shape

        # Data BN: flatten (C, V) → (C*V), BN over time
        x = x.permute(0, 1, 3, 2).contiguous().view(b, c * v, t)
        x = self.data_bn(x)
        x = x.view(b, c, v, t).permute(0, 1, 3, 2).contiguous()    # (B, C, T, V)

        for block in self.blocks:
            x = block(x, self.A)

        # Global pool over (T, V)
        x = x.mean(dim=(2, 3))            # (B, Cout)
        return self.head(x)               # (B, EMBED_DIM)


def build_encoder() -> STGCNEncoder:
    return STGCNEncoder()
