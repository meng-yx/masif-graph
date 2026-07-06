"""Relational message passing + fusion head (Phase 2, design docs/03 §3), pure PyTorch.

Design decision D-P2.3: the frozen 80-D surface descriptor is NOT message-passed. The GNN
operates only on invariant chemistry+geometry (element/degree/backbone/aromatic/flex-depth node
features; bond-order + rotatable covalent edges; RBF spatial edges) and emits one role-independent
readout vector `g` per surface atom. The descriptor is fused at the head:

    fused_straight = Head(surf_straight ⊕ g)      # p1 / target role
    fused_flipped  = Head(surf_flipped  ⊕ g)      # p2 / binder role

Complementarity (straight vs flipped) stays entirely in the surface channel (as in the reference
flip trick); `g` is a symmetric, rotation-invariant context the head learns to use to make the
matched embedding robust to sidechain conformation. Under a fixed-backbone repack the surface
channel shifts but the covalent-anchored part of `g` does not, so the fused pair distance moves
less — the robustness mechanism the M2 ablation tests.

Ablation is by which edge types / features the GNN sees (AblationConfig), NOT separate models.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AblationConfig:
    use_covalent: bool = True
    use_rotatable: bool = True   # adds rotatable-flag edge feat + flex-depth node feat
    use_spatial: bool = True
    name: str = "full"


# node/edge feature dims produced by graph.build + graph.dataset
N_BASE = 10          # element(6)+backbone(1)+aromatic(1)+degree(1)+is_surface(1)
N_NODE_IN = N_BASE + 1   # + flex_depth slot (zeroed when use_rotatable is False)
N_COV_EDGE = 4 + 1       # bond-order one-hot(4) + rotatable flag(1, zeroed if not use_rotatable)
N_SP_EDGE = 16           # RBF


def _scatter_mean(msg, dst, n):
    out = torch.zeros(n, msg.shape[1], dtype=msg.dtype, device=msg.device)
    out.index_add_(0, dst, msg)
    deg = torch.zeros(n, dtype=msg.dtype, device=msg.device)
    deg.index_add_(0, dst, torch.ones(dst.shape[0], dtype=msg.dtype, device=msg.device))
    return out / deg.clamp(min=1.0).unsqueeze(1)


class RelLayer(nn.Module):
    """One relational MP layer: self transform + per-edge-type message aggregation."""

    def __init__(self, dim, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        self.self_lin = nn.Linear(dim, dim)
        if cfg.use_covalent:
            self.cov_lin = nn.Linear(dim + N_COV_EDGE, dim)
        if cfg.use_spatial:
            self.sp_lin = nn.Linear(dim + N_SP_EDGE, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, h, cov_edge, cov_feat, sp_edge, sp_feat):
        n = h.shape[0]
        agg = self.self_lin(h)
        if self.cfg.use_covalent and cov_edge.shape[1] > 0:
            src, dst = cov_edge[0], cov_edge[1]
            msg = self.cov_lin(torch.cat([h[src], cov_feat], dim=1))
            agg = agg + _scatter_mean(msg, dst, n)
        if self.cfg.use_spatial and sp_edge.shape[1] > 0:
            src, dst = sp_edge[0], sp_edge[1]
            msg = self.sp_lin(torch.cat([h[src], sp_feat], dim=1))
            agg = agg + _scatter_mean(msg, dst, n)
        return self.norm(h + self.act(agg))   # residual


class GraphEncoder(nn.Module):
    """Invariant GNN over one chain's atom graph → per-surface-atom readout `g`."""

    def __init__(self, cfg: AblationConfig, dim=64, n_layers=3, out_dim=32):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(N_NODE_IN, dim)
        self.layers = nn.ModuleList([RelLayer(dim, cfg) for _ in range(n_layers)])
        self.readout = nn.Linear(dim, out_dim)
        self.out_dim = out_dim

    def forward(self, g):
        """g: a dict of tensors for one chain (see graph.dataset.graph_to_tensors)."""
        surface_only = not (self.cfg.use_covalent or self.cfg.use_spatial)
        surf_idx = g["surf_idx"]           # rows (into node table) that are surface atoms
        if surface_only:
            return torch.zeros(surf_idx.shape[0], self.out_dim, device=surf_idx.device)
        h = self.embed(g["node_feat"])
        for layer in self.layers:
            h = layer(h, g["cov_edge"], g["cov_feat"], g["sp_edge"], g["sp_feat"])
        return self.readout(h[surf_idx])   # (n_surface, out_dim)


class FusionHead(nn.Module):
    """Head fusing the 80-D surface descriptor with the graph readout → fused embedding."""

    def __init__(self, graph_out_dim=32, desc_dim=80, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(desc_dim + graph_out_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, desc, g):
        return self.net(torch.cat([desc, g], dim=1))


class MaSIFGraphModel(nn.Module):
    """GraphEncoder + FusionHead. One forward per chain per role (straight/flipped)."""

    def __init__(self, cfg: AblationConfig, graph_dim=64, n_layers=3, graph_out_dim=32,
                 fused_dim=64):
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphEncoder(cfg, dim=graph_dim, n_layers=n_layers, out_dim=graph_out_dim)
        self.head = FusionHead(graph_out_dim=graph_out_dim, out_dim=fused_dim)

    def graph_readout(self, g):
        return self.encoder(g)   # (n_surface, graph_out_dim)

    def fuse(self, desc, g_readout):
        return self.head(desc, g_readout)
