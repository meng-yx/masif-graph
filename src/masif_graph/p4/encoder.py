"""Phase-4 heterogeneous GNN encoder E (from scratch; design `docs/08-phase4-design.md` §4).

One shared encoder maps a chain's HeteroSurfaceGraph -> a per-surface-atom embedding `z`. Message
passing runs over the three invariant edge types (atom-atom covalent, vertex-vertex mesh,
vertex-atom), so a surface atom's embedding is informed by its sidechain connectivity AND the
surface signal that diffuses over the mesh and onto it. Nothing is frozen; the surface descriptor
is *learned by this GNN* (vertex-vertex MP is the learnable geodesic conv that replaces MaSIF's CNN).

Implemented with torch **core ops only** (`index_add_`, no torch_scatter / PyG), so the exact same
module imports in the `masif-graph` (CPU, Jed) and `atomsurf_h100` (GPU, Kuma) envs — the
embedding-level rotation-invariance gate runs on Jed CPU, training runs on Kuma.

Invariance guarantee: the forward reads only node features (pose-independent) and edge features
(distances + cos-angles, provably invariant — M0 gate). No coordinate ever enters the network. So
`E(rotate(graph)) == E(graph)` up to float error; `encoder_rotation_maxdiff` checks it numerically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def scatter_mean(msg: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean-aggregate messages (E, d) into (n, d) by destination `index` (E,). Core-op only."""
    d = msg.shape[1]
    out = torch.zeros(n, d, dtype=msg.dtype, device=msg.device)
    out.index_add_(0, index, msg)
    cnt = torch.zeros(n, 1, dtype=msg.dtype, device=msg.device)
    cnt.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=msg.dtype, device=msg.device))
    return out / cnt.clamp_min(1.0)


def _mlp(d_in, d_out, hidden=None):
    hidden = hidden or d_out
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))


class HeteroMPLayer(nn.Module):
    """One heterogeneous message-passing layer. Updates atom + vertex node states (residual)."""

    def __init__(self, d: int, d_aa: int, d_vv: int, d_va: int):
        super().__init__()
        # per-edge-type message functions: message = MLP([h_src, edge_feat])
        self.msg_aa = _mlp(d + d_aa, d)   # atom  -> atom  (covalent, both directions pre-built)
        self.msg_vv = _mlp(d + d_vv, d)   # vertex-> vertex (mesh, symmetrized)
        self.msg_va = _mlp(d + d_va, d)   # vertex-> atom
        self.msg_av = _mlp(d + d_va, d)   # atom  -> vertex
        # node updates: h_new = h + MLP([h, aggregated messages])
        self.upd_atom = _mlp(3 * d, d)    # [h_atom, agg_aa, agg_va]
        self.upd_vert = _mlp(3 * d, d)    # [h_vert, agg_vv, agg_av]
        self.norm_atom = nn.LayerNorm(d)
        self.norm_vert = nn.LayerNorm(d)

    def forward(self, ha, hv, g):
        na, nv = ha.shape[0], hv.shape[0]
        # atom <- atom (covalent). g.aa_edge is (2,E): row0 src, row1 dst.
        if g["aa_edge"].shape[1] > 0:
            src, dst = g["aa_edge"][0], g["aa_edge"][1]
            m = self.msg_aa(torch.cat([ha[src], g["aa_feat"]], dim=1))
            agg_aa = scatter_mean(m, dst, na)
        else:
            agg_aa = torch.zeros_like(ha)
        # atom <- vertex (va: vertex src -> atom dst)
        if g["va_v"].shape[0] > 0:
            m = self.msg_va(torch.cat([hv[g["va_v"]], g["va_feat"]], dim=1))
            agg_va = scatter_mean(m, g["va_a"], na)
            # vertex <- atom (av: atom src -> vertex dst), same edges reversed, same edge feat
            m2 = self.msg_av(torch.cat([ha[g["va_a"]], g["va_feat"]], dim=1))
            agg_av = scatter_mean(m2, g["va_v"], nv)
        else:
            agg_va = torch.zeros_like(ha)
            agg_av = torch.zeros_like(hv)
        # vertex <- vertex (mesh, symmetrized in the dataset)
        if g["vv_edge"].shape[1] > 0:
            src, dst = g["vv_edge"][0], g["vv_edge"][1]
            m = self.msg_vv(torch.cat([hv[src], g["vv_feat"]], dim=1))
            agg_vv = scatter_mean(m, dst, nv)
        else:
            agg_vv = torch.zeros_like(hv)

        ha = self.norm_atom(ha + self.upd_atom(torch.cat([ha, agg_aa, agg_va], dim=1)))
        hv = self.norm_vert(hv + self.upd_vert(torch.cat([hv, agg_vv, agg_av], dim=1)))
        return ha, hv


class HeteroEncoder(nn.Module):
    """Heterogeneous GNN: (atom_feat, vert_feat, 3 edge types) -> per-surface-atom embedding z (n_surf, d_out)."""

    def __init__(self, f_atom: int, f_vert: int, d_aa: int, d_vv: int, d_va: int,
                 d: int = 64, d_out: int = 32, n_layers: int = 4):
        super().__init__()
        self.embed_atom = _mlp(f_atom, d)
        self.embed_vert = _mlp(f_vert, d)
        self.layers = nn.ModuleList([HeteroMPLayer(d, d_aa, d_vv, d_va) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d_out))
        self.d, self.d_out = d, d_out

    def forward(self, g) -> torch.Tensor:
        ha = self.embed_atom(g["atom_feat"])
        hv = self.embed_vert(g["vert_feat"])
        for layer in self.layers:
            ha, hv = layer(ha, hv, g)
        z = self.readout(ha[g["surf_node_idx"]])  # gather surface atoms, ordered by surf row
        return z


@torch.no_grad()
def encoder_rotation_maxdiff(encoder: HeteroEncoder, g_orig: dict, g_rot: dict) -> float:
    """Embedding-level rotation gate: max |z(orig) - z(rot)| (should be ~float eps)."""
    encoder.eval()
    z0 = encoder(g_orig)
    z1 = encoder(g_rot)
    return float((z0 - z1).abs().max().item())
