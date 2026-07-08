"""Phase-4 dataset: load PASS-1 `.npz` -> torch graph dicts + contacts (either env; Kuma for training).

Builds the exact graph-dict interface `p4.encoder.HeteroEncoder` consumes:
  atom_feat, vert_feat, aa_edge(2,E), aa_feat(E,5), vv_edge(2,E) [symmetrized],
  vv_feat(E,9), va_v, va_a, va_feat(E,9), surf_node_idx.
Edge geometric scalars (dist) are RBF-expanded here (cos passed through); all invariant.
Also exposes the mean-pooled frozen descriptors + coords for the frozen-ceiling reference and
negative sampling in eval.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def rbf(dist: torch.Tensor, n: int, d_max: float) -> torch.Tensor:
    centers = torch.linspace(0.0, d_max, n, dtype=dist.dtype, device=dist.device)
    gamma = 1.0 / ((centers[1] - centers[0]) ** 2 + 1e-9)
    return torch.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2)


# edge-feature dims (must match encoder construction): aa=5, vv=9, va=9
D_AA, D_VV, D_VA = 5, 9, 8 + 1
VV_RBF_N, VV_RBF_DMAX = 8, 4.0
VA_RBF_N, VA_RBF_DMAX = 8, 5.0


def load_chain_graph(npz_path, device="cpu"):
    z = np.load(npz_path)
    # nan_to_num guards against rare non-finite reference features (e.g. undefined shape-index at a
    # few degenerate MSMS vertices — found in 1AKJ_AB_DE p1: 5 NaN si). 0 is the neutral normalized
    # value; a single NaN input otherwise propagates through the GNN and corrupts training weights.
    t = lambda a, dt=torch.float32: torch.tensor(
        np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0), dtype=dt, device=device)
    li = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)

    aa_edge = li(z["aa_edge"])
    aa_feat = torch.cat([t(z["aa_order"]).reshape(-1, 4), t(z["aa_rot"]).reshape(-1, 1)], dim=1) \
        if aa_edge.shape[1] > 0 else torch.zeros(0, D_AA, device=device)

    # vertex-vertex: symmetrize (npz stores canonical undirected (2,E))
    vv = li(z["vv_edge"])
    vv_dist = t(z["vv_dist"]); vv_cos = t(z["vv_cos"])
    if vv.shape[1] > 0:
        vv_edge = torch.cat([vv, vv.flip(0)], dim=1)
        vv_dist = torch.cat([vv_dist, vv_dist]); vv_cos = torch.cat([vv_cos, vv_cos])
        vv_feat = torch.cat([rbf(vv_dist, VV_RBF_N, VV_RBF_DMAX), vv_cos[:, None]], dim=1)
    else:
        vv_edge = torch.zeros(2, 0, dtype=torch.long, device=device)
        vv_feat = torch.zeros(0, D_VV, device=device)

    va_v = li(z["va_v"]); va_a = li(z["va_a"])
    va_dist = t(z["va_dist"]); va_cos = t(z["va_cos"])
    va_feat = torch.cat([rbf(va_dist, VA_RBF_N, VA_RBF_DMAX), va_cos[:, None]], dim=1) \
        if va_v.shape[0] > 0 else torch.zeros(0, D_VA, device=device)

    return {
        "atom_feat": t(z["atom_feat"]),
        "vert_feat": t(z["vert_feat"]),
        "aa_edge": aa_edge, "aa_feat": aa_feat,
        "vv_edge": vv_edge, "vv_feat": vv_feat,
        "va_v": va_v, "va_a": va_a, "va_feat": va_feat,
        "surf_node_idx": li(z["surf_node_idx"]),
        "n_surf": int(z["n_surf"]),
        # references (not fed to the encoder):
        "desc_straight": t(z["desc_straight"]),
        "desc_flipped": t(z["desc_flipped"]),
        "coord": t(z["coord"]),
    }


def hetero_to_dict(g, device="cpu"):
    """Convert an in-memory HeteroSurfaceGraph (graph/hetero.py) to the encoder graph-dict.

    Mirrors `load_chain_graph` but from the dataclass — used for the embedding-level rotation gate
    (build g on original vs rotated coords, encode both, compare z)."""
    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    li = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)
    aa_edge = li(g.aa_edge)
    aa_feat = torch.cat([t(g.aa_order).reshape(-1, 4), t(g.aa_rot).reshape(-1, 1)], dim=1) \
        if aa_edge.shape[1] > 0 else torch.zeros(0, D_AA, device=device)
    vv = li(g.vv_edge); vv_dist = t(g.vv_dist); vv_cos = t(g.vv_cos)
    if vv.shape[1] > 0:
        vv_edge = torch.cat([vv, vv.flip(0)], dim=1)
        vv_dist = torch.cat([vv_dist, vv_dist]); vv_cos = torch.cat([vv_cos, vv_cos])
        vv_feat = torch.cat([rbf(vv_dist, VV_RBF_N, VV_RBF_DMAX), vv_cos[:, None]], dim=1)
    else:
        vv_edge = torch.zeros(2, 0, dtype=torch.long, device=device); vv_feat = torch.zeros(0, D_VV, device=device)
    va_v = li(g.va_v); va_a = li(g.va_a); va_dist = t(g.va_dist); va_cos = t(g.va_cos)
    va_feat = torch.cat([rbf(va_dist, VA_RBF_N, VA_RBF_DMAX), va_cos[:, None]], dim=1) \
        if va_v.shape[0] > 0 else torch.zeros(0, D_VA, device=device)
    idx = np.nonzero(g.atom_surf_row >= 0)[0]
    sni = idx[np.argsort(g.atom_surf_row[idx])]
    return {
        "atom_feat": t(g.atom_feat), "vert_feat": t(g.vert_feat),
        "aa_edge": aa_edge, "aa_feat": aa_feat,
        "vv_edge": vv_edge, "vv_feat": vv_feat,
        "va_v": va_v, "va_a": va_a, "va_feat": va_feat,
        "surf_node_idx": li(sni), "n_surf": int(g.n_surf),
    }


class ComplexP4:
    """Holo complex: p1/p2 graph dicts + contact positive rows (p1_row, p2_row)."""

    def __init__(self, data_dir, cid, device="cpu"):
        self.cid = cid
        self.p1 = load_chain_graph(os.path.join(data_dir, f"{cid}__holo__p1.npz"), device)
        self.p2 = load_chain_graph(os.path.join(data_dir, f"{cid}__holo__p2.npz"), device)
        c = np.load(os.path.join(data_dir, f"{cid}__contacts.npz"))
        self.pos = torch.tensor(c["pos"].reshape(-1, 2), dtype=torch.long, device=device)
        # sc-filtered contacts (MaSIF's clean-complementarity set; the ~0.90 frozen-ceiling gate)
        psc = c["pos_sc"].reshape(-1, 2) if "pos_sc" in c.files else np.zeros((0, 2), np.int64)
        self.pos_sc = torch.tensor(psc, dtype=torch.long, device=device)

    @property
    def n_pos(self):
        return self.pos.shape[0]

    def to(self, device):
        """Return a device VIEW (self stays put). CPU-preload → per-step GPU copy, bounded GPU memory."""
        import types
        mv = lambda g: {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in g.items()}
        v = types.SimpleNamespace(cid=self.cid, p1=mv(self.p1), p2=mv(self.p2),
                                  pos=self.pos.to(device), pos_sc=self.pos_sc.to(device))
        return v


def usable_complexes(data_dir, ids):
    out = []
    for cid in ids:
        if all(os.path.exists(os.path.join(data_dir, f"{cid}__holo__{p}.npz")) for p in ("p1", "p2")) \
           and os.path.exists(os.path.join(data_dir, f"{cid}__contacts.npz")):
            c = np.load(os.path.join(data_dir, f"{cid}__contacts.npz"))
            if c["pos"].reshape(-1, 2).shape[0] > 0:
                out.append(cid)
    return out
