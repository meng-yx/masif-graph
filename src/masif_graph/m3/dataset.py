"""M3 training dataset (atomsurf_h100 env). Joins PASS-1 graph `.npz` + PASS-2 surf `.pt` per chain.

Per complex, provides everything a forward + the losses need, for both states (holo, af3):
  - a reconstructed DiffusionData surface object (operators) per chain, + per-vertex frozen desc
    (straight/flipped) + vertex->surface-atom map + n_surf + chem-graph tensors + surface-atom keys.
  - complementarity contacts (holo surface-atom row pairs p1<->p2).
  - holo<->af3 identity-matched surface-atom row pairs per chain (the invariance targets).

Only complexes with BOTH holo and af3 preprocessed are usable for the invariance loss; holo-only
complexes can still contribute the complementarity loss.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from diffusion_net import DiffusionData


def _load_surf(pt_path, device):
    d = torch.load(pt_path, map_location=device)
    s = DiffusionData()
    s.pos = d["verts"].to(device)
    s.face = d["faces"].t().contiguous().to(device)
    s.mass = d["mass"].to(device); s.L = d["L"].to(device)
    s.evals = d["evals"].to(device); s.evecs = d["evecs"].to(device)
    s.gradX = d["gradX"].to(device); s.gradY = d["gradY"].to(device)
    s.batch = torch.zeros(d["n_vert"], dtype=torch.long, device=device)
    return s


def _load_graph(npz_path, device):
    z = np.load(npz_path)
    return {
        "desc_straight": torch.tensor(z["desc_straight"], device=device),
        "desc_flipped": torch.tensor(z["desc_flipped"], device=device),
        "vertex_surf_idx": torch.tensor(z["vertex_surf_idx"], dtype=torch.long, device=device),
        "n_surf": int(z["n_surf"]),
        "gt": {"node_feat": torch.tensor(z["node_feat"], device=device),
               "cov_edge": torch.tensor(z["cov_edge"], dtype=torch.long, device=device),
               "cov_feat": torch.tensor(z["cov_feat"], device=device),
               "sp_edge": torch.zeros((2, 0), dtype=torch.long, device=device),
               "sp_feat": torch.zeros((0, 16), device=device),
               "surf_idx": torch.tensor(z["surf_idx"], dtype=torch.long, device=device)},
        "keys": [k.decode() if isinstance(k, bytes) else str(k) for k in z["keys"]],
    }


class Chain:
    def __init__(self, data_dir, cid, state, pid, device):
        self.surf = _load_surf(os.path.join(data_dir, f"{cid}__{state}__{pid}.surf.pt"), device)
        g = _load_graph(os.path.join(data_dir, f"{cid}__{state}__{pid}.npz"), device)
        self.__dict__.update(g)
        self.key2row = {k: i for i, k in enumerate(self.keys)}


class ComplexData:
    def __init__(self, data_dir, cid, device):
        self.cid = cid
        self.holo = {p: Chain(data_dir, cid, "holo", p, device) for p in ("p1", "p2")}
        self.af3 = None
        if os.path.exists(os.path.join(data_dir, f"{cid}__af3__p1.npz")) and \
           os.path.exists(os.path.join(data_dir, f"{cid}__af3__p1.surf.pt")):
            try:
                self.af3 = {p: Chain(data_dir, cid, "af3", p, device) for p in ("p1", "p2")}
            except FileNotFoundError:
                self.af3 = None
        c = np.load(os.path.join(data_dir, f"{cid}__contacts.npz"))
        self.contacts = torch.tensor(c["pos"], dtype=torch.long, device=device)  # (P,2) holo rows
        # holo<->af3 identity-matched surface-atom rows per chain (invariance targets)
        self.match = {}
        if self.af3 is not None:
            for p in ("p1", "p2"):
                h, a = self.holo[p], self.af3[p]
                hr, ar = [], []
                for k, i in h.key2row.items():
                    j = a.key2row.get(k)
                    if j is not None:
                        hr.append(i); ar.append(j)
                self.match[p] = (torch.tensor(hr, dtype=torch.long, device=device),
                                 torch.tensor(ar, dtype=torch.long, device=device))


def usable_complexes(data_dir, ids):
    out = []
    for cid in ids:
        if os.path.exists(os.path.join(data_dir, f"{cid}__holo__p1.npz")) and \
           os.path.exists(os.path.join(data_dir, f"{cid}__holo__p1.surf.pt")) and \
           os.path.exists(os.path.join(data_dir, f"{cid}__contacts.npz")):
            out.append(cid)
    return out
