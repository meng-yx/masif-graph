"""Dataset glue: build per-chain / per-complex records (holo + repacked), convert graphs to
GNN tensors, and map holo-defined positive pairs across sidechain states by atom identity.

A "complex" here is identified by its holo id (e.g. 3TDM_A_B). Its repacked twin lives under
id {PDBID}RP_{C1}_{C2}. Positives (sc-filtered vertex contacts -> owner atoms) are defined on the
HOLO backbone and mapped to a state's surface-atom rows by (chain,resseq,atom_name) identity.
For the differential holo->repack metric we evaluate the SAME positive pairs in both states —
the intersection that is a surface atom in both — so the degradation is apples-to-apples.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from masif_graph.io.reference import load_complex, PDB_DIR, complex_is_available
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts
from masif_graph.graph.build import build_atom_graph, rbf_expand


def repack_id(holo_id: str) -> str:
    pdb, c1, c2 = holo_id.split("_")
    return f"{pdb}RP_{c1}_{c2}"


def _atom_keys(chain, surf):
    """(chain,resseq,name) key per surface atom, for cross-state identity mapping."""
    keys = []
    for r in surf.atom_idx:
        ch, seq, _rn = chain.atom_resid[r].split(":")
        keys.append((ch, seq, str(chain.atom_name[r])))
    return keys


def graph_to_tensors(g, use_rotatable: bool, device="cpu"):
    """Convert an AtomGraph to GNN input tensors (with the ablation's node/edge features)."""
    n = g.n_node
    # node features: base(10) + flex_depth slot (zeroed unless use_rotatable)
    node_feat = np.zeros((n, g.node_feat.shape[1] + 1), dtype=np.float32)
    node_feat[:, : g.node_feat.shape[1]] = g.node_feat
    if use_rotatable:
        node_feat[:, -1] = np.clip(g.flex_depth, 0, 8) / 8.0
    # covalent edge feat: bond-order one-hot(4) + rotatable flag(1, zeroed unless use_rotatable)
    ce = g.cov_edge.shape[1]
    cov_feat = np.zeros((ce, 5), dtype=np.float32)
    if ce > 0:
        cov_feat[:, :4] = g.cov_order
        if use_rotatable:
            cov_feat[:, 4] = g.cov_rot
    # spatial edge feat: RBF
    se = g.sp_edge.shape[1]
    sp_feat = rbf_expand(g.sp_dist) if se > 0 else np.zeros((0, 16), dtype=np.float32)
    surf_idx = np.nonzero(g.is_surface)[0]
    # order surf_idx to match the surface-atom-table row order (surf_row gives row per node)
    order = np.argsort(g.surf_row[surf_idx])
    surf_idx = surf_idx[order]
    return {
        "node_feat": torch.tensor(node_feat, device=device),
        "cov_edge": torch.tensor(g.cov_edge, dtype=torch.long, device=device),
        "cov_feat": torch.tensor(cov_feat, device=device),
        "sp_edge": torch.tensor(g.sp_edge, dtype=torch.long, device=device),
        "sp_feat": torch.tensor(sp_feat, device=device),
        "surf_idx": torch.tensor(surf_idx, dtype=torch.long, device=device),
    }


@dataclass
class ChainRec:
    cid: str
    pid: str
    state: str                      # "holo" | "repack"
    n_surf: int
    desc_straight: np.ndarray       # (n_surf, 80) mean-pooled
    desc_flipped: np.ndarray        # (n_surf, 80)
    coord: np.ndarray               # (n_surf, 3)
    keys: list                      # (chain,resseq,name) per surface row
    key2row: dict                   # key -> surface row
    graph: object                   # AtomGraph (tensors built lazily per ablation)


@dataclass
class ComplexRec:
    holo_id: str
    holo: dict                      # {"p1": ChainRec, "p2": ChainRec}
    repack: dict | None             # same or None if repack unavailable
    holo_pos: np.ndarray            # (P,2) positive pairs in HOLO surface rows
    inter_pos_holo: np.ndarray      # positives present as surface atoms in BOTH states (holo rows)
    inter_pos_repack: np.ndarray | None  # same pairs mapped to repack rows


def _load_state(holo_id: str, state_id: str, state: str) -> dict | None:
    if not complex_is_available(state_id):
        return None
    p1, p2 = load_complex(state_id)
    recs = {}
    for pid, ch in (("p1", p1), ("p2", p2)):
        surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                                   ch.desc_straight, ch.desc_flipped, ops=("mean",))
        pdb_path = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
        graph = build_atom_graph(ch, surf, pdb_path)
        keys = _atom_keys(ch, surf)
        recs[pid] = ChainRec(
            cid=holo_id, pid=pid, state=state, n_surf=surf.coord.shape[0],
            desc_straight=surf.emb_straight["mean"], desc_flipped=surf.emb_flipped["mean"],
            coord=surf.coord, keys=keys, key2row={k: i for i, k in enumerate(keys)},
            graph=graph,
        )
    return recs


def build_complex_record(holo_id: str, sc_band=(0.5, 1.0)) -> ComplexRec | None:
    """Load holo (+ repacked twin if present), compute holo positives, and the intersection
    positive set valid in both states."""
    if not complex_is_available(holo_id):
        return None
    holo = _load_state(holo_id, holo_id, "holo")
    if holo is None:
        return None

    # holo positives: sc-filtered vertex contacts -> owner surface-atom pairs
    p1c, p2c = load_complex(holo_id)
    surf1 = build_surface_atoms(p1c.verts, p1c.atom_coords, p1c.atom_element, p1c.atom_resid,
                                p1c.desc_straight, p1c.desc_flipped, ops=("mean",))
    surf2 = build_surface_atoms(p2c.verts, p2c.atom_coords, p2c.atom_element, p2c.atom_resid,
                                p2c.desc_straight, p2c.desc_flipped, ops=("mean",))
    vpairs, _ = vertex_contacts(p1c.verts, p2c.verts, pos_cutoff=1.0, sc1=p1c.sc, sc_band=sc_band)
    holo_pos = atom_positives_from_vertex_contacts(vpairs, surf1.vertex_surf_idx, surf2.vertex_surf_idx)

    repack = _load_state(holo_id, repack_id(holo_id), "repack")

    inter_holo, inter_rp = [], []
    if repack is not None and len(holo_pos) > 0:
        rp1, rp2 = repack["p1"], repack["p2"]
        h1, h2 = holo["p1"], holo["p2"]
        for i, j in holo_pos:
            ki, kj = h1.keys[i], h2.keys[j]
            if ki in rp1.key2row and kj in rp2.key2row:
                inter_holo.append((i, j))
                inter_rp.append((rp1.key2row[ki], rp2.key2row[kj]))
    inter_pos_holo = np.array(inter_holo, dtype=np.int64) if inter_holo else np.zeros((0, 2), np.int64)
    inter_pos_repack = np.array(inter_rp, dtype=np.int64) if inter_rp else None

    return ComplexRec(
        holo_id=holo_id, holo=holo, repack=repack, holo_pos=holo_pos,
        inter_pos_holo=inter_pos_holo, inter_pos_repack=inter_pos_repack,
    )
