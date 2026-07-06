"""Heterogeneous atom-graph builder (Phase 2, design docs/03 §3).

One graph per chain, molecule-agnostic (wired for proteins). Nodes = heavy atoms
(surface + sub-surface). Typed edges:
  * covalent  — from biotite residue-template connectivity; carries bond order (one-hot)
                and a *sidechain-rotatable* flag (the "how atoms can move" signal).
  * spatial   — radius graph over heavy-atom coords, carries an RBF-expanded distance.

The covalent topology + node chemistry are **invariant to a fixed-backbone repack** (bonds
don't change, only sidechain coords). Spatial edges + coords *do* move under repack. That
contrast is exactly what the M2 ablation dissects.

The node table is aligned 1:1 with the Phase-1 `io.reference` heavy-atom table (same PDB,
same altloc handling — verified 100% overlap), so `surf.full_to_surf` maps a node to its
surface-atom row (or -1). Descriptors are attached and readout happens at surface nodes only;
the descriptor itself is NOT message-passed (design decision D-P2.3), keeping the
straight/flipped complementarity entirely in the surface channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

import biotite.structure.io.pdb as _pdb
import biotite.structure as _struc

_ELEMENTS = ["C", "N", "O", "S", "P"]  # + "other"
_BACKBONE = {"N", "CA", "C", "O"}
_AROMATIC_BONDTYPES = {5, 6, 7, 8}  # biotite AROMATIC_SINGLE/DOUBLE/TRIPLE/AROMATIC
_HYDROGEN = {"H", "D"}
_SOLVENT = {"HOH", "WAT", "DOD", "TIP", "SOL"}


@dataclass
class AtomGraph:
    """Per-chain heterogeneous atom graph (all edge types present; the model masks per cell)."""

    n_node: int
    coords: np.ndarray            # (n, 3)
    node_feat: np.ndarray         # (n, F_base) invariant base features (see build_node_features)
    flex_depth: np.ndarray        # (n,) # sidechain-rotatable bonds to backbone (rotatability signal)
    is_surface: np.ndarray        # (n,) bool
    surf_row: np.ndarray          # (n,) row into the surface-atom table, or -1
    # covalent edges (directed, both ways)
    cov_edge: np.ndarray          # (2, Ec)
    cov_order: np.ndarray         # (Ec, 4) one-hot [single, double, aromatic, other]
    cov_rot: np.ndarray           # (Ec,) float {0,1} sidechain-rotatable flag
    # spatial edges (directed, both ways)
    sp_edge: np.ndarray           # (2, Es)
    sp_dist: np.ndarray           # (Es,) interatomic distance (Å)

    @property
    def n_surface(self) -> int:
        return int(self.is_surface.sum())


def parse_heavy_biotite(pdb_path: str):
    """Parse heavy atoms of a PDB with biotite; return AtomArray + a (chain,resseq,name)->idx map.

    Altloc handling: keep the first altloc per (residue, atom) so the count matches the
    Phase-1 custom parser (which keeps altloc ' '/'A')."""
    f = _pdb.PDBFile.read(pdb_path)
    arr = f.get_structure(model=1, altloc="first")
    mask = ~np.isin(arr.element, list(_HYDROGEN)) & ~np.isin(arr.res_name, list(_SOLVENT))
    heavy = arr[mask]
    return heavy


def _bond_graph(n, bond_arr):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j, t in bond_arr:
        G.add_edge(int(i), int(j), order=int(t))
    return G


def _rotatable_flags(heavy, G, bond_arr):
    """General rotatable = single, acyclic (bridge), non-terminal, non-amide.
    Sidechain-rotatable additionally excludes bonds internal to the backbone (N-CA, CA-C, C-N),
    so the flag encodes *sidechain* flexibility — the mover under a fixed-backbone repack."""
    bridges = set(map(frozenset, nx.bridges(G)))
    deg = dict(G.degree())
    elem = heavy.element
    name = heavy.atom_name

    def is_carbonyl_C(idx):
        for nb in G.neighbors(idx):
            if elem[nb] == "O" and G[idx][nb]["order"] == 2:
                return True
        return False

    sc_rot = {}
    for i, j, t in bond_arr:
        i, j = int(i), int(j)
        single = int(t) == 1
        acyclic = frozenset((i, j)) in bridges
        nonterminal = deg[i] >= 2 and deg[j] >= 2
        amide = False
        if {elem[i], elem[j]} == {"C", "N"}:
            c_idx = i if elem[i] == "C" else j
            amide = is_carbonyl_C(c_idx)
        rot = single and acyclic and nonterminal and not amide
        both_backbone = (name[i] in _BACKBONE) and (name[j] in _BACKBONE)
        sc_rot[frozenset((i, j))] = bool(rot and not both_backbone)
    return sc_rot


def _flex_depth(n, G, sc_rot, heavy):
    """0-1 BFS: min # sidechain-rotatable bonds from each atom to any backbone atom."""
    INF = 10**9
    depth = np.full(n, INF, dtype=np.int64)
    from collections import deque
    dq = deque()
    for k in range(n):
        if heavy.atom_name[k] in _BACKBONE:
            depth[k] = 0
            dq.append(k)
    # 0-weight edges first (deque appendleft), 1-weight edges appended right.
    while dq:
        u = dq.popleft()
        for v in G.neighbors(u):
            w = 1 if sc_rot.get(frozenset((u, v)), False) else 0
            if depth[u] + w < depth[v]:
                depth[v] = depth[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    depth[depth == INF] = 8  # isolated / non-protein: cap
    return depth


def build_node_features(heavy, G, is_surface):
    n = heavy.array_length()
    elem = heavy.element
    name = heavy.atom_name
    deg = dict(G.degree())
    # aromatic membership: incident aromatic bond
    aromatic = np.zeros(n, dtype=bool)
    for i, j, t in heavy.bonds.as_array():
        if int(t) in _AROMATIC_BONDTYPES:
            aromatic[int(i)] = True
            aromatic[int(j)] = True
    feat = np.zeros((n, len(_ELEMENTS) + 1 + 4), dtype=np.float32)
    for k in range(n):
        e = elem[k]
        if e in _ELEMENTS:
            feat[k, _ELEMENTS.index(e)] = 1.0
        else:
            feat[k, len(_ELEMENTS)] = 1.0  # "other"
    off = len(_ELEMENTS) + 1
    feat[:, off + 0] = np.array([1.0 if name[k] in _BACKBONE else 0.0 for k in range(n)])
    feat[:, off + 1] = aromatic.astype(np.float32)
    feat[:, off + 2] = np.array([min(deg.get(k, 0), 6) / 6.0 for k in range(n)], dtype=np.float32)
    feat[:, off + 3] = is_surface.astype(np.float32)
    return feat  # F_base = 10


def _rbf(dist, n_rbf=16, d_max=5.0):
    centers = np.linspace(0.0, d_max, n_rbf)
    gamma = 1.0 / ((centers[1] - centers[0]) ** 2 + 1e-9)
    return np.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2).astype(np.float32)


def build_atom_graph(
    chain,
    surf,
    pdb_path: str,
    spatial_cutoff: float = 5.0,
    max_spatial_deg: int = 24,
):
    """Build the per-chain AtomGraph, aligned to `chain` / `surf` (io.reference atom table).

    chain: io.reference.Chain (atom_coords/element/resid/name).
    surf:  surface.atoms.SurfaceAtoms (full_to_surf maps atom row -> surface row or -1).
    """
    from scipy.spatial import cKDTree

    n = chain.n_atom
    coords = chain.atom_coords.astype(np.float64)

    # --- covalent topology from biotite, mapped onto the io.reference atom rows by key ---
    heavy = parse_heavy_biotite(pdb_path)
    heavy.bonds = _struc.connect_via_residue_names(heavy)
    bio_key = {}
    for k in range(heavy.array_length()):
        bio_key.setdefault(
            (str(heavy.chain_id[k]), str(heavy.res_id[k]), str(heavy.atom_name[k])), k
        )
    # ref key -> ref row
    ref_row = {}
    for r in range(n):
        ch, seq, _rn = chain.atom_resid[r].split(":")
        ref_row[(ch, seq, str(chain.atom_name[r]))] = r
    bio2ref = {k: ref_row[key] for key, k in bio_key.items() if key in ref_row}

    bond_arr = heavy.bonds.as_array()
    G = _bond_graph(heavy.array_length(), bond_arr)
    sc_rot = _rotatable_flags(heavy, G, bond_arr)
    flex_bio = _flex_depth(heavy.array_length(), G, sc_rot, heavy)

    # map flex depth to ref rows (default 8 for unmapped)
    flex_depth = np.full(n, 8, dtype=np.int64)
    for bk, rr in bio2ref.items():
        flex_depth[rr] = flex_bio[bk]

    is_surface = surf.full_to_surf >= 0
    node_feat_bio = build_node_features(heavy, G, np.array(
        [is_surface[bio2ref[k]] if k in bio2ref else False for k in range(heavy.array_length())]))
    # reindex node features to ref rows
    F = node_feat_bio.shape[1]
    node_feat = np.zeros((n, F), dtype=np.float32)
    for bk, rr in bio2ref.items():
        node_feat[rr] = node_feat_bio[bk]
    # ensure is_surface column is authoritative from surf
    node_feat[:, -1] = is_surface.astype(np.float32)

    # covalent edges in ref indexing (both directions)
    ci, cj, corder, crot = [], [], [], []
    order_map = {1: 0, 2: 1, 5: 2, 6: 2, 7: 2, 8: 2}  # single/double/aromatic; else other(3)
    for i, j, t in bond_arr:
        i, j = int(i), int(j)
        if i not in bio2ref or j not in bio2ref:
            continue
        ri, rj = bio2ref[i], bio2ref[j]
        oh = order_map.get(int(t), 3)
        rot = 1.0 if sc_rot.get(frozenset((i, j)), False) else 0.0
        for a, b in ((ri, rj), (rj, ri)):
            ci.append(a); cj.append(b); corder.append(oh); crot.append(rot)
    cov_edge = np.array([ci, cj], dtype=np.int64) if ci else np.zeros((2, 0), dtype=np.int64)
    cov_order = np.zeros((len(corder), 4), dtype=np.float32)
    if corder:
        cov_order[np.arange(len(corder)), np.array(corder)] = 1.0
    cov_rot = np.array(crot, dtype=np.float32)

    # spatial edges: radius graph, exclude covalent-bonded pairs, cap degree
    covset = set(zip(ci, cj))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(spatial_cutoff, output_type="ndarray")
    si, sj, sd = [], [], []
    if len(pairs):
        d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        # cap per-node degree by keeping nearest neighbours
        from collections import defaultdict
        buckets = defaultdict(list)
        for (a, b), dist in zip(pairs, d):
            buckets[int(a)].append((dist, int(b)))
            buckets[int(b)].append((dist, int(a)))
        seen = set()
        for a, lst in buckets.items():
            lst.sort()
            for dist, b in lst[:max_spatial_deg]:
                if (a, b) in covset:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                si.append(a); sj.append(b); sd.append(dist)
    sp_edge = np.array([si, sj], dtype=np.int64) if si else np.zeros((2, 0), dtype=np.int64)
    sp_dist = np.array(sd, dtype=np.float32)

    return AtomGraph(
        n_node=n, coords=coords, node_feat=node_feat, flex_depth=flex_depth,
        is_surface=is_surface, surf_row=surf.full_to_surf.copy(),
        cov_edge=cov_edge, cov_order=cov_order, cov_rot=cov_rot,
        sp_edge=sp_edge, sp_dist=sp_dist,
    )


def rbf_expand(dist, n_rbf=16, d_max=5.0):
    return _rbf(np.asarray(dist, dtype=np.float64), n_rbf=n_rbf, d_max=d_max)
