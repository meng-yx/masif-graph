"""Contacting-pair (positive) and neg_mix (negative) construction, at vertex or atom level.

Identical logic is applied at both granularities so the per-vertex baseline and per-atom
metric are apples-to-apples (design §4, handoff §9). A "pair" is (i, j): i indexes a p1
entity (scored with its *straight* descriptor), j indexes a p2 entity (scored *flipped*).

Positives:
  * vertex          : each p1 vertex whose nearest p2 vertex is < `pos_cutoff` (1.0 Å).
  * atom (primary)  : owner surface-atoms of the vertex positives, deduplicated (§4.2).
  * atom (secondary): heavy-atom pairs with inter-atom distance < `atom_contact_cutoff`.

Negatives (neg_mix, split cross/within/hard). Within/hard are same-complex here; cross is
sampled at the batch level (needs other complexes). Positives are always excluded explicitly.
  * hard  : spatially closest non-positive cross-chain pairs (just outside the contact) —
            geometrically near-contact but unlabeled.
  * within: random non-positive cross-chain pairs beyond `within_min` (clearly non-contact).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def vertex_contacts(
    verts1: np.ndarray,
    verts2: np.ndarray,
    pos_cutoff: float = 1.0,
    sc1: np.ndarray | None = None,
    sc_band: tuple = (0.5, 1.0),
):
    """Reference vertex-contact positives: p1 vertex -> nearest p2 vertex < cutoff.

    If `sc1` (per-p1-vertex shape complementarity) is given, additionally require it to lie in
    `sc_band` (0.5, 1.0) — the reference's shape-complementarity gate, which is what reproduces
    the ~0.98 descriptor-separation baseline. Returns (pairs[k,2] int, dists[k]).
    """
    tree2 = cKDTree(verts2)
    d, j = tree2.query(verts1, k=1)
    contact = d < pos_cutoff
    if sc1 is not None:
        contact &= (sc1 > sc_band[0]) & (sc1 < sc_band[1])
    i = np.nonzero(contact)[0]
    pairs = np.column_stack([i, j[i]]).astype(np.int64)
    return pairs, d[i]


def atom_positives_from_vertex_contacts(vpairs, vertex_surf_idx1, vertex_surf_idx2):
    """Map contacting vertex pairs to owner surface-atom pairs; dedup. Returns pairs[k,2]."""
    if len(vpairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a = vertex_surf_idx1[vpairs[:, 0]]
    b = vertex_surf_idx2[vpairs[:, 1]]
    ab = np.column_stack([a, b])
    return np.unique(ab, axis=0).astype(np.int64)


def atom_contacts_direct(coords1, coords2, cutoff: float = 4.0):
    """Secondary positive definition: heavy-atom pairs within `cutoff` Å. Returns pairs[k,2]."""
    tree2 = cKDTree(coords2)
    pairs = []
    for i, c in enumerate(coords1):
        for j in tree2.query_ball_point(c, cutoff):
            pairs.append((i, j))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)


def _pos_key_set(positives):
    return set(map(tuple, positives.tolist())) if len(positives) else set()


def sample_hard_negatives(coords1, coords2, positives, n_hard, radius, rng):
    """Closest non-positive cross-chain pairs within `radius` (geometrically hard)."""
    if n_hard <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    pos = _pos_key_set(positives)
    tree2 = cKDTree(coords2)
    cand_i, cand_j, cand_d = [], [], []
    # limit per-anchor candidates to keep this bounded on large chains
    for i, c in enumerate(coords1):
        js = tree2.query_ball_point(c, radius)
        for j in js:
            if (i, j) in pos:
                continue
            cand_i.append(i); cand_j.append(j)
            cand_d.append(float(np.linalg.norm(c - coords2[j])))
    if not cand_i:
        return np.zeros((0, 2), dtype=np.int64)
    cand = np.column_stack([cand_i, cand_j]).astype(np.int64)
    d = np.asarray(cand_d)
    order = np.argsort(d, kind="stable")  # closest first
    take = order[: n_hard]
    # if fewer candidates than requested, that's fine (caller tops up elsewhere)
    return cand[take]


def sample_within_negatives(coords1, coords2, positives, n_within, within_min, rng, max_tries=50):
    """Random non-positive cross-chain pairs with inter-entity distance > within_min."""
    if n_within <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    pos = _pos_key_set(positives)
    n1, n2 = len(coords1), len(coords2)
    out = set()
    tries = 0
    target = n_within
    while len(out) < target and tries < max_tries * max(target, 1):
        i = int(rng.integers(n1)); j = int(rng.integers(n2))
        tries += 1
        if (i, j) in pos or (i, j) in out:
            continue
        if np.linalg.norm(coords1[i] - coords2[j]) > within_min:
            out.add((i, j))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(out), dtype=np.int64)
