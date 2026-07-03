"""Vertex->atom mapping, surface-atom definition, and per-atom pooling (Phase-1 D1-A).

The surface heavy atom is the fundamental unit of MaSIF-graph. For a preprocessed chain we
build the persisted vertex->atom index (which the reference throws away), define the surface
atoms as those owning >=1 vertex, and pool the per-vertex reference descriptors onto them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SurfaceAtoms:
    """Per-chain surface-atom representation (see docs/01-phase1-design.md §2)."""

    # Indices (into the chain's full heavy-atom array) of atoms owning >=1 vertex.
    atom_idx: np.ndarray  # (n_surf,)
    coord: np.ndarray  # (n_surf, 3) heavy-atom coordinates
    element: np.ndarray  # (n_surf,)
    resid: np.ndarray  # (n_surf,)
    n_owned: np.ndarray  # (n_surf,) number of owned vertices (exposure)
    owned_vertices: list  # list of np.ndarray, owned vertex indices per surface atom
    # Pooled embeddings, straight and flipped, for each pooling operator.
    emb_straight: dict  # op -> (n_surf, 80)
    emb_flipped: dict  # op -> (n_surf, 80)
    normal: np.ndarray | None  # (n_surf, 3) unit mean of owned-vertex normals (or None)
    # Map from full heavy-atom index -> row in this table (-1 if not a surface atom).
    full_to_surf: np.ndarray  # (n_atom_total,)
    # Per-vertex owning-atom index, expressed as a row into this surface-atom table.
    vertex_surf_idx: np.ndarray  # (n_vert,)


def map_vertices_to_atoms(verts: np.ndarray, atom_coords: np.ndarray) -> np.ndarray:
    """Nearest-heavy-atom index a(v) for each surface vertex (KDTree)."""
    tree = cKDTree(atom_coords)
    _, nn = tree.query(verts, k=1)
    return np.asarray(nn, dtype=np.int64)


def _pool(desc: np.ndarray, owned_vertices: list, op: str) -> np.ndarray:
    out = np.empty((len(owned_vertices), desc.shape[1]), dtype=np.float64)
    if op == "mean":
        for i, vs in enumerate(owned_vertices):
            out[i] = desc[vs].mean(axis=0)
    elif op == "max":
        for i, vs in enumerate(owned_vertices):
            out[i] = desc[vs].max(axis=0)
    else:
        raise ValueError(f"unknown pooling op {op!r}")
    return out


def build_surface_atoms(
    verts: np.ndarray,
    atom_coords: np.ndarray,
    atom_element: np.ndarray,
    atom_resid: np.ndarray,
    desc_straight: np.ndarray,
    desc_flipped: np.ndarray,
    ops=("mean", "max"),
    vertex_normals: np.ndarray | None = None,
) -> SurfaceAtoms:
    """Build the per-chain surface-atom table + pooled embeddings.

    Pools the *precomputed* desc_flipped directly (never flips the pooled straight vector):
    the flip is a non-linear transform through the reference net (design §2.2).
    """
    vertex_atom_idx = map_vertices_to_atoms(verts, atom_coords)  # a(v), full-atom index
    n_atom_total = len(atom_coords)

    # Surface atoms = atoms owning >=1 vertex. Group vertices by owning atom.
    order = np.argsort(vertex_atom_idx, kind="stable")
    sorted_atoms = vertex_atom_idx[order]
    uniq_atoms, starts = np.unique(sorted_atoms, return_index=True)
    owned_vertices = np.split(order, starts[1:])  # aligned with uniq_atoms

    full_to_surf = np.full(n_atom_total, -1, dtype=np.int64)
    full_to_surf[uniq_atoms] = np.arange(len(uniq_atoms))
    vertex_surf_idx = full_to_surf[vertex_atom_idx]

    n_owned = np.array([len(vs) for vs in owned_vertices], dtype=np.int64)
    emb_straight = {op: _pool(desc_straight, owned_vertices, op) for op in ops}
    emb_flipped = {op: _pool(desc_flipped, owned_vertices, op) for op in ops}

    normal = None
    if vertex_normals is not None:
        normal = np.zeros((len(uniq_atoms), 3), dtype=np.float64)
        for i, vs in enumerate(owned_vertices):
            m = vertex_normals[vs].mean(axis=0)
            nrm = np.linalg.norm(m)
            normal[i] = m / nrm if nrm > 0 else m

    return SurfaceAtoms(
        atom_idx=uniq_atoms,
        coord=atom_coords[uniq_atoms],
        element=atom_element[uniq_atoms],
        resid=atom_resid[uniq_atoms],
        n_owned=n_owned,
        owned_vertices=owned_vertices,
        emb_straight=emb_straight,
        emb_flipped=emb_flipped,
        normal=normal,
        full_to_surf=full_to_surf,
        vertex_surf_idx=vertex_surf_idx,
    )
