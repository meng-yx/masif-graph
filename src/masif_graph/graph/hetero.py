"""Phase-4 heterogeneous surface graph builder (atom nodes + surface-vertex nodes).

ONE graph per chain, three edge types, ALL geometry expressed as SE(3)-invariant scalars
(design `docs/08-phase4-design.md` §4). This is the M0 deliverable; its correctness gate is the
rotation-invariance test at the bottom of this module (`rotation_invariance_report`).

Nodes
-----
* **atom nodes** — every heavy atom (surface + buried). Features (all conformation/pose invariant):
  Phase-2 base(10) [element one-hot, backbone flag, aromatic, degree, is_surface] + flex_depth(1)
  + element-chem(3) [electronegativity, valence, covalent radius]. Coords are kept for *edge
  geometry only* and are never a node feature.
* **surface-vertex nodes** — MSMS vertices carrying MaSIF's hand-crafted input channels
  `[si, hbond, charge, hphob]` (shape index from the precompute patch self-row; the other three
  from the `.ply`; all pre-normalized ~[-1,1], all invariant). Normals + coords kept for edge
  geometry only.

Edges (features are distances + cos-angles → invariant to any joint rotation+translation)
-----
* **atom–atom, covalent only** — bond-order one-hot(4) + sidechain-rotatable flag(1). Reused from
  the Phase-2 builder. NO through-space atom edges (design §4: they inject pose-sensitivity).
* **vertex–vertex, mesh adjacency** — from the `.ply` triangle faces (topology → coord-independent
  connectivity). Features: edge length + cos(normal_i, normal_j).
* **vertex–atom** — each vertex to the heavy atoms within a radius (rotation-invariant ball), capped
  at k nearest. Features: distance + cos(normal_v, unit(atom - vertex)). Buried atoms fall outside
  every vertex ball → no vertex edges, exactly as the design intends.

Everything geometric is stored as RAW scalars (dist, cos); RBF expansion happens at tensor-build
time (`p4/dataset.py`), so the invariance test operates directly on the scalars the model consumes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.graph.build import build_atom_graph
from masif_graph.m3.chem_graph import element_chem_features
from masif_graph.io.reference import PRECOMP_DIR, SURFACE_DIR


# ---------------------------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------------------------
@dataclass
class HeteroSurfaceGraph:
    """Per-chain heterogeneous graph (atom nodes + vertex nodes; 3 invariant edge types)."""

    # --- atom nodes ---
    n_atom: int
    atom_feat: np.ndarray          # (n_atom, F_atom) invariant node features
    atom_coords: np.ndarray        # (n_atom, 3) geometry ONLY (never a feature)
    is_surface_atom: np.ndarray    # (n_atom,) bool
    atom_surf_row: np.ndarray      # (n_atom,) row into the surface-atom table, or -1
    n_surf: int                    # number of surface atoms (readout targets)
    # covalent atom-atom edges (directed, both ways)
    aa_edge: np.ndarray            # (2, Eaa)
    aa_order: np.ndarray           # (Eaa, 4) bond-order one-hot [single, double, aromatic, other]
    aa_rot: np.ndarray             # (Eaa,) sidechain-rotatable flag {0,1}

    # --- vertex nodes ---
    n_vert: int
    vert_feat: np.ndarray          # (n_vert, 4) [si, hbond, charge, hphob] invariant
    vert_coords: np.ndarray        # (n_vert, 3) geometry ONLY
    vert_normals: np.ndarray       # (n_vert, 3) geometry ONLY (rotates with coords)
    vertex_surf_idx: np.ndarray    # (n_vert,) owning surface-atom row (nearest heavy atom's row)
    # vertex-vertex mesh edges (directed, both ways)
    vv_edge: np.ndarray            # (2, Evv)
    vv_dist: np.ndarray            # (Evv,) edge length (Å)
    vv_cos: np.ndarray             # (Evv,) cos angle between endpoint normals
    # vertex-atom edges (directed: rows are [vertex_idx, atom_idx] and [atom_idx, vertex_idx])
    va_v: np.ndarray               # (Eva,) vertex endpoint of each undirected v-a edge
    va_a: np.ndarray               # (Eva,) atom endpoint
    va_dist: np.ndarray            # (Eva,) distance (Å)
    va_cos: np.ndarray             # (Eva,) cos angle between vertex normal and unit(atom - vertex)

    @property
    def atom_feat_dim(self) -> int:
        return self.atom_feat.shape[1]


# ---------------------------------------------------------------------------------------------
# Per-vertex surface inputs (si from precompute patch self-row; charge/hbond/hphob + normals + faces
# from the .ply). Standalone .ply reader (avoids importing m3.surface_encoder, which needs diffusion_net).
# ---------------------------------------------------------------------------------------------
def load_ply_geometry(pdb_id: str, chain_ids: str):
    """Read verts, faces and normals from a reference MaSIF `.ply` (geometry only).

    Returns (verts[V,3], faces[F,3] int, normals[V,3] unit). Per-vertex chemistry comes from the
    precompute `input_feat` (normalized) via `load_vertex_feats`, not the `.ply` (raw)."""
    from plyfile import PlyData

    ply_path = os.path.join(SURFACE_DIR, f"{pdb_id}_{chain_ids}.ply")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    verts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    normals = np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float64)
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)  # (F, 3)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    normals = normals / nrm
    return verts, faces, normals


def load_vertex_feats(complex_id: str, pid: str, n_vert_expected: int) -> np.ndarray:
    """Per-vertex MaSIF input channels [si, hbond, charge, hphob] from the precompute patch self-row.

    input_feat is (V,200,5) with channel order [si, ddc, hbond, charge, hphob]; the self row (patch
    idx 0, rho≈0) holds the vertex's own channels, all pre-normalized to ~[-1,1]. ddc≡0 at the self
    row (curvature relative to center) → dropped; the mesh normals recover local curvature. Aligned
    to vertex order (same convention as X/Y/Z and the descriptors). Returns (V,4)."""
    p = os.path.join(PRECOMP_DIR, complex_id, f"{pid}_input_feat.npy")
    feat = np.load(p, mmap_mode="r")
    if feat.shape[0] != n_vert_expected:
        raise ValueError(f"{complex_id} {pid}: input_feat V={feat.shape[0]} != {n_vert_expected}")
    return np.asarray(feat[:, 0, :][:, [0, 2, 3, 4]], dtype=np.float32)  # [si, hbond, charge, hphob]


# ---------------------------------------------------------------------------------------------
# Edge geometry helpers (pure functions of coords/normals; the invariance lives here)
# ---------------------------------------------------------------------------------------------
def _mesh_edges_from_faces(faces: np.ndarray, n_vert: int):
    """Undirected unique vertex-vertex edges from triangle faces. Connectivity is topology-only."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)  # canonical (min,max)
    e = np.unique(e, axis=0)
    e = e[(e[:, 0] >= 0) & (e[:, 1] < n_vert)]
    return e  # (Evv, 2)


def _vertex_atom_edges(verts, atom_coords, radius: float, k_max: int):
    """Each vertex → heavy atoms within `radius`, capped to `k_max` nearest.

    The radius ball is rotation-invariant; ties are broken by (distance, atom-index) which is also
    rotation-stable, so the connectivity is invariant. Returns (va_v, va_a) index arrays."""
    tree = cKDTree(atom_coords)
    va_v, va_a = [], []
    for vi, c in enumerate(verts):
        idx = tree.query_ball_point(c, radius)
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        d = np.linalg.norm(atom_coords[idx] - c, axis=1)
        # deterministic order: distance, then atom index (rotation-stable tie-break)
        order = np.lexsort((idx, d))
        keep = idx[order][:k_max]
        va_v.extend([vi] * len(keep))
        va_a.extend(keep.tolist())
    return np.asarray(va_v, dtype=np.int64), np.asarray(va_a, dtype=np.int64)


def _subsample_vertices(verts, max_vert, seed=0):
    """Uniform vertex subsample to <= max_vert (seeded). Returns kept indices (sorted)."""
    n = len(verts)
    if max_vert is None or n <= max_vert:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n, size=max_vert, replace=False))
    return keep


# ---------------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------------
def build_hetero_graph(
    chain,
    surf,
    pdb_path: str,
    *,
    va_radius: float = 5.0,
    va_kmax: int = 8,
    max_vert: int | None = None,
    subsample_seed: int = 0,
    _override_verts: np.ndarray | None = None,
    _override_normals: np.ndarray | None = None,
    _override_atom_coords: np.ndarray | None = None,
) -> HeteroSurfaceGraph:
    """Build the per-chain HeteroSurfaceGraph.

    chain : io.reference.Chain      (atom table + verts + per-vertex chem)
    surf  : surface.atoms.SurfaceAtoms (full_to_surf, vertex_surf_idx)
    pdb_path : chain PDB (for biotite covalent topology, via the Phase-2 builder)

    The `_override_*` args inject rotated/translated geometry for the invariance test; they replace
    the coords/normals used for *edge geometry* while node features stay as-is (features never depend
    on pose), so the test can assert connectivity + edge scalars are invariant.
    """
    # ---- atom side: reuse Phase-2 covalent builder (drop its spatial edges per design §4) ----
    ag = build_atom_graph(chain, surf, pdb_path)
    chem = element_chem_features(chain.atom_element)  # (n_atom, 3) invariant element-chem
    # atom node features: Phase-2 base(10) + flex_depth(1, normalized) + element-chem(3)
    flex = (np.clip(ag.flex_depth, 0, 8) / 8.0).astype(np.float32)[:, None]
    atom_feat = np.concatenate([ag.node_feat, flex, chem], axis=1).astype(np.float32)
    atom_coords = ag.coords.astype(np.float64)
    if _override_atom_coords is not None:
        atom_coords = _override_atom_coords.astype(np.float64)

    # ---- vertex side: normalized MaSIF channels (precompute) + mesh + normals (.ply) ----
    verts_ply, faces, normals = load_ply_geometry(chain.pdb_id, chain.chain_ids)
    # align .ply vertex order to the precompute/descriptor order used everywhere else
    if verts_ply.shape[0] != chain.n_vert or not np.allclose(verts_ply, chain.verts, atol=1e-3):
        raise ValueError(
            f"{chain.complex_id} {chain.pid}: .ply vertex order != precompute verts "
            f"(V_ply={verts_ply.shape[0]} V_pc={chain.n_vert})"
        )
    vert_feat_full = load_vertex_feats(chain.complex_id, chain.pid, chain.n_vert)  # (V,4) normalized

    verts = chain.verts.astype(np.float64)
    if _override_verts is not None:
        verts = _override_verts.astype(np.float64)
    if _override_normals is not None:
        normals = _override_normals.astype(np.float64)

    # ---- optional vertex cap/coarsen ----
    keep = _subsample_vertices(verts, max_vert, seed=subsample_seed)
    subsampled = len(keep) != len(verts)
    if subsampled:
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        verts = verts[keep]
        normals = normals[keep]
        vert_feat_full = vert_feat_full[keep]
        vsurf = surf.vertex_surf_idx[keep]
        # remap faces to kept vertices (drop faces touching a dropped vertex), then edges
        fmask = np.all(np.isin(faces, keep), axis=1)
        faces_k = remap[faces[fmask]]
        vv = _mesh_edges_from_faces(faces_k, len(verts))
    else:
        vsurf = surf.vertex_surf_idx.copy()
        vv = _mesh_edges_from_faces(faces, len(verts))

    # ---- vertex-vertex mesh edge features (dist + cos of endpoint normals) ----
    if len(vv):
        di = verts[vv[:, 0]] - verts[vv[:, 1]]
        vv_dist = np.linalg.norm(di, axis=1)
        vv_cos = np.sum(normals[vv[:, 0]] * normals[vv[:, 1]], axis=1)
    else:
        vv_dist = np.zeros(0); vv_cos = np.zeros(0)

    # ---- vertex-atom edges + features (dist + cos(normal_v, unit(atom - vertex))) ----
    va_v, va_a = _vertex_atom_edges(verts, atom_coords, radius=va_radius, k_max=va_kmax)
    if len(va_v):
        vec = atom_coords[va_a] - verts[va_v]
        va_dist = np.linalg.norm(vec, axis=1)
        unit = vec / np.clip(va_dist, 1e-9, None)[:, None]
        va_cos = np.sum(normals[va_v] * unit, axis=1)
    else:
        va_dist = np.zeros(0); va_cos = np.zeros(0)

    # ---- directed covalent edges already both-ways from build_atom_graph ----
    return HeteroSurfaceGraph(
        n_atom=ag.n_node,
        atom_feat=atom_feat,
        atom_coords=atom_coords,
        is_surface_atom=ag.is_surface.copy(),
        atom_surf_row=ag.surf_row.copy(),
        n_surf=int(ag.is_surface.sum()),
        aa_edge=ag.cov_edge.copy(),
        aa_order=ag.cov_order.copy(),
        aa_rot=ag.cov_rot.copy(),
        n_vert=len(verts),
        vert_feat=vert_feat_full,
        vert_coords=verts,
        vert_normals=normals,
        vertex_surf_idx=vsurf.astype(np.int64),
        vv_edge=vv.T.astype(np.int64) if len(vv) else np.zeros((2, 0), np.int64),
        vv_dist=vv_dist.astype(np.float32),
        vv_cos=vv_cos.astype(np.float32),
        va_v=va_v, va_a=va_a,
        va_dist=va_dist.astype(np.float32),
        va_cos=va_cos.astype(np.float32),
    )


# ---------------------------------------------------------------------------------------------
# Rotation-invariance sanity test — THE M0 hard gate
# ---------------------------------------------------------------------------------------------
def random_se3(seed=0):
    """Random rotation R (proper, det=+1) and translation t."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = rng.normal(size=3) * 10.0
    return Q, t


def _edge_key_set(src, dst):
    return set(zip(src.tolist(), dst.tolist()))


def rotation_invariance_report(chain, surf, pdb_path, *, seed=0, tol=1e-4, **build_kw):
    """Build the graph on original vs randomly SE(3)-transformed geometry; report invariance.

    Returns a dict of checks (all must be True / max-diff < tol for the M0 gate to PASS)."""
    g0 = build_hetero_graph(chain, surf, pdb_path, **build_kw)

    R, t = random_se3(seed)
    verts_r = chain.verts @ R.T + t
    # rebuild needs full-resolution rotated inputs; load normals fresh and rotate (unit-preserving)
    _, _, normals0 = load_ply_geometry(chain.pdb_id, chain.chain_ids)
    normals_r = normals0 @ R.T
    atom_coords0 = build_atom_graph(chain, surf, pdb_path).coords
    atom_r = atom_coords0 @ R.T + t

    g1 = build_hetero_graph(
        chain, surf, pdb_path,
        _override_verts=verts_r, _override_normals=normals_r, _override_atom_coords=atom_r,
        **build_kw,
    )

    rep = {}
    rep["n_atom_match"] = g0.n_atom == g1.n_atom
    rep["n_vert_match"] = g0.n_vert == g1.n_vert
    # node features must be byte-identical (they never depend on pose)
    rep["atom_feat_identical"] = np.array_equal(g0.atom_feat, g1.atom_feat)
    rep["vert_feat_identical"] = np.array_equal(g0.vert_feat, g1.vert_feat)
    # connectivity must be identical
    rep["aa_edges_identical"] = _edge_key_set(g0.aa_edge[0], g0.aa_edge[1]) == \
        _edge_key_set(g1.aa_edge[0], g1.aa_edge[1])
    rep["vv_edges_identical"] = _edge_key_set(g0.vv_edge[0], g0.vv_edge[1]) == \
        _edge_key_set(g1.vv_edge[0], g1.vv_edge[1])
    rep["va_edges_identical"] = _edge_key_set(g0.va_v, g0.va_a) == _edge_key_set(g1.va_v, g1.va_a)
    # edge scalar features must match (order is deterministic → compare directly)
    def _maxdiff(a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).max()) if len(a) else 0.0
    rep["vv_dist_maxdiff"] = _maxdiff(g0.vv_dist, g1.vv_dist)
    rep["vv_cos_maxdiff"] = _maxdiff(g0.vv_cos, g1.vv_cos)
    rep["va_dist_maxdiff"] = _maxdiff(g0.va_dist, g1.va_dist)
    rep["va_cos_maxdiff"] = _maxdiff(g0.va_cos, g1.va_cos)
    rep["aa_order_identical"] = np.array_equal(g0.aa_order, g1.aa_order)
    rep["aa_rot_identical"] = np.array_equal(g0.aa_rot, g1.aa_rot)

    bool_checks = [k for k, v in rep.items() if isinstance(v, (bool, np.bool_))]
    diff_checks = [k for k in rep if k.endswith("maxdiff")]
    rep["PASS"] = all(rep[k] for k in bool_checks) and all(rep[k] < tol for k in diff_checks)
    return rep, g0
