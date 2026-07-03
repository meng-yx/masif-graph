"""Milestone-2 global, atom-level alignment prototype (handoff §5, design §3.4).

Replaces the reference per-patch RANSAC loop with a single *global* fit over all target x binder
surface-atom correspondences whose pooled surface embeddings indicate complementarity, gated by
reference MaSIF-site interface propensity (design D8). No learned scorer (that is Phase 2).
Sanity metrics: interface-RMSD of the recovered binder pose vs native + fraction of native
contacts recovered, starting from a randomized binder pose.

Divergence from the handoff's Open3D recipe (documented, with evidence, in the progress log):
contacting atom *centers* are ~4-5 A apart (van der Waals), not coincident, which destabilizes
Open3D's point-to-point `registration_ransac_based_on_correspondence` (3-point exact fit) — it
returns ~19.6 A iRMSD even on ground-truth correspondences, whereas a direct least-squares Kabsch
over the same pairs gives ~2.4 A. We therefore use a custom RANSAC-Kabsch (refit least-squares on
all inliers, inlier threshold above the vdW offset) and a point-to-POINT Kabsch-ICP refine
(point-to-plane would let the binder slide tangentially along the interface).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.io.reference import load_complex
from masif_graph.surface.atoms import build_surface_atoms


def _rigid(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def kabsch(P, Q):
    """Least-squares rigid transform mapping points P -> Q (Umeyama without scale)."""
    cP = P.mean(0)
    cQ = Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return _rigid(R, cQ - R @ cP)


def apply_T(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def random_pose(seed, max_trans=20.0):
    """Deterministic random rigid transform (to perturb the binder away from native)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return _rigid(q, rng.uniform(-max_trans, max_trans, size=3))


def ransac_kabsch(src, tgt, corr, thr=6.0, iters=5000, seed=0):
    """RANSAC over correspondences with a Kabsch model refit on all inliers.

    src/tgt: (n,3) point sets; corr: (m,2) (src_idx, tgt_idx). Returns (T, n_inliers).
    """
    rng = np.random.default_rng(seed)
    S = src[corr[:, 0]]
    Tg = tgt[corr[:, 1]]
    if len(corr) < 3:
        return np.eye(4), 0
    best_mask, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(corr), 3, replace=False)
        T = kabsch(S[idx], Tg[idx])
        res = np.linalg.norm(apply_T(T, S) - Tg, axis=1)
        mask = res < thr
        n = int(mask.sum())
        if n > best_n:
            best_n, best_mask = n, mask
    if best_mask is None or best_mask.sum() < 3:
        return np.eye(4), 0
    return kabsch(S[best_mask], Tg[best_mask]), best_n


def kabsch_icp(src, tgt, T_init, max_dist=5.0, iters=30):
    """Point-to-point ICP by repeated Kabsch on nearest-neighbour correspondences within max_dist."""
    tree = cKDTree(tgt)
    T = T_init.copy()
    for _ in range(iters):
        now = apply_T(T, src)
        d, nn = tree.query(now, k=1)
        m = d < max_dist
        if m.sum() < 3:
            break
        T_new = kabsch(src[m], tgt[nn[m]])
        if np.allclose(T_new, T, atol=1e-6):
            T = T_new
            break
        T = T_new
    return T


def per_atom_iface(sa, iface):
    return np.array([iface[vs].mean() for vs in sa.owned_vertices])


def native_contacts(coord1, coord2, cutoff=5.0):
    """Native interface surface-atom pairs (p1_idx, p2_idx) within cutoff. Ground truth."""
    tree2 = cKDTree(coord2)
    pairs = []
    for i, c in enumerate(coord1):
        for j in tree2.query_ball_point(c, cutoff):
            pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _interface_atom_idx(pairs_col):
    return np.unique(pairs_col)


@dataclass
class AlignResult:
    complex_id: str
    n_iface_p1: int
    n_iface_p2: int
    n_corres: int
    corres_precision: float  # fraction of correspondences that are true native contacts (eval only)
    n_native_contacts: int
    irmsd_start: float
    irmsd_ransac: float
    irmsd_icp: float
    contact_recovery_icp: float
    success: bool


def align_one(cid, op="mean", iface_thr=0.5, desc_cutoff=1.6, ransac_thr=6.0,
              icp_dist=5.0, contact_cutoff=5.0, seed=0):
    p1, p2 = load_complex(cid)
    sa1 = build_surface_atoms(p1.verts, p1.atom_coords, p1.atom_element, p1.atom_resid,
                              p1.desc_straight, p1.desc_flipped)
    sa2 = build_surface_atoms(p2.verts, p2.atom_coords, p2.atom_element, p2.atom_resid,
                              p2.desc_straight, p2.desc_flipped)
    c1 = sa1.coord
    c2 = sa2.coord
    if1 = per_atom_iface(sa1, p1.iface)
    if2 = per_atom_iface(sa2, p2.iface)

    nat = native_contacts(c1, c2, contact_cutoff)
    if len(nat) < 10:
        return AlignResult(cid, 0, 0, 0, 0.0, len(nat), np.nan, np.nan, np.nan, np.nan, False)
    p2_iface = _interface_atom_idx(nat[:, 1])
    natset = set(map(tuple, np.column_stack([nat[:, 1], nat[:, 0]]).tolist()))

    T_rand = random_pose(seed)
    c2_start = apply_T(T_rand, c2)

    def irmsd(now):
        return float(np.sqrt(np.mean(np.sum((now[p2_iface] - c2[p2_iface]) ** 2, axis=1))))

    irmsd_start = irmsd(c2_start)

    # interface-gated embedding correspondences (complementary: p1 straight vs p2 flipped)
    g1 = np.where(if1 > iface_thr)[0]
    g2 = np.where(if2 > iface_thr)[0]
    if len(g1) < 4 or len(g2) < 4:
        return AlignResult(cid, len(g1), len(g2), 0, 0.0, len(nat), irmsd_start,
                           np.nan, np.nan, np.nan, False)
    e1 = sa1.emb_straight[op][g1]
    e2 = sa2.emb_flipped[op]
    tree1 = cKDTree(e1)
    corr = []
    for j in g2:
        d, idx = tree1.query(e2[j], k=3)
        idx = np.atleast_1d(idx); d = np.atleast_1d(d)
        for k in range(len(idx)):
            if d[k] < desc_cutoff:
                corr.append((int(j), int(g1[idx[k]])))
    corr = np.asarray(corr, dtype=np.int64).reshape(-1, 2)
    if len(corr) < 4:
        return AlignResult(cid, len(g1), len(g2), len(corr), 0.0, len(nat), irmsd_start,
                           np.nan, np.nan, np.nan, False)
    precision = float(np.mean([tuple(x) in natset for x in corr]))

    # RANSAC-Kabsch (src = p2 in start frame, tgt = p1). This is the reported pose.
    T_ransac, _ = ransac_kabsch(c2_start, c1, corr, thr=ransac_thr, seed=seed)
    c2_ransac = apply_T(T_ransac, c2_start)
    irmsd_ransac = irmsd(c2_ransac)

    # Point-to-point Kabsch-ICP refine, kept as a DIAGNOSTIC only: on atom centers it overpacks
    # (co-locating nearest centers pulls the binder ~vdW into the target and can slide it), so it
    # degrades rather than improves the RANSAC-Kabsch pose. Reported to document that finding.
    T_icp = kabsch_icp(c2_start, c1, T_ransac, max_dist=icp_dist)
    irmsd_icp = irmsd(apply_T(T_icp, c2_start))

    # fraction of native contacts recovered from the RANSAC-Kabsch pose (same atom pairs within cutoff)
    d_after = np.linalg.norm(c1[nat[:, 0]] - c2_ransac[nat[:, 1]], axis=1)
    contact_recovery = float((d_after < contact_cutoff).mean())

    return AlignResult(
        cid, len(g1), len(g2), len(corr), precision, len(nat),
        irmsd_start, irmsd_ransac, irmsd_icp, contact_recovery, True,
    )


def run_m2(complex_ids, seed=0, op="mean"):
    """Run the M2 prototype over several complexes. Returns list of result dicts."""
    out = []
    for cid in complex_ids:
        try:
            r = align_one(cid, op=op, seed=seed)
            out.append(r.__dict__)
        except Exception as e:  # noqa: BLE001
            out.append({"complex_id": cid, "success": False, "error": str(e)})
    return out
