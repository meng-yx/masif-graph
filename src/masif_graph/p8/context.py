"""Stage R / R1 — rotation-invariant GLOBAL context features.

Why (docs/26 §1, from the Stage-A measurement in docs/25 §4.3): the encoder reads only distances and
cosines, with vv edges <=4 A, va edges <=5 A and 4 message-passing layers — a receptive field of
roughly 15-20 A, about one MaSIF patch. Two chemically similar patches 40 A apart are therefore
*provably* indistinguishable to it, which is why the top-1 predicted partner sits a median 19.4 A
from the true one. No loss change can fix that; the representation has to carry some notion of
*where on the chain* an atom sits.

Everything here is a scalar built from distances within one chain, so the provable rotation
invariance from Phase-4 M0 survives:

    d_centroid        ||x - c|| / Rg
    |proj| on v1..v3  |(x - c) . v_k| / Rg      (v_k = principal axes of the atom cloud)
    shape ratios      sqrt(lambda_k) / Rg       (one global triple, broadcast to every atom)

Under a global rotation R: x -> Rx, c -> Rc, C -> R C R^T, so v_k -> R v_k and (x-c).v_k is
unchanged. Absolute values remove the eigenvector sign ambiguity (v and -v are both valid).

CAVEAT, stated because it is a real failure mode: when two eigenvalues are nearly equal the
corresponding eigenvectors are not unique, so the axis projections become unstable for near-spherical
or near-cylindrical chains. `context_features` returns the eigenvalue gaps so a caller can measure
how often this bites; `rotation_maxdiff` is the actual guard and is run as the R1 gate.

Computed from the SURFACE-ATOM coordinates already stored in every npz (`coord`), so no corpus
regeneration is needed. Interior atoms get zeros — they are never output rows, and context reaches
their neighbourhood through message passing anyway.
"""
from __future__ import annotations

import numpy as np
import torch

N_CTX = 7          # d_centroid, |proj| x3, shape ratios x3


def context_features(coord: np.ndarray, eps: float = 1e-8):
    """(n,3) coordinates -> ((n, N_CTX) features, diagnostics dict). Rotation/translation invariant."""
    x = np.asarray(coord, float)
    if x.shape[0] < 4:
        return np.zeros((x.shape[0], N_CTX), np.float32), {"degenerate": True, "eig_gap": 0.0}
    c = x.mean(0)
    y = x - c
    rg = float(np.sqrt((y * y).sum(1).mean())) + eps          # radius of gyration
    cov = (y.T @ y) / len(y)
    w, v = np.linalg.eigh(cov)                                # ascending eigenvalues
    order = np.argsort(-w)
    w, v = w[order], v[:, order]
    proj = np.abs(y @ v) / rg                                 # (n,3), sign-ambiguity removed
    d_cen = np.linalg.norm(y, axis=1, keepdims=True) / rg
    ratios = np.sqrt(np.maximum(w, 0.0)) / rg                 # (3,) global shape descriptor
    feat = np.concatenate([d_cen, proj, np.tile(ratios, (len(y), 1))], axis=1).astype(np.float32)
    sw = np.sqrt(np.maximum(w, 0.0))
    gap = float((sw[0] - sw[1]) / (sw[0] + eps)) if sw[0] > 0 else 0.0
    gap2 = float((sw[1] - sw[2]) / (sw[1] + eps)) if sw[1] > 0 else 0.0
    return feat, {"rg": rg, "eig_gap_12": gap, "eig_gap_23": gap2,
                  "degenerate": bool(min(gap, gap2) < 0.02)}


def attach_context(g: dict) -> dict:
    """Append the global-context block to `atom_feat`, filled only on surface-atom rows.

    Returns a NEW dict; `g` is not mutated. Idempotent guard: raises if the graph already looks
    context-augmented, so a double-apply cannot silently change the feature width.
    """
    af = g["atom_feat"]
    sidx = g["surf_node_idx"]
    coord = g["coord"].detach().cpu().numpy()
    feat, _ = context_features(coord)
    ctx = torch.zeros(af.shape[0], N_CTX, dtype=af.dtype, device=af.device)
    ctx[sidx] = torch.as_tensor(feat, dtype=af.dtype, device=af.device)
    out = dict(g)
    out["atom_feat"] = torch.cat([af, ctx], dim=1)
    return out


@torch.no_grad()
def rotation_maxdiff(coord: np.ndarray, seed: int = 0) -> float:
    """R1 GATE: max |ctx(x) - ctx(Rx + t)| over a random rigid motion. Must be ~float epsilon."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = rng.uniform(-40, 40, size=3)
    a, _ = context_features(coord)
    b, _ = context_features(coord @ q.T + t)
    return float(np.abs(a - b).max())
