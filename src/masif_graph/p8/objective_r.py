"""Stage R / R2 — the redesigned atom-level objective (docs/26 §2).

What the old loss did (`p4.objective.info_nce_complex`, used in Stage A and kept in Stage B at
`--w-atom 0.5`):

    for each contact row (i, j):
        s = (z1[i]^T T z2) / tau           # over ALL ~850 partner atoms
        loss = CrossEntropy(s, target = j) # ONE hard label

Two things that breaks, both measured in Stage A (docs/25 §4.3):

1. **It demands an identity match that is probably not identifiable.** Given a true interface atom,
   the trained model puts the true partner at median rank 109 of 854, and its top pick sits a median
   19.4 A away. Local surface chemistry does not single out one atom among its neighbours — and it
   does not need to: everything downstream tolerates a 3-5 A error. `soft_target_infonce` therefore
   asks for a REGION, with a target that decays with real distance from the true partner.

2. **It never asks whether an atom is at an interface at all.** The anchor is always `z1[pos[:, 0]]`,
   a true contacting atom. Gating experiments measured that missing axis as worth 7x correspondence
   precision (query gated) to 18-24x (both sides gated). `dustbin` adds an explicit "no contact"
   column and trains non-interface queries against it.

As a side effect (1) also fixes multi-positive contamination: an interface atom has a mean of 1.8
true partners, and single-label cross-entropy was treating the other 0.8 as negatives.

Together these are SuperGlue's formulation for the same problem shape — match points across two
views, with unmatched points allowed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DustbinScore(nn.Module):
    """Bilinear score with a learnable "no partner" column (and row).

    The dustbin is a single learnable logit, broadcast: a query atom that matches nothing should
    score higher against the dustbin than against any real partner atom. This is what calibrates
    interface against non-interface, which the old objective left entirely unsupervised.
    """

    def __init__(self, comp, bin_init: float = 0.0):
        super().__init__()
        self.comp = comp
        self.bin_score = nn.Parameter(torch.tensor(float(bin_init)))

    @property
    def tau(self):
        return self.comp.log_tau.exp().clamp(1e-2, 1.0)

    def logits(self, zq: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        """(nq,d),(nc,d) -> (nq, nc+1); the last column is the dustbin."""
        s = self.comp.score(zq, zc) / self.tau
        b = self.bin_score.expand(s.shape[0], 1) / self.tau
        return torch.cat([s, b], dim=1)


def soft_targets(coord_c: torch.Tensor, true_idx: torch.Tensor, sigma: float,
                 n_cand: int) -> torch.Tensor:
    """Distance-decayed target distribution over candidates (+ a zero dustbin column).

    coord_c   (nc,3) candidate coordinates
    true_idx  (P,)   index of ONE true partner per query row
    Returns   (P, nc+1) rows summing to 1.

    target_k ~ exp(-||x_k - x_true||^2 / 2 sigma^2). With sigma ~3-5 A this says "land within a few
    angstroms of the right place", which is identifiable and is what the distogram and any later
    pose fit actually need.
    """
    d = torch.cdist(coord_c[true_idx], coord_c)                 # (P, nc)
    w = torch.exp(-(d ** 2) / (2.0 * sigma * sigma))
    w = torch.cat([w, torch.zeros(w.shape[0], 1, device=w.device, dtype=w.dtype)], dim=1)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(1).mean()


def soft_target_infonce(z1, z2, pos, c1, c2, scorer: DustbinScore, sigma: float = 4.0,
                        n_neg_query: int = 0, seed: int = 0):
    """Symmetric soft-target InfoNCE with an optional non-interface (dustbin) query term.

    z1,z2  (n1,d),(n2,d) normalized embeddings   c1,c2 (n1,3),(n2,3) coordinates
    pos    (P,2) contact rows (i in chain1, j in chain2)
    n_neg_query: how many NON-interface query atoms per chain to train against the dustbin
                 (0 disables the term). These are the atoms the old objective never saw.
    """
    if pos.shape[0] == 0:
        return z1.sum() * 0.0, {}
    i, j = pos[:, 0], pos[:, 1]
    l1 = scorer.logits(z1[i], z2)
    l2 = scorer.logits(z2[j], z1)
    loss = 0.5 * (_soft_ce(l1, soft_targets(c2, j, sigma, z2.shape[0]))
                  + _soft_ce(l2, soft_targets(c1, i, sigma, z1.shape[0])))
    stats = {"n_pos": int(pos.shape[0])}

    if n_neg_query > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        db = 0.0
        n_used = 0
        for z, other, iface in ((z1, z2, i), (z2, z1, j)):
            mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
            mask[iface] = False
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            k = min(n_neg_query, idx.numel())
            sel = idx[torch.randperm(idx.numel(), generator=g).to(idx.device)[:k]]
            lg = scorer.logits(z[sel], other)
            tgt = torch.zeros(lg.shape[0], lg.shape[1], device=lg.device, dtype=lg.dtype)
            tgt[:, -1] = 1.0                     # correct answer is "no partner"
            db = db + _soft_ce(lg, tgt)
            n_used += 1
        if n_used:
            loss = loss + (db / n_used)
            stats["n_neg_query"] = n_neg_query
    return loss, stats


@torch.no_grad()
def top1_spatial_error(z1, z2, pos, c2, scorer: DustbinScore):
    """R4 PRIMARY METRIC: distance from the top-1 predicted partner to the nearest true partner.

    Computed for queries that ARE true interface atoms — the deployment condition, where the query
    patch is specified. Stage-A baseline on the current encoder: median 19.4 A. Gate: < 5 A.
    Returns a 1-D tensor of per-query distances (A).

    Fully vectorised: this runs every eval, and a python loop over ~500 queries per complex costs
    more in CUDA launch overhead than the forward pass.
    """
    if pos.shape[0] == 0:
        return torch.zeros(0, device=z1.device)
    q_unique, inv = torch.unique(pos[:, 0], return_inverse=True)     # (Q,), (P,)
    s = scorer.comp.score(z1[q_unique], z2)                          # no dustbin: forced choice
    best = s.argmax(1)                                               # (Q,) predicted partner atom
    # distance from each query's PREDICTION to each of ITS true partners, then min per query
    d = (c2[best[inv]] - c2[pos[:, 1]]).norm(dim=1)                  # (P,)
    out = torch.full((q_unique.shape[0],), float("inf"), device=z1.device, dtype=d.dtype)
    out.scatter_reduce_(0, inv, d, reduce="amin", include_self=True)
    return out
