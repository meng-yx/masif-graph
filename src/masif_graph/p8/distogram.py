"""Stage R / R3 — the inter-chain distogram that REPLACES real-space pose prediction (docs/26 §3).

The explicit rigid pose is dropped (user decision, 2026-08-12). Stage 2 now predicts, for a
candidate atom pair, a distribution over inter-atom distances including an explicit "no contact"
bin. Reasons, in the order they matter:

1. **No alignment, so no alignment bias.** Every superposition target carries a systematic offset
   against experimental structures. Stage A hit this concretely: fitting on atom centres co-locates
   contacting atoms that are really 3.79 A apart, interpenetrating the chains (1.06 A closest
   approach vs 2.55 A native) and imposing a 1.9 A iRMSD floor. A distance is supervised directly
   against the crystal, in no frame at all.
2. **~5 orders of magnitude more supervision.** One complex yields one pose label but ~10^5-10^6
   pairwise distances. This is why AlphaFold predicts a distogram rather than coordinates.
3. **It subsumes contact detection** — a head that can output ">20 A" is a contact classifier, which
   is exactly the axis Stage 1 was never trained on.
4. **It is the north star's quantity**: the *spread* of the predicted distance distribution is
   "how much mismatch is tolerable here", per pair, in angstroms (D8-9's sigma, made concrete).

Geometric consistency is not discarded — it is measured as a SCALAR (`consistency_residual`) and
handed to Stage 3, so the strong rigid-body prior still contributes without committing to a pose.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bin edges in angstroms. Fine where contact chemistry lives (2-6 A), coarse beyond, plus a final
# open-ended "no contact" bin. Edges are UPPER bounds; the last bin is everything above 20 A.
BIN_EDGES = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0], float)
N_BINS = len(BIN_EDGES) + 1                      # 14, the last one = "no contact"
NO_CONTACT_BIN = N_BINS - 1


def bin_distances(d: torch.Tensor) -> torch.Tensor:
    """Distances (A) -> bin indices. Bucketize is right-continuous on the given boundaries."""
    edges = torch.as_tensor(BIN_EDGES, dtype=d.dtype, device=d.device)
    return torch.bucketize(d, edges)


def bin_centres(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Representative distance per bin; the open last bin is given a nominal 25 A."""
    e = np.concatenate([[1.5], BIN_EDGES])
    c = 0.5 * (e[:-1] + e[1:])
    return torch.as_tensor(np.concatenate([c, [25.0]]), dtype=dtype, device=device)


class DistogramHead(nn.Module):
    """Pair features -> distance-bin logits.

    Input per pair is [z_i, z_j, z_i * z_j, s_ij] — symmetric in the elementwise product and the
    score, and the concatenation is symmetrised by averaging the two orderings, so the head cannot
    learn a spurious dependence on which chain is called 'first'.
    """

    def __init__(self, d_out: int, hidden: int = 128, n_bins: int = N_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_out + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_bins))
        self.n_bins = n_bins

    def _feat(self, za, zb, s):
        return torch.cat([za, zb, za * zb, s[:, None]], dim=1)

    def forward(self, zi, zj, s):
        return 0.5 * (self.net(self._feat(zi, zj, s)) + self.net(self._feat(zj, zi, s)))


def n_negatives(n_pos, neg_per_pos=4, max_neg=4096, min_neg=256):
    """How many random pairs to draw, PROPORTIONAL to the number of true contacts.

    A fixed count (the first attempt used 2048) makes the no-contact class ~10x the contact class,
    and a 1-epoch GPU probe showed the head simply learning the prior: distogram accuracy 0.778 with
    **contact recall 0.032**. Proportional sampling plus the class weighting in `distogram_loss`
    keeps the two aggregate contributions comparable.
    """
    return int(min(max_neg, max(min_neg, neg_per_pos * max(n_pos, 1))))


def sample_pairs(n1, n2, pos, n_neg, device, seed=0):
    """All true contacts + `n_neg` random non-contact pairs.

    Random pairs are overwhelmingly far apart, which is the point: the head must learn "no contact"
    as a positive prediction, not as an absence. Their true distance is computed, not assumed, so a
    random pair that happens to be in contact is labelled correctly.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(0, n1, (n_neg,), generator=g).to(device)
    b = torch.randint(0, n2, (n_neg,), generator=g).to(device)
    if pos.shape[0]:
        p = torch.as_tensor(pos, dtype=torch.long, device=device)
        a = torch.cat([p[:, 0], a])
        b = torch.cat([p[:, 1], b])
    return a, b


def distogram_loss(head, comp, z1, z2, c1, c2, pos, n_neg=None, seed=0, neg_per_pos=4):
    """Cross-entropy of predicted bins against the true (binned) inter-atom distances.

    Class-balanced: the "no contact" bin is down-weighted so that contact and non-contact bins
    contribute comparably in aggregate. Without this the head learns the prior (measured: accuracy
    0.778, contact recall 0.032).
    """
    n_pos = int(pos.shape[0]) if pos is not None else 0
    if n_neg is None:
        n_neg = n_negatives(n_pos, neg_per_pos=neg_per_pos)
    a, b = sample_pairs(z1.shape[0], z2.shape[0], pos, n_neg, z1.device, seed)
    d = (c1[a] - c2[b]).norm(dim=1)
    s = (z1[a] @ comp.T * z2[b]).sum(1)
    logits = head(z1[a], z2[b], s)
    tgt = bin_distances(d)
    w = torch.ones(head.n_bins, device=logits.device, dtype=logits.dtype)
    n_far = int((tgt == NO_CONTACT_BIN).sum())
    n_near = int(len(tgt)) - n_far
    if n_far > 0 and n_near > 0:
        w[NO_CONTACT_BIN] = max(n_near / n_far, 1e-3)
    loss = F.cross_entropy(logits, tgt, weight=w)
    with torch.no_grad():
        pred = logits.argmax(1)
        ctr = bin_centres(d.device, d.dtype)
        stats = {"disto_acc": float((pred == tgt).float().mean()),
                 "disto_mae_binned": float((ctr[pred] - ctr[tgt]).abs().mean()),
                 "contact_recall": float((pred[tgt < NO_CONTACT_BIN] < NO_CONTACT_BIN).float().mean())
                 if int((tgt < NO_CONTACT_BIN).sum()) else float("nan"),
                 "n_pairs": int(len(d))}
    return loss, stats


@torch.no_grad()
def consistency_residual(head, comp, z1, z2, c1, c2, pairs, weights=None):
    """R3b — how well is the PREDICTED distance set explained by a single rigid motion?

    A rigid pose is 6 DOF explaining hundreds of contacts, so mutual consistency is a strong signal
    for real-vs-spurious interfaces; dropping the pose must not mean dropping that. This fits the
    rigid transform that best reproduces the predicted distances (weighted Kabsch on the target
    points implied by them) and returns the residual as a scalar for Stage 3.

    Deliberately NOT the co-location fit that biased Stage A: the target separation is the predicted
    distance, not zero.
    """
    a, b = pairs
    if len(a) < 4:
        return float("nan")
    s = (z1[a] @ comp.T * z2[b]).sum(1)
    p = F.softmax(head(z1[a], z2[b], s), dim=1)
    d_pred = (p * bin_centres(z1.device, p.dtype)[None, :]).sum(1)          # expected distance
    src, tgt = c2[b], c1[a]
    w = torch.ones_like(d_pred) if weights is None else weights
    # Move each source point to the predicted distance from its target along the current direction,
    # then Kabsch onto those. One pass is enough for a consistency statistic.
    v = src - tgt
    n = v.norm(dim=1, keepdim=True).clamp_min(1e-6)
    goal = tgt + v / n * d_pred[:, None]
    sw = w.sum().clamp_min(1e-6)
    mu_s = (src * w[:, None]).sum(0) / sw
    mu_g = (goal * w[:, None]).sum(0) / sw
    H = ((src - mu_s) * w[:, None]).t() @ (goal - mu_g)
    U, _, Vt = torch.linalg.svd(H)
    dsign = torch.sign(torch.det(Vt.t() @ U.t()))
    D = torch.diag(torch.tensor([1.0, 1.0, float(dsign)], device=H.device, dtype=H.dtype))
    R = Vt.t() @ D @ U.t()
    fit = (R @ (src - mu_s).t()).t() + mu_g
    return float(((fit - goal).norm(dim=1) * w).sum() / sw)
