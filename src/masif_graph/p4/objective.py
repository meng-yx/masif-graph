"""Phase-4 Stage-A objective: symmetric bilinear complementarity T + InfoNCE (design §5.1-5.2).

Contacting atoms sit in *complementary* environments, so we score through a learned symmetric bilinear
form  s(X,Y) = zX^T T zY,  T = T^T  (subsumes MaSIF's hardcoded flip trick T=diag(±1); D3-A). Symmetric
T => order-independent (either partner can be the query) and keeps deployment a fast inner-product search
(transform every DB atom once as T z_d, then max-inner-product).

Contrastive loss = InfoNCE with in-complex negatives (the partner's non-contacting surface atoms — the
'medium' tier of the design's escalating negatives) + optional cross-complex bank (easy tier). Symmetrized
over both query directions. Embeddings are L2-normalized so the bilinear score is a bounded learned metric
on the unit sphere and InfoNCE can't collapse (positives must still beat all partner negatives).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Complementarity(nn.Module):
    def __init__(self, d: int, tau_init: float = 0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, d) * (1.0 / math.sqrt(d)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def T(self) -> torch.Tensor:
        return 0.5 * (self.A + self.A.t())

    def score(self, zX: torch.Tensor, zY: torch.Tensor) -> torch.Tensor:
        """(nX,d),(nY,d) -> (nX,nY) bilinear scores zX^T T zY."""
        return zX @ self.T @ zY.t()


def normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, dim=1, eps=1e-8)


def info_nce_complex(z1, z2, pos, comp: Complementarity, bank2=None, bank1=None):
    """Symmetric InfoNCE for one complex.

    z1 (n1,d), z2 (n2,d): normalized per-surface-atom embeddings of the two chains.
    pos (P,2): contact rows (i in chain1, j in chain2).
    bank2/bank1: optional (M,d) cross-complex embeddings appended as extra (easy) negatives.
    Returns scalar loss (mean of both query directions).
    """
    tau = comp.log_tau.exp().clamp(1e-2, 1.0)
    i, j = pos[:, 0], pos[:, 1]

    # direction 1: anchor on chain1, retrieve the true chain2 atom among all chain2 atoms (+bank)
    cand2 = z2 if bank2 is None else torch.cat([z2, bank2], dim=0)
    s1 = comp.score(z1[i], cand2) / tau           # (P, n2[+M])
    loss1 = F.cross_entropy(s1, j)                # target column = true partner row

    # direction 2: anchor on chain2, retrieve the true chain1 atom
    cand1 = z1 if bank1 is None else torch.cat([z1, bank1], dim=0)
    s2 = comp.score(z2[j], cand1) / tau
    loss2 = F.cross_entropy(s2, i)

    return 0.5 * (loss1 + loss2)
