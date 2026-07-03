"""Core logic for the Milestone-1 pooling feasibility probe.

Builds a granularity-generic "view" of each complex (p1 straight embeddings + coords in the
p1 role, p2 flipped embeddings + coords in the p2 role) and computes descriptor-separation
distances for positives and neg_mix negatives. The SAME code path scores the per-vertex
baseline and the per-atom test — the only thing that changes is which embeddings/coords go in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from masif_graph.metrics.separation import pair_distances
from masif_graph.pairs.construct import (
    sample_hard_negatives,
    sample_within_negatives,
)

# neg_mix defaults (mirror the reference custom_params.py).
NEG_RATIO = 5
NEG_MIX = {"cross": 0.5, "within": 0.3, "hard": 0.2}


@dataclass
class ComplexView:
    """One complex at one granularity, in scoring form."""

    complex_id: str
    s1: np.ndarray  # (n1, 80) p1 straight embeddings
    f2: np.ndarray  # (n2, 80) p2 flipped embeddings
    c1: np.ndarray  # (n1, 3) p1 coords
    c2: np.ndarray  # (n2, 3) p2 coords
    positives: np.ndarray  # (P, 2) index pairs (i into s1/c1, j into f2/c2)
    exp1: np.ndarray | None = None  # (n1,) exposure (atoms only)
    exp2: np.ndarray | None = None  # (n2,) exposure (atoms only)
    # radii for negative bands (granularity-specific)
    hard_radius: float = 4.0
    within_min: float = 5.0


@dataclass
class ScoredComplex:
    complex_id: str
    pos_dists: np.ndarray
    neg_dists: np.ndarray
    neg_kinds: np.ndarray  # parallel to neg_dists: "cross"/"within"/"hard"
    pos_min_exposure: np.ndarray | None  # (P,) min owned-vertex count over the pair (atoms)
    n_pos: int
    n_neg: int


def _counts(n_pos: int):
    n_neg = NEG_RATIO * n_pos
    n_cross = int(round(NEG_MIX["cross"] * n_neg))
    n_within = int(round(NEG_MIX["within"] * n_neg))
    n_hard = n_neg - n_cross - n_within
    return n_cross, n_within, n_hard


def score_complex(view: ComplexView, cross_pool, seed: int) -> ScoredComplex:
    """Compute positive + neg_mix negative distances for one complex view.

    cross_pool: list of (complex_id, f2) tuples for cross-complex negatives (any complex !=
    this one is eligible). seed: per-complex RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    P = len(view.positives)
    pos_dists = pair_distances(view.s1, view.f2, view.positives)

    pos_min_exp = None
    if view.exp1 is not None and view.exp2 is not None and P > 0:
        pos_min_exp = np.minimum(
            view.exp1[view.positives[:, 0]], view.exp2[view.positives[:, 1]]
        )

    if P == 0:
        return ScoredComplex(view.complex_id, pos_dists, np.zeros(0), np.array([], dtype=object),
                             pos_min_exp, 0, 0)

    n_cross, n_within, n_hard = _counts(P)

    # hard: closest non-positive same-complex pairs
    hard = sample_hard_negatives(view.c1, view.c2, view.positives, n_hard, view.hard_radius, rng)
    # top up hard shortfall with within
    n_within_eff = n_within + (n_hard - len(hard))
    within = sample_within_negatives(
        view.c1, view.c2, view.positives, n_within_eff, view.within_min, rng
    )
    within_dists = pair_distances(view.s1, view.f2, within)
    hard_dists = pair_distances(view.s1, view.f2, hard)

    # cross: this complex's p1 entity vs a random OTHER complex's p2 entity
    others = [(cid, f2) for (cid, f2) in cross_pool if cid != view.complex_id and len(f2) > 0]
    cross_dists = np.zeros(0)
    if n_cross > 0 and others and len(view.s1) > 0:
        ii = rng.integers(len(view.s1), size=n_cross)
        pick = rng.integers(len(others), size=n_cross)
        cd = np.empty(n_cross)
        for k in range(n_cross):
            cid, f2 = others[pick[k]]
            j = int(rng.integers(len(f2)))
            diff = view.s1[ii[k]] - f2[j]
            cd[k] = np.sqrt(np.dot(diff, diff))
        cross_dists = cd

    neg_dists = np.concatenate([cross_dists, within_dists, hard_dists])
    neg_kinds = np.array(
        ["cross"] * len(cross_dists) + ["within"] * len(within_dists) + ["hard"] * len(hard_dists),
        dtype=object,
    )
    return ScoredComplex(
        view.complex_id, pos_dists, neg_dists, neg_kinds, pos_min_exp, P, len(neg_dists)
    )


def random_negative_dists(view: ComplexView, seed: int) -> np.ndarray:
    """Scheme-A sanity negatives: positive p1 entity vs a random same-complex p2 entity
    (excluding the true partner). Reproduces the reference 'pos_neg' sanity check."""
    rng = np.random.default_rng(seed + 777)
    P = len(view.positives)
    if P == 0 or len(view.f2) < 2:
        return np.zeros(0)
    i = view.positives[:, 0]
    true_j = view.positives[:, 1]
    rand_j = rng.integers(len(view.f2), size=P)
    # avoid accidentally hitting the true partner
    clash = rand_j == true_j
    rand_j[clash] = (rand_j[clash] + 1) % len(view.f2)
    diff = view.s1[i] - view.f2[rand_j]
    return np.sqrt(np.sum(diff * diff, axis=1))
