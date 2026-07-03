"""Descriptor-separation ROC-AUC (the Phase-1 gating metric).

Matches the reference convention (`nn_models/.../model_data_paper/compute_roc_auc.py`):
score for a candidate pair is 1/L2-distance between the p1 *straight* and p2 *flipped*
descriptors; positives (true contacts) should have small distance, negatives large. AUC is
computed on ytrue=1 (pos) / 0 (neg) with ypred = 1/dist.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

_EPS = 1e-12


def pair_distances(desc_straight_1: np.ndarray, desc_flipped_2: np.ndarray, pairs: np.ndarray):
    """L2 distance ||straight1[i] - flipped2[j]|| for each pair (i, j)."""
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float64)
    diff = desc_straight_1[pairs[:, 0]] - desc_flipped_2[pairs[:, 1]]
    return np.sqrt(np.sum(diff * diff, axis=1))


def separation_auc(pos_dists: np.ndarray, neg_dists: np.ndarray) -> float:
    """ROC-AUC separating positives (small dist) from negatives (large dist), 1/dist score."""
    pos_dists = np.asarray(pos_dists, dtype=np.float64)
    neg_dists = np.asarray(neg_dists, dtype=np.float64)
    if len(pos_dists) == 0 or len(neg_dists) == 0:
        return float("nan")
    ytrue = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
    ypred = 1.0 / (np.concatenate([pos_dists, neg_dists]) + _EPS)
    return float(roc_auc_score(ytrue, ypred))


def shuffled_label_auc(pos_dists: np.ndarray, neg_dists: np.ndarray, rng) -> float:
    """Control: shuffle the pos/neg labels; AUC must collapse to ~0.5 if the metric is sound."""
    pos_dists = np.asarray(pos_dists, dtype=np.float64)
    neg_dists = np.asarray(neg_dists, dtype=np.float64)
    if len(pos_dists) == 0 or len(neg_dists) == 0:
        return float("nan")
    dists = np.concatenate([pos_dists, neg_dists])
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    labels = labels.copy()
    rng.shuffle(labels)
    ypred = 1.0 / (dists + _EPS)
    return float(roc_auc_score(labels, ypred))
