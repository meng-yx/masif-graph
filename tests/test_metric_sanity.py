"""Hand-checkable sanity tests for the descriptor-separation metric (guardrail §2.3).

Run: pytest tests/test_metric_sanity.py
"""
import numpy as np

from masif_graph.metrics.separation import (
    pair_distances,
    separation_auc,
    shuffled_label_auc,
)


def test_perfect_separation():
    # positives all closer than all negatives -> AUC == 1.0
    assert separation_auc(np.array([0.1, 0.2, 0.3]), np.array([5.0, 6.0, 7.0])) == 1.0


def test_inverted_separation():
    # positives all FARTHER than negatives -> AUC == 0.0 (1/dist ranks them last)
    assert separation_auc(np.array([5.0, 6.0]), np.array([0.1, 0.2])) == 0.0


def test_chance_when_identical():
    # identical distributions -> AUC ~ 0.5
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(separation_auc(d, d) - 0.5) < 1e-9


def test_hand_computed_auc():
    # pos dists {1, 3}, neg dists {2, 4}. scores = 1/dist.
    # ranking by score desc: 1/1=1.0(P), 1/2=0.5(N), 1/3=0.33(P), 1/4=0.25(N)
    # AUC = P(random pos ranked above random neg). Pairs (P,N):
    #   (1,2):pos>neg ok; (1,4):ok; (3,2):pos_score .33<.5 -> neg ranked above -> miss;
    #   (3,4):.33>.25 ok. 3/4 => 0.75
    assert abs(separation_auc(np.array([1.0, 3.0]), np.array([2.0, 4.0])) - 0.75) < 1e-9


def test_shuffled_collapses():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.1, 1.0, 500)
    neg = rng.uniform(5.0, 6.0, 2500)
    assert separation_auc(pos, neg) > 0.99  # clearly separable
    # shuffling labels must collapse to ~0.5
    aucs = [shuffled_label_auc(pos, neg, np.random.default_rng(s)) for s in range(20)]
    assert abs(np.mean(aucs) - 0.5) < 0.05


def test_pair_distances():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0]])
    pairs = np.array([[0, 0], [1, 1]])
    d = pair_distances(a, b, pairs)
    assert np.allclose(d, [0.0, np.sqrt(2.0)])
