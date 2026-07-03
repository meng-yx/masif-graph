"""Distance-distribution overlap: plots + quantitative summary (handoff §0.3).

Reads the pos/neg distance arrays saved by run_m1 and produces, per positive-definition, a
panel of pos-vs-neg histograms (vertex | atom_mean | atom_max) plus a quantitative overlap
table (overlap coefficient, Cohen's d, medians).
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def overlap_coefficient(pos, neg, bins=60):
    """Histogram overlap coefficient (0 = disjoint, 1 = identical) on a shared support."""
    lo = min(pos.min(), neg.min())
    hi = max(pos.max(), neg.max())
    edges = np.linspace(lo, hi, bins + 1)
    ph, _ = np.histogram(pos, bins=edges, density=True)
    nh, _ = np.histogram(neg, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(ph, nh)) * width)


def cohens_d(pos, neg):
    n1, n2 = len(pos), len(neg)
    s = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2))
    return float((neg.mean() - pos.mean()) / s) if s > 0 else float("nan")


def summarize(pos, neg):
    return {
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "pos_median": float(np.median(pos)), "neg_median": float(np.median(neg)),
        "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
        "overlap_coefficient": overlap_coefficient(pos, neg),
        "cohens_d": cohens_d(pos, neg),
        # fraction of negatives whose distance is below the positive median (lower = better sep)
        "frac_neg_below_pos_median": float((neg < np.median(pos)).mean()),
    }


def make_plots(m1_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    overlap = {}
    for pdef in ("sc", "unfiltered", "atom_direct"):
        grans = ["atom_mean", "atom_max"] if pdef == "atom_direct" else ["vertex", "atom_mean", "atom_max"]
        present = [g for g in grans if os.path.exists(os.path.join(m1_dir, f"{pdef}__{g}__pos.npy"))]
        if not present:
            continue
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
        overlap[pdef] = {}
        for k, g in enumerate(present):
            pos = np.load(os.path.join(m1_dir, f"{pdef}__{g}__pos.npy"))
            neg = np.load(os.path.join(m1_dir, f"{pdef}__{g}__neg.npy"))
            if len(pos) == 0 or len(neg) == 0:
                continue
            overlap[pdef][g] = summarize(pos, neg)
            ax = axes[0][k]
            rng = (0, max(np.percentile(pos, 99), np.percentile(neg, 99)))
            ax.hist(pos, bins=50, range=rng, density=True, alpha=0.6, label=f"pos (n={len(pos)})", color="tab:green")
            ax.hist(neg, bins=50, range=rng, density=True, alpha=0.6, label=f"neg (n={len(neg)})", color="tab:red")
            ax.set_title(f"{g}\nOVL={overlap[pdef][g]['overlap_coefficient']:.3f} d={overlap[pdef][g]['cohens_d']:.2f}")
            ax.set_xlabel("straight-vs-flipped descriptor L2 distance")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        fig.suptitle(f"Descriptor-distance overlap (positive def = {pdef})")
        fig.tight_layout()
        out = os.path.join(fig_dir, f"overlap_{pdef}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)
    with open(os.path.join(fig_dir, "overlap_summary.json"), "w") as f:
        json.dump(overlap, f, indent=2)
    print("wrote", os.path.join(fig_dir, "overlap_summary.json"))
    return overlap


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_dir", default="logs/m1")
    ap.add_argument("--fig_dir", default="docs/figures")
    args = ap.parse_args()
    make_plots(args.m1_dir, args.fig_dir)
