"""Milestone-1 pooling feasibility probe — the Phase-1 gate.

Compares descriptor-separation ROC-AUC at per-vertex (baseline) vs per-atom (mean/max pooled)
granularity on the SAME complexes with the SAME pair-construction logic. Runs the mandatory
guardrail controls (shuffled label, random-negative sanity anchor), exposure stratification +
min-exposure filter, and saves distance-overlap data + plots.

Usage:
    python -m masif_graph.experiments.run_m1 --n 40 --out logs/m1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from masif_graph.io.reference import complex_is_available, load_complex
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import (
    atom_contacts_direct,
    atom_positives_from_vertex_contacts,
    vertex_contacts,
)
from masif_graph.metrics.separation import (
    separation_auc,
    shuffled_label_auc,
)
from masif_graph.experiments.probe_core import (
    ComplexView,
    random_negative_dists,
    score_complex,
)

SEED = 0
VERTEX_HARD_R, VERTEX_WITHIN = 3.0, 3.0
ATOM_HARD_R, ATOM_WITHIN = 6.0, 6.0
# Per-complex AUC is only meaningful with enough positives; below this a single complex's AUC
# is dominated by 1-2 pairs (e.g. n_pos=1 -> AUC is trivially 1.0). Used for spread stats only.
MIN_POS_SPREAD = 10


def deterministic_complex_list(n: int, candidates_file: str):
    """First `n` complexes (in the deterministic candidate order) whose M0 outputs exist."""
    order = [l.strip() for l in open(candidates_file) if l.strip()]
    used, skipped = [], []
    for cid in order:
        if len(used) >= n:
            break
        if complex_is_available(cid):
            used.append(cid)
        else:
            skipped.append((cid, "M0 outputs missing/incomplete"))
    return used, skipped


def build_views(cid, positive_def):
    """Build vertex + atom(mean/max) views for one complex under a positive definition.

    positive_def in {"sc", "unfiltered", "atom_direct"}. Returns dict granularity->ComplexView.
    Positive derivation is identical across granularities (atoms inherit the vertex contacts),
    except "atom_direct" which uses direct heavy-atom proximity (atom granularities only).
    """
    p1, p2 = load_complex(cid)
    sa1 = build_surface_atoms(
        p1.verts, p1.atom_coords, p1.atom_element, p1.atom_resid,
        p1.desc_straight, p1.desc_flipped,
    )
    sa2 = build_surface_atoms(
        p2.verts, p2.atom_coords, p2.atom_element, p2.atom_resid,
        p2.desc_straight, p2.desc_flipped,
    )

    views = {}
    if positive_def in ("sc", "unfiltered"):
        sc1 = p1.sc if positive_def == "sc" else None
        vpairs, _ = vertex_contacts(p1.verts, p2.verts, 1.0, sc1=sc1)
        # vertex view
        views["vertex"] = ComplexView(
            cid, p1.desc_straight, p2.desc_flipped, p1.verts, p2.verts, vpairs,
            hard_radius=VERTEX_HARD_R, within_min=VERTEX_WITHIN,
        )
        apairs = atom_positives_from_vertex_contacts(vpairs, sa1.vertex_surf_idx, sa2.vertex_surf_idx)
        for op in ("mean", "max"):
            views[f"atom_{op}"] = ComplexView(
                cid, sa1.emb_straight[op], sa2.emb_flipped[op], sa1.coord, sa2.coord, apairs,
                exp1=sa1.n_owned, exp2=sa2.n_owned,
                hard_radius=ATOM_HARD_R, within_min=ATOM_WITHIN,
            )
    elif positive_def == "atom_direct":
        # map direct heavy-atom contacts (<4A) to SURFACE-atom rows (skip buried atoms)
        raw = atom_contacts_direct(sa1.coord, sa2.coord, 4.0)  # already in surface-atom index space
        for op in ("mean", "max"):
            views[f"atom_{op}"] = ComplexView(
                cid, sa1.emb_straight[op], sa2.emb_flipped[op], sa1.coord, sa2.coord, raw,
                exp1=sa1.n_owned, exp2=sa2.n_owned,
                hard_radius=ATOM_HARD_R, within_min=ATOM_WITHIN,
            )
    return views


def run(n, out_dir, candidates_file):
    os.makedirs(out_dir, exist_ok=True)
    used, skipped = deterministic_complex_list(n, candidates_file)
    print(f"Using {len(used)} complexes; skipped {len(skipped)}")

    results = {"config": {
        "n_requested": n, "n_used": len(used), "seed": SEED,
        "neg_ratio": 5, "neg_mix": {"cross": 0.5, "within": 0.3, "hard": 0.2},
        "vertex_hard_radius": VERTEX_HARD_R, "vertex_within_min": VERTEX_WITHIN,
        "atom_hard_radius": ATOM_HARD_R, "atom_within_min": ATOM_WITHIN,
        "pos_cutoff": 1.0, "sc_band": [0.5, 1.0], "atom_direct_cutoff": 4.0,
    }, "complexes_used": used, "skipped": skipped, "by_positive_def": {}}

    for positive_def in ("sc", "unfiltered", "atom_direct"):
        grans = ["atom_mean", "atom_max"] if positive_def == "atom_direct" else ["vertex", "atom_mean", "atom_max"]
        # build all views first (needed for cross-complex negative pools)
        all_views = {}
        for ci, cid in enumerate(used):
            try:
                all_views[cid] = build_views(cid, positive_def)
            except Exception as e:  # noqa: BLE001
                print(f"  [{positive_def}] {cid} build failed: {e}")
        cross_pools = {
            g: [(cid, v[g].f2) for cid, v in all_views.items() if g in v] for g in grans
        }

        def_block = {}
        for g in grans:
            scored = []
            for ci, cid in enumerate(used):
                if cid not in all_views or g not in all_views[cid]:
                    continue
                sc_res = score_complex(all_views[cid][g], cross_pools[g], seed=SEED + ci)
                scored.append(sc_res)
            # per-complex AUCs (neg_mix) + pooled
            per_complex = []
            all_pos, all_neg = [], []
            all_pos_minexp = []
            randneg_pos, randneg_neg = [], []
            for ci, (cid, sc_res) in enumerate(zip(used, scored)):
                if sc_res.n_pos == 0 or sc_res.n_neg == 0:
                    per_complex.append({"id": cid, "n_pos": int(sc_res.n_pos), "auc": None})
                    continue
                auc = separation_auc(sc_res.pos_dists, sc_res.neg_dists)
                # scheme-A random-negative anchor (identical construction across granularities;
                # no hard/within-radius confound -> the cleanest apples-to-apples comparison)
                rn = random_negative_dists(all_views[cid][g], seed=SEED + ci)
                auc_rand = separation_auc(sc_res.pos_dists, rn)
                per_complex.append({"id": cid, "n_pos": int(sc_res.n_pos),
                                    "n_neg": int(sc_res.n_neg), "auc": auc, "auc_randneg": auc_rand})
                all_pos.append(sc_res.pos_dists); all_neg.append(sc_res.neg_dists)
                if sc_res.pos_min_exposure is not None:
                    all_pos_minexp.append(sc_res.pos_min_exposure)
                randneg_pos.append(sc_res.pos_dists); randneg_neg.append(rn)

            all_pos = np.concatenate(all_pos) if all_pos else np.zeros(0)
            all_neg = np.concatenate(all_neg) if all_neg else np.zeros(0)
            pooled_auc = separation_auc(all_pos, all_neg)
            rng = np.random.default_rng(SEED + 12345)
            shuf = shuffled_label_auc(all_pos, all_neg, rng)
            randneg_auc = separation_auc(
                np.concatenate(randneg_pos) if randneg_pos else np.zeros(0),
                np.concatenate(randneg_neg) if randneg_neg else np.zeros(0),
            )
            # spread stats over complexes with enough positives to be meaningful
            aucs = [c["auc"] for c in per_complex
                    if c["auc"] is not None and c.get("n_pos", 0) >= MIN_POS_SPREAD]

            block = {
                "pooled_auc_negmix": pooled_auc,
                "pooled_auc_randomneg": randneg_auc,
                "shuffled_label_auc": shuf,
                "n_pos_total": int(len(all_pos)),
                "n_neg_total": int(len(all_neg)),
                "min_pos_for_spread": MIN_POS_SPREAD,
                "n_complexes_in_spread": len(aucs),
                "per_complex_auc_mean": float(np.mean(aucs)) if aucs else None,
                "per_complex_auc_std": float(np.std(aucs)) if aucs else None,
                "per_complex_auc_median": float(np.median(aucs)) if aucs else None,
                "per_complex_auc_min": float(np.min(aucs)) if aucs else None,
                "per_complex_auc_max": float(np.max(aucs)) if aucs else None,
                "per_complex": per_complex,
                "pos_dist_mean": float(all_pos.mean()) if len(all_pos) else None,
                "neg_dist_mean": float(all_neg.mean()) if len(all_neg) else None,
            }
            # save distance arrays for plotting
            np.save(os.path.join(out_dir, f"{positive_def}__{g}__pos.npy"), all_pos)
            np.save(os.path.join(out_dir, f"{positive_def}__{g}__neg.npy"), all_neg)

            # exposure stratification (atoms only)
            if g.startswith("atom") and all_pos_minexp:
                minexp = np.concatenate(all_pos_minexp)
                block["exposure"] = exposure_analysis(all_pos, all_neg, minexp)
            def_block[g] = block
        results["by_positive_def"][positive_def] = def_block

    with open(os.path.join(out_dir, "m1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(out_dir, 'm1_results.json')}")
    return results


def exposure_analysis(pos_dists, neg_dists, pos_minexp):
    """AUC stratified by min owned-vertex exposure of the positive pair + min-exposure filter sweep.

    Bins split positives by exposure; each bin's positives are scored against the FULL negative
    pool (negatives have no single exposure). The filter sweep keeps positives with min-exposure
    >= T and recomputes pooled AUC.
    """
    bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10**9)]
    strat = []
    for lo, hi in bins:
        m = (pos_minexp >= lo) & (pos_minexp <= hi)
        if m.sum() < 5:
            strat.append({"bin": f"{lo}-{hi if hi < 10**9 else '+'}", "n_pos": int(m.sum()), "auc": None})
            continue
        auc = separation_auc(pos_dists[m], neg_dists)
        strat.append({"bin": f"{lo}-{hi if hi < 10**9 else '+'}",
                      "n_pos": int(m.sum()), "auc": auc,
                      "pos_dist_mean": float(pos_dists[m].mean())})
    sweep = []
    for T in (1, 2, 3, 4, 5, 6, 8):
        m = pos_minexp >= T
        if m.sum() < 5:
            sweep.append({"min_exposure": T, "n_pos": int(m.sum()), "auc": None, "frac_pos_kept": float(m.mean())})
            continue
        sweep.append({"min_exposure": T, "n_pos": int(m.sum()),
                      "auc": separation_auc(pos_dists[m], neg_dists),
                      "frac_pos_kept": float(m.mean())})
    return {"stratified": strat, "min_exposure_sweep": sweep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="logs/m1")
    ap.add_argument("--candidates", default="logs/phase1_candidates.txt")
    args = ap.parse_args()
    run(args.n, args.out, args.candidates)
