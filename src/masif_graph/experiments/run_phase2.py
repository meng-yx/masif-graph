"""Phase-2 driver: build records (holo + repack), M1 controls, M2 ablation, differential metric.

Usage:
  python -m masif_graph.experiments.run_phase2 --ids <file> --out <dir> \
      [--steps 200] [--seeds 3] [--min-pos 10] [--max-complexes 40] [--split-frac 0.5]

Emits <out>/phase2_results.json and prints a summary. Records are cached to <out>/records.pkl.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch

from masif_graph.graph.dataset import build_complex_record
from masif_graph.graph.build import build_atom_graph
from masif_graph.graph.model import MaSIFGraphModel, AblationConfig
from masif_graph.graph.dataset import graph_to_tensors
from masif_graph.train.harness import CELLS, HParams, train_cell, eval_state, _embed_chain, _tensor_cache


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_records(ids, min_pos, max_complexes, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        log(f"loading cached records from {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    recs = []
    for cid in ids:
        try:
            r = build_complex_record(cid)
        except Exception as e:
            log(f"  {cid}: build FAILED ({type(e).__name__}: {e})")
            continue
        if r is None:
            continue
        npos = len(r.holo_pos)
        has_rp = r.repack is not None
        inter = len(r.inter_pos_holo) if r.inter_pos_holo is not None else 0
        log(f"  {cid}: holo_pos={npos} repack={'Y' if has_rp else 'N'} intersection_pos={inter}")
        if npos >= min_pos and has_rp and inter >= min_pos:
            recs.append(r)
        if len(recs) >= max_complexes:
            break
    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump(recs, fh)
        log(f"cached {len(recs)} records -> {cache_path}")
    return recs


def rotation_invariance_check(rec, seed=0):
    """Architectural check: rotate a chain's coords, rebuild the graph, compare fused embedding.
    Rotation-invariant by construction (only distances/bond-types feed the model)."""
    from masif_graph.io.reference import load_complex, PDB_DIR
    from masif_graph.surface.atoms import build_surface_atoms
    torch.manual_seed(seed)
    cfg = AblationConfig(True, True, True, "full")
    model = MaSIFGraphModel(cfg).eval()
    r1 = rec.holo["p1"]
    t0 = graph_to_tensors(r1.graph, use_rotatable=True)
    with torch.no_grad():
        e0 = _embed_chain(model, {id(r1.graph): t0}, r1, "straight").numpy()
    # rebuild graph from a rotated copy of the chain
    p1, _ = load_complex(rec.holo_id)
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(A) < 0:
        A[:, 0] = -A[:, 0]
    t = rng.standard_normal(3) * 10
    class _C:  # shallow rotated view
        pass
    rc = copy.copy(p1)
    rc.atom_coords = p1.atom_coords @ A.T + t
    rc.verts = p1.verts @ A.T + t
    surf = build_surface_atoms(rc.verts, rc.atom_coords, rc.atom_element, rc.atom_resid,
                               rc.desc_straight, rc.desc_flipped, ops=("mean",))
    g2 = build_atom_graph(rc, surf, os.path.join(PDB_DIR, f"{rc.pdb_id}_{rc.chain_ids}.pdb"))
    from masif_graph.train.harness import _chain  # noqa
    from masif_graph.graph.dataset import ChainRec
    r2 = ChainRec(rec.holo_id, "p1", "holo", surf.coord.shape[0], surf.emb_straight["mean"],
                  surf.emb_flipped["mean"], surf.coord, r1.keys, r1.key2row, g2)
    t2 = graph_to_tensors(g2, use_rotatable=True)
    with torch.no_grad():
        e2 = _embed_chain(model, {id(g2): t2}, r2, "straight").numpy()
    return float(np.abs(e0 - e2).max())


def run(args):
    os.makedirs(args.out, exist_ok=True)
    ids = [l.strip() for l in open(args.ids) if l.strip()]
    log(f"{len(ids)} candidate ids")
    recs = build_records(ids, args.min_pos, args.max_complexes,
                         cache_path=os.path.join(args.out, "records.pkl"))
    log(f"usable records (>= {args.min_pos} intersection positives + repack): {len(recs)}")
    if len(recs) < 4:
        log("FATAL: too few usable records; need repacked complexes with positives.")
        json.dump({"error": "too few records", "n": len(recs)}, open(os.path.join(args.out, "phase2_results.json"), "w"))
        return

    # split by complex (holo+repack of a complex stay together -> no twin leakage).
    # Test complexes need >= test_min_pos intersection positives for a stable per-complex AUC;
    # the remaining (incl. lower-positive) complexes form the train pool. Documented bias.
    rng = np.random.default_rng(args.split_seed)
    eligible_test = [r for r in recs if len(r.inter_pos_holo) >= args.test_min_pos]
    rng.shuffle(eligible_test)
    n_test = min(len(eligible_test), max(2, int(round((1 - args.split_frac) * len(recs)))))
    test_recs = eligible_test[:n_test]
    test_ids = {r.holo_id for r in test_recs}
    train_recs = [r for r in recs if r.holo_id not in test_ids]
    log(f"split: {len(train_recs)} train / {len(test_recs)} test (by complex; "
        f"test requires >= {args.test_min_pos} intersection positives)")
    log(f"  train: {[r.holo_id for r in train_recs]}")
    log(f"  test:  {[r.holo_id for r in test_recs]}")

    # --- M1 controls ---
    rot_err = rotation_invariance_check(recs[0])
    log(f"M1 rotation-invariance max|Δembed| = {rot_err:.2e} (should be ~1e-5)")

    results = {"n_train": len(train_recs), "n_test": len(test_recs),
               "train_ids": [r.holo_id for r in train_recs],
               "test_ids": [r.holo_id for r in test_recs],
               "rotation_invariance_max_abs_diff": rot_err,
               "cells": {}}

    # --- raw-desc baseline (Phase-1 mean-pool anchor; no model, no training) ---
    for state in ("holo", "repack"):
        rb = eval_state(None, test_recs, state, use_rotatable=False, seed=args.eval_seed)
        results.setdefault("raw_baseline", {})[state] = rb
        log(f"RAW-DESC {state}: randneg pooled {rb['pooled_randneg']:.3f} median {rb['median_randneg']:.3f} "
            f"| negmix pooled {rb['pooled_negmix']:.3f} | shuffled {rb['shuffled_randneg']:.3f} (n={rb['n_complexes']})")

    # --- M2 ablation ---
    for cfg in CELLS:
        cell_out = {"seeds": []}
        for s in range(args.seeds):
            hp = HParams(steps=args.steps, seed=s, p_aug=args.p_aug)
            t0 = time.time()
            model = train_cell(cfg, train_recs, hp, log=log)
            holo = eval_state(model, test_recs, "holo", cfg.use_rotatable, seed=args.eval_seed)
            repack = eval_state(model, test_recs, "repack", cfg.use_rotatable, seed=args.eval_seed)
            dd_rn = holo["pooled_randneg"] - repack["pooled_randneg"]
            dd_nm = holo["pooled_negmix"] - repack["pooled_negmix"]
            cell_out["seeds"].append({"seed": s, "holo": holo, "repack": repack,
                                      "degradation_randneg": dd_rn, "degradation_negmix": dd_nm})
            log(f"[{cfg.name}] seed {s}: holo_rn {holo['pooled_randneg']:.3f} repack_rn {repack['pooled_randneg']:.3f} "
                f"Δrn {dd_rn:+.3f} | holo_nm {holo['pooled_negmix']:.3f} repack_nm {repack['pooled_negmix']:.3f} Δnm {dd_nm:+.3f} "
                f"| shuf {repack['shuffled_randneg']:.2f} ({time.time()-t0:.0f}s)")
        # aggregate across seeds
        def agg(key, sub):
            vals = [sd[sub][key] for sd in cell_out["seeds"]]
            return float(np.mean(vals)), float(np.std(vals))
        cell_out["holo_randneg_mean_sd"] = agg("pooled_randneg", "holo")
        cell_out["repack_randneg_mean_sd"] = agg("pooled_randneg", "repack")
        cell_out["degradation_randneg_mean_sd"] = (
            float(np.mean([sd["degradation_randneg"] for sd in cell_out["seeds"]])),
            float(np.std([sd["degradation_randneg"] for sd in cell_out["seeds"]])),
        )
        results["cells"][cfg.name] = cell_out
        hm, hs = cell_out["holo_randneg_mean_sd"]
        rm, rs = cell_out["repack_randneg_mean_sd"]
        dm, ds = cell_out["degradation_randneg_mean_sd"]
        log(f"=== {cfg.name}: holo_rn {hm:.3f}±{hs:.3f} repack_rn {rm:.3f}±{rs:.3f} DEGRADATION {dm:+.3f}±{ds:.3f} ===")

    json.dump(results, open(os.path.join(args.out, "phase2_results.json"), "w"), indent=2)
    log(f"wrote {os.path.join(args.out, 'phase2_results.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--test-min-pos", type=int, default=10)
    ap.add_argument("--max-complexes", type=int, default=60)
    ap.add_argument("--split-frac", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--p-aug", type=float, default=0.5,
                    help="prob a train chain uses its repacked state (rotamer augmentation). "
                         "Set 0 to isolate the graph's intrinsic robustness (no perturbation seen in training).")
    run(ap.parse_args())
