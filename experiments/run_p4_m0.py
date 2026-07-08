"""Phase-4 M0 driver: build the heterogeneous graph over the holo train pool + rotation-invariance gate.

Runs in the `masif-graph` env (CPU, Jed). Produces `logs/phase4/m0_report.json`:
  - per-chain build success/failure across the pool (catch crashers early),
  - node/edge scale statistics,
  - the rotation-invariance gate result on a random sample (the M0 hard gate).

Usage:
  python experiments/run_p4_m0.py --n-rot 20 --out logs/phase4/m0_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from masif_graph.io import reference as R
from masif_graph.io.reference import load_complex, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.graph.hetero import build_hetero_graph, rotation_invariance_report


def holo_train_pool():
    train = set(l.strip() for l in open("data/lists/training.txt") if l.strip())
    m1 = set(l.strip() for l in open("logs/phase3/m1_ids.txt") if l.strip())

    def ok(d):
        if not os.path.isdir(os.path.join(R.DESC_DIR, d)):
            return False
        p = d.split("_")
        return len(p) == 3 and "AF" not in p[0] and R.complex_is_available(d)

    avail = set(d for d in os.listdir(R.DESC_DIR) if ok(d))
    return sorted((avail & train) - m1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rot", type=int, default=20, help="chains to run the rotation gate on")
    ap.add_argument("--max-build", type=int, default=0, help="cap #complexes to build (0=all)")
    ap.add_argument("--out", default="logs/phase4/m0_report.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool = holo_train_pool()
    if args.max_build:
        pool = pool[: args.max_build]
    print(f"pool = {len(pool)} holo complexes")

    stats = []       # per-chain scale
    failures = []    # (cid, pid, error)
    build_times = []
    rng = np.random.default_rng(0)
    rot_targets = set()  # (cid, pid) sampled for the rotation gate
    order = list(range(len(pool)))
    rng.shuffle(order)
    for k in order[: args.n_rot]:
        rot_targets.add((pool[k], "p1"))

    rot_results = []
    for ci, cid in enumerate(pool):
        try:
            p1, p2 = load_complex(cid)
        except Exception as e:
            failures.append((cid, "load", f"{type(e).__name__}: {e}"))
            continue
        for ch in (p1, p2):
            try:
                surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element,
                                           ch.atom_resid, ch.desc_straight, ch.desc_flipped,
                                           ops=("mean",))
                pdb = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
                t0 = time.perf_counter()
                g = build_hetero_graph(ch, surf, pdb)
                build_times.append(time.perf_counter() - t0)
                stats.append({
                    "cid": cid, "pid": ch.pid, "n_atom": g.n_atom, "n_surf": g.n_surf,
                    "n_vert": g.n_vert, "aa_edge": int(g.aa_edge.shape[1]),
                    "vv_edge": int(g.vv_edge.shape[1]), "va_edge": int(len(g.va_v)),
                    "atom_feat_dim": g.atom_feat_dim, "vert_feat_dim": g.vert_feat.shape[1],
                })
                if (cid, ch.pid) in rot_targets:
                    rep, _ = rotation_invariance_report(ch, surf, pdb, seed=ci + 1)
                    rot_results.append({"cid": cid, "pid": ch.pid, **{
                        k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                        for k, v in rep.items()}})
            except Exception as e:
                failures.append((cid, ch.pid, f"{type(e).__name__}: {e}"))
        if (ci + 1) % 20 == 0:
            print(f"  built {ci+1}/{len(pool)} complexes; failures so far={len(failures)}", flush=True)

    nv = np.array([s["n_vert"] for s in stats]) if stats else np.zeros(1)
    na = np.array([s["n_atom"] for s in stats]) if stats else np.zeros(1)
    n_rot_pass = sum(1 for r in rot_results if r["PASS"])
    summary = {
        "pool_size": len(pool),
        "chains_built": len(stats),
        "failures": failures,
        "build_time_s": {"median": float(np.median(build_times)) if build_times else None,
                          "max": float(np.max(build_times)) if build_times else None,
                          "total": float(np.sum(build_times)) if build_times else None},
        "n_vert": {"min": int(nv.min()), "median": float(np.median(nv)),
                   "p95": float(np.percentile(nv, 95)), "max": int(nv.max())},
        "n_atom": {"min": int(na.min()), "median": float(np.median(na)), "max": int(na.max())},
        "rotation_gate": {"n_tested": len(rot_results), "n_pass": n_rot_pass,
                          "PASS": n_rot_pass == len(rot_results) and len(rot_results) > 0,
                          "results": rot_results},
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n=== M0 REPORT ===")
    print(f"chains built: {len(stats)}/{2*len(pool)}  failures: {len(failures)}")
    print(f"rotation gate: {n_rot_pass}/{len(rot_results)} PASS -> {summary['rotation_gate']['PASS']}")
    print(f"verts/chain: median={summary['n_vert']['median']:.0f} p95={summary['n_vert']['p95']:.0f} max={summary['n_vert']['max']}")
    print(f"build: median={summary['build_time_s']['median']:.2f}s total={summary['build_time_s']['total']:.0f}s")
    if failures:
        print("FAILURES:")
        for f in failures[:20]:
            print("  ", f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
