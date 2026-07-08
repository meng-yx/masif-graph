"""Phase-4 M0 gate: heterogeneous graph builder + rotation-invariance sanity test.

Runs the SE(3)-invariance report (graph/hetero.rotation_invariance_report) over N complexes x both
chains x several random SE(3) transforms. The M0 hard gate PASSES iff every check is invariant:
identical connectivity (atom-atom covalent, vertex-vertex mesh, vertex-atom) and edge scalar features
(distances, cos-angles) unchanged under rotation+translation, and node features byte-identical.

Also reports scale (atoms / surface-atoms / vertices / edges per chain) so the vertex-count profile is
on record for the coarsening decision.

Run (masif-graph env, Jed CPU):
  python -m masif_graph.experiments.p4_m0_gate --n 12 --seeds 1 2 3
Exit code 0 iff all PASS.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from masif_graph.io.reference import (
    load_complex, PDB_DIR, DESC_DIR, PRECOMP_DIR, complex_is_available,
)
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.graph.hetero import build_hetero_graph, rotation_invariance_report


def _holo_ids():
    train = set(l.strip() for l in open("data/lists/training.txt") if l.strip())
    m1 = set(l.strip() for l in open("logs/phase3/m1_ids.txt") if l.strip())

    def ok(d):
        if not os.path.isdir(os.path.join(DESC_DIR, d)):
            return False
        p = d.split("_")
        if len(p) != 3 or "AF" in p[0]:
            return False
        try:
            return complex_is_available(d)
        except Exception:
            return False

    holo = [d for d in os.listdir(DESC_DIR) if ok(d)]
    pool = sorted((set(holo) & train) - m1)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of complexes to test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--va-radius", type=float, default=5.0)
    ap.add_argument("--va-kmax", type=int, default=8)
    ap.add_argument("--ids", default=None, help="optional explicit id-list file")
    args = ap.parse_args()

    if args.ids:
        pool = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    else:
        pool = _holo_ids()[: args.n]

    n_pass = n_fail = 0
    vcounts, acounts, scounts = [], [], []
    vv_e, va_e = [], []
    failures = []
    for cid in pool:
        try:
            p1, p2 = load_complex(cid)
        except Exception as e:
            print(f"{cid}: LOAD FAIL {type(e).__name__}: {e}", flush=True)
            continue
        for ch in (p1, p2):
            surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                                       ch.desc_straight, ch.desc_flipped, ops=("mean",))
            pdb = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
            bk = dict(va_radius=args.va_radius, va_kmax=args.va_kmax)
            ok_all = True
            g0 = None
            for seed in args.seeds:
                rep, g0 = rotation_invariance_report(ch, surf, pdb, seed=seed, **bk)
                if not rep["PASS"]:
                    ok_all = False
                    failures.append((cid, ch.pid, seed,
                                     {k: v for k, v in rep.items() if v is not True and k != "PASS"}))
            tag = "PASS" if ok_all else "FAIL"
            if ok_all:
                n_pass += 1
            else:
                n_fail += 1
            vcounts.append(g0.n_vert); acounts.append(g0.n_atom); scounts.append(g0.n_surf)
            vv_e.append(g0.vv_edge.shape[1]); va_e.append(len(g0.va_v))
            print(f"{cid} {ch.pid} ({ch.pdb_id}_{ch.chain_ids}): "
                  f"atoms={g0.n_atom} surf={g0.n_surf} verts={g0.n_vert} "
                  f"aa={g0.aa_edge.shape[1]} vv={g0.vv_edge.shape[1]} va={len(g0.va_v)} "
                  f"rot[{','.join(map(str,args.seeds))}]={tag}", flush=True)

    vc = np.array(vcounts); ac = np.array(acounts); sc = np.array(scounts)
    print("\n=== SCALE (per chain) ===")
    print(f"  chains tested: {len(vc)}")
    print(f"  verts: min={vc.min()} med={np.median(vc):.0f} p95={np.percentile(vc,95):.0f} max={vc.max()}")
    print(f"  atoms: min={ac.min()} med={np.median(ac):.0f} max={ac.max()}")
    print(f"  surf-atoms: med={np.median(sc):.0f}")
    print(f"  vv_edges: med={np.median(vv_e):.0f} | va_edges: med={np.median(va_e):.0f}")
    print("\n=== M0 GATE ===")
    print(f"  chains PASS={n_pass}  FAIL={n_fail}  seeds={args.seeds}")
    if failures:
        print("  FAILURES:")
        for cid, pid, seed, bad in failures[:20]:
            print(f"    {cid} {pid} seed={seed}: {bad}")
    verdict = "PASS" if (n_fail == 0 and n_pass > 0) else "FAIL"
    print(f"  >>> M0 ROTATION-INVARIANCE GATE: {verdict} <<<")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
