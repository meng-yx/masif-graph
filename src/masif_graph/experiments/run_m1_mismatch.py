"""M1 refinement (per user guidance): separate ADDRESSABLE induced-fit degradation from UNADDRESSABLE
structural-mismatch (domain swap / context-dependent fold, e.g. 1A2W).

A holo interface is 'structural-mismatch' in the AF3 monomer if EITHER (structure-fixed thresholds):
  (a) exposure loss: interface-atom RETENTION < 0.5 (holo interface atoms that stop being AF3 surface
      atoms — the binding surface is physically gone), OR
  (b) local geometry: interface-LOCAL Cα-RMSD > 4.0 A (interface residues present but in a fundamentally
      different local backbone conformation, beyond induced-fit).
Complex = mismatch if EITHER chain is. 1A2W is the positive control (must be flagged).

Reports: per-complex table; #mismatch vs #induced-fit; and the af3_holo gap UNFILTERED vs FILTERED to
induced-fit-only (how much of the gap a better descriptor could even close). Both always reported.

Usage: python -m masif_graph.experiments.run_m1_mismatch --ids <file> --out <dir> [--min-pos 8]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np

import masif_graph.experiments.run_m1_af3 as M
from masif_graph.af3 import analyze as A
from masif_graph.af3.sequence import chain_sequence
from masif_graph.af3.relabel import relabel_af3_chain_to_pdb
from masif_graph.io.reference import PDB_DIR
from masif_graph.metrics.separation import separation_auc

RETENTION_MIN = 0.5      # structure-fixed
IFACE_LOCAL_RMSD_MAX = 4.0  # A, structure-fixed


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def af3_chain_pdb(pdb_id, chain, tmpdir):
    cif = A.af3_model_cif(pdb_id, chain)
    if cif is None:
        return None
    _seq, mapres = chain_sequence(os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb"))
    out = os.path.join(tmpdir, f"{pdb_id}_{chain}_af.pdb")
    relabel_af3_chain_to_pdb(cif, mapres, chain, out)
    return out


def classify(rec, td):
    """Return (is_mismatch, retention, max_iface_local_rmsd, reason)."""
    cid = rec["holo_id"]; pdb, c1, c2 = cid.split("_")
    retention = rec["retention"]
    # interface-local RMSD per chain
    max_local = 0.0
    from masif_graph.experiments.run_m1_analyze import interface_resseqs_from_inter
    have_rmsd = False
    for which, c in ((0, c1), (1, c2)):
        afpdb = af3_chain_pdb(pdb, c, td)
        if afpdb is None:
            continue
        ifres = interface_resseqs_from_inter(rec, which)
        _w, _g, local, _n = A.chain_ca_rmsd(pdb, c, afpdb, ifres)
        if np.isfinite(local):
            have_rmsd = True
            max_local = max(max_local, local)
    reasons = []
    if np.isfinite(retention) and retention < RETENTION_MIN:
        reasons.append(f"retention<{RETENTION_MIN}({retention:.2f})")
    if have_rmsd and max_local > IFACE_LOCAL_RMSD_MAX:
        reasons.append(f"ifaceRMSD>{IFACE_LOCAL_RMSD_MAX}({max_local:.1f})")
    return (len(reasons) > 0), retention, (max_local if have_rmsd else float("nan")), ";".join(reasons)


def af3_holo_auc_over(recs, cross_pools, seed=0):
    per = {}
    for r in recs:
        sc = M.score_complex(r, "af3_holo", cross_pools, seed=seed)
        if sc is not None and len(sc["pos"]) > 0:
            per[r["holo_id"]] = sc
    hh = {}
    for r in recs:
        sc = M.score_complex(r, "hh", cross_pools, seed=seed)
        if sc is not None and len(sc["pos"]) > 0:
            hh[r["holo_id"]] = sc
    if not per:
        return float("nan"), float("nan"), 0
    ap = np.concatenate([d["pos"] for d in per.values()])
    an = np.concatenate([d["randneg"] for d in per.values()])
    hp = np.concatenate([d["pos"] for d in hh.values()])
    hn = np.concatenate([d["randneg"] for d in hh.values()])
    return separation_auc(ap, an), separation_auc(hp, hn), len(per)


def run(args):
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    all_af3 = []
    for cid in ids:
        try:
            r = M.build_record(cid)
        except Exception as e:
            log(f"  {cid}: build FAIL {e}"); continue
        if r is None or r["af3"] is None:
            continue
        all_af3.append(r)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for r in all_af3:
            is_mm, ret, rmsd, reason = classify(r, td)
            rows.append({"complex": r["holo_id"], "n_inter": int(len(r["inter"])),
                         "retention": ret, "iface_local_rmsd": rmsd,
                         "structural_mismatch": is_mm, "reason": reason})
            log(f"  {r['holo_id']}: inter={len(r['inter'])} retention={ret:.2f} "
                f"ifaceRMSD_local={rmsd if np.isfinite(rmsd) else float('nan'):.2f} "
                f"-> {'MISMATCH' if is_mm else 'induced-fit'} {reason}")

    mm = [x for x in rows if x["structural_mismatch"]]
    log("=" * 74)
    log(f"structural-mismatch: {len(mm)}/{len(rows)} complexes ({100*len(mm)/len(rows):.0f}%): "
        f"{', '.join(x['complex'] for x in mm)}")

    # positive control: 1A2W must be flagged (if present)
    a1a2w = next((x for x in rows if x["complex"].startswith("1A2W")), None)
    if a1a2w is not None:
        ok = a1a2w["structural_mismatch"]
        log(f"POSITIVE CONTROL 1A2W flagged as mismatch: {ok} "
            f"(retention={a1a2w['retention']:.2f}) {'PASS' if ok else 'FAIL — detector wrong'}")

    # gap unfiltered vs induced-fit-only (both need >= min-pos intersection positives)
    mm_ids = {x["complex"] for x in mm}
    usable = [r for r in all_af3 if len(r["inter"]) >= args.min_pos]
    usable_if = [r for r in usable if r["holo_id"] not in mm_ids]
    cross = M.build_cross_pools(usable)
    cross_if = M.build_cross_pools(usable_if)
    af3_all, hh_all, n_all = af3_holo_auc_over(usable, cross)
    af3_if, hh_if, n_if = af3_holo_auc_over(usable_if, cross_if)
    log("-" * 74)
    log(f"UNFILTERED (all usable, N={n_all}):     af3_holo {af3_all:.3f} | hh {hh_all:.3f} "
        f"| gap {hh_all - af3_all:+.3f}")
    log(f"INDUCED-FIT ONLY (N={n_if}):            af3_holo {af3_if:.3f} | hh {hh_if:.3f} "
        f"| gap {hh_if - af3_if:+.3f}")
    log(f"  -> of the usable set, {n_all - n_if} were structural-mismatch; excluding them the "
        f"ADDRESSABLE gap is {hh_if - af3_if:+.3f}")

    out = {"thresholds": {"retention_min": RETENTION_MIN, "iface_local_rmsd_max": IFACE_LOCAL_RMSD_MAX},
           "n_af3_complexes": len(rows), "n_structural_mismatch": len(mm),
           "mismatch_ids": sorted(mm_ids), "rows": rows,
           "gap_unfiltered": {"af3_holo": af3_all, "hh": hh_all, "gap": hh_all - af3_all, "n": n_all},
           "gap_induced_fit": {"af3_holo": af3_if, "hh": hh_if, "gap": hh_if - af3_if, "n": n_if}}
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "m1_mismatch.json"), "w"), indent=2)
    log(f"results -> {os.path.join(args.out, 'm1_mismatch.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
