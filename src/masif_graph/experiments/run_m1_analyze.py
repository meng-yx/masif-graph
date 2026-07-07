"""M1 stratified analysis: per-complex holo->AF3 gap vs AF3 confidence (pLDDT) and conformational
deviation (interface CA-RMSD). Answers *where* the gap lives.

Reuses build_record / score_complex from run_m1_af3 and the pLDDT/RMSD helpers from af3.analyze.
For RMSD we reconstruct the relabelled AF3 chain PDB on the fly (deterministic from the holo seq +
the AF3 model.cif). Emits <out>/m1_strata.json + a printed table.

Usage: python -m masif_graph.experiments.run_m1_analyze --ids <file> --out <dir> [--min-pos 8]
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def interface_resseqs_from_inter(rec, which):
    """resseq set of intersection-positive residues for chain p1(which=0) or p2(which=1)."""
    inter = rec["inter"]
    holo = rec["holo"]
    rows = inter[:, which]
    keyattr = holo["p1" if which == 0 else "p2"].keys
    return {keyattr[r][1] for r in set(rows.tolist())}


def af3_chain_pdb(pdb_id, chain, tmpdir):
    cif = A.af3_model_cif(pdb_id, chain)
    if cif is None:
        return None
    holo_pdb = os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb")
    _seq, mapres = chain_sequence(holo_pdb)
    out = os.path.join(tmpdir, f"{pdb_id}_{chain}_af.pdb")
    relabel_af3_chain_to_pdb(cif, mapres, chain, out)
    return out


def per_complex_auc(rec, regime, cross_pools, seed=0):
    sc = M.score_complex(rec, regime, cross_pools, seed=seed)
    if sc is None or len(sc["pos"]) == 0:
        return float("nan")
    return separation_auc(sc["pos"], sc["randneg"])


def run(args):
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        try:
            r = M.build_record(cid)
        except Exception as e:
            log(f"  {cid}: build FAIL {type(e).__name__}:{e}")
            continue
        if r and r["af3"] is not None and len(r["inter"]) >= args.min_pos:
            recs.append(r)
    log(f"usable: {len(recs)}")
    cross = M.build_cross_pools(recs)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for r in recs:
            cid = r["holo_id"]
            pdb, c1, c2 = cid.split("_")
            hh = per_complex_auc(r, "hh", cross)
            ah = per_complex_auc(r, "af3_holo", cross)
            plddt1 = A.chain_mean_plddt(pdb, c1)
            plddt2 = A.chain_mean_plddt(pdb, c2)
            # interface RMSD per chain: whole-chain, iface-under-global-fit, iface-under-LOCAL-fit
            rms = {}
            for which, c in ((0, c1), (1, c2)):
                afpdb = af3_chain_pdb(pdb, c, td)
                if afpdb is None:
                    rms[c] = (float("nan"), float("nan"), float("nan"))
                    continue
                ifres = interface_resseqs_from_inter(r, which)
                whole, iface_g, iface_l, _n = A.chain_ca_rmsd(pdb, c, afpdb, ifres)
                rms[c] = (whole, iface_g, iface_l)
            row = {
                "complex": cid, "n_inter": int(len(r["inter"])),
                "auc_hh": hh, "auc_af3_holo": ah, "gap": hh - ah,
                "mean_plddt": float(np.nanmean([plddt1, plddt2])),
                "plddt_c1": plddt1, "plddt_c2": plddt2,
                "whole_rmsd_c1": rms[c1][0], "whole_rmsd_c2": rms[c2][0],
                "iface_rmsd_globalfit_c1": rms[c1][1], "iface_rmsd_globalfit_c2": rms[c2][1],
                "iface_rmsd_localfit_c1": rms[c1][2], "iface_rmsd_localfit_c2": rms[c2][2],
                # honest stratifier = interface-LOCAL-fit RMSD (isolates local interface change)
                "max_iface_rmsd_local": float(np.nanmax([rms[c1][2], rms[c2][2]])),
                "max_iface_rmsd_global": float(np.nanmax([rms[c1][1], rms[c2][1]])),
                "max_whole_rmsd": float(np.nanmax([rms[c1][0], rms[c2][0]])),
            }
            rows.append(row)
            log(f"  {cid}: hh={hh:.3f} af3={ah:.3f} gap={row['gap']:+.3f} pLDDT={row['mean_plddt']:.0f} "
                f"ifaceRMSD(local)={row['max_iface_rmsd_local']:.2f} "
                f"(global-fit {row['max_iface_rmsd_global']:.2f}, whole {row['max_whole_rmsd']:.1f})")

    # correlations across complexes
    def corr(xk, yk):
        x = np.array([r[xk] for r in rows], float)
        y = np.array([r[yk] for r in rows], float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    summary = {
        "n": len(rows),
        "gap_mean": float(np.nanmean([r["gap"] for r in rows])),
        "gap_median": float(np.nanmedian([r["gap"] for r in rows])),
        "af3_holo_pooled_mean": float(np.nanmean([r["auc_af3_holo"] for r in rows])),
        "corr_gap_vs_plddt": corr("gap", "mean_plddt"),
        "corr_gap_vs_ifaceRMSD_local": corr("gap", "max_iface_rmsd_local"),
        "corr_gap_vs_ifaceRMSD_global": corr("gap", "max_iface_rmsd_global"),
        "corr_gap_vs_wholeRMSD": corr("gap", "max_whole_rmsd"),
        "corr_af3AUC_vs_plddt": corr("auc_af3_holo", "mean_plddt"),
        "corr_af3AUC_vs_ifaceRMSD_local": corr("auc_af3_holo", "max_iface_rmsd_local"),
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(args.out, "m1_strata.json"), "w"), indent=2)
    log("=" * 70)
    log(f"STRATA: n={summary['n']} gap_mean={summary['gap_mean']:+.3f} "
        f"gap_median={summary['gap_median']:+.3f}")
    log(f"  corr(gap, pLDDT)            = {summary['corr_gap_vs_plddt']:+.2f}  "
        f"(more negative = lower pLDDT -> bigger gap)")
    log(f"  corr(gap, ifaceRMSD LOCAL) = {summary['corr_gap_vs_ifaceRMSD_local']:+.2f}  "
        f"(honest: interface-local fit; bigger local interface change -> bigger gap)")
    log(f"  corr(gap, ifaceRMSD global)= {summary['corr_gap_vs_ifaceRMSD_global']:+.2f}  "
        f"(whole-chain fit; conflates domain motion)")
    log(f"  corr(gap, wholechain RMSD) = {summary['corr_gap_vs_wholeRMSD']:+.2f}")
    log(f"strata -> {os.path.join(args.out, 'm1_strata.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
