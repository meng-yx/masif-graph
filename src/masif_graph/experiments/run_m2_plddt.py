"""M2 lever-0: pLDDT-weighted (gated) matching — the cheapest robustness hypothesis.

Motivation (M1 strata): the holo->AF3 descriptor gap correlates strongly with AF3 confidence
(corr(gap, pLDDT) ~ -0.9). Hypothesis: the gap is carried by low-pLDDT query atoms; down-weighting
them (here, hard-gating at a pLDDT threshold) recovers the holo-like match.

Test: for the af3_holo regime, keep only positives whose AF3 *query* atom has pLDDT >= T, and
recompute the absolute AF3->holo AUC and the same-subset holo->holo AUC (so the gap is apples-to-
apples on identical atoms). Sweep T. If the gap shrinks toward 0 as T rises (while retaining a useful
fraction of atoms), pLDDT-weighting is a real, free lever. Absolute AF3 AUC is the headline (Phase-2
lesson). NO training -> no leakage.

Usage: python -m masif_graph.experiments.run_m2_plddt --ids <file> --out <dir> [--min-pos 8]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import masif_graph.experiments.run_m1_af3 as M
from masif_graph.af3.analyze import af3_atom_plddt
from masif_graph.metrics.separation import pair_distances, separation_auc, shuffled_label_auc

THRESHOLDS = [0, 50, 70, 80, 85, 90, 95]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def af3_row_plddt(rec, pid):
    """per-surface-row pLDDT for the AF3 chain `pid` (p1/p2), aligned to af3 ChainState rows."""
    cid = rec["holo_id"]
    pdb, c1, c2 = cid.split("_")
    chain = c1 if pid == "p1" else c2
    lut = af3_atom_plddt(pdb, chain)          # {(chain,resseq,name): pLDDT}
    cs = rec["af3"][pid]
    return np.array([lut.get(k, np.nan) for k in cs.keys], dtype=float)


def gated_dists(rec, thresh, seed):
    """Pooled pos/neg distances for af3_holo and hh on the pLDDT-gated intersection positives
    (gate on the AF3 query atom in each direction). Returns dict per regime or None."""
    inter = rec["inter"]
    if len(inter) == 0:
        return None
    holo, af3 = rec["holo"], rec["af3"]
    ih, jh, ia, ja = inter[:, 0], inter[:, 1], inter[:, 2], inter[:, 3]
    pl1 = af3_row_plddt(rec, "p1")            # af3 p1 per-row pLDDT
    pl2 = af3_row_plddt(rec, "p2")
    # per-positive query-atom pLDDT (aligned to the intersection list), one per direction
    qpl_dir1 = pl1[ia]                        # direction 1: query = af3 p1 atom (row ia)
    qpl_dir2 = pl2[ja]                        # direction 2: query = af3 p2 atom (row ja)
    rng = np.random.default_rng(seed)

    def side(q_s, d_f, qrows, drows, qpl, dn):
        # gate positives by the AF3 query-atom pLDDT >= thresh (qpl aligned to the positive list)
        keep = qpl >= thresh
        qr, dr = qrows[keep], drows[keep]
        if len(qr) == 0:
            return np.zeros(0), np.zeros(0)
        pos = pair_distances(q_s, d_f, np.stack([qr, dr], 1))
        rj = rng.integers(dn, size=len(qr))
        clash = rj == dr
        rj[clash] = (rj[clash] + 1) % dn
        diff = q_s[qr] - d_f[rj]
        neg = np.sqrt(np.sum(diff * diff, axis=1))
        return pos, neg

    out = {}
    # af3_holo: query=af3 (straight rows ia/ja), db=holo (flipped rows jh/ih). Gate on AF3 query pLDDT.
    ph1, nh1 = side(af3["p1"].straight, holo["p2"].flipped, ia, jh, qpl_dir1, holo["p2"].n)
    ph2, nh2 = side(af3["p2"].straight, holo["p1"].flipped, ja, ih, qpl_dir2, holo["p1"].n)
    out["af3_holo"] = (np.concatenate([ph1, ph2]), np.concatenate([nh1, nh2]))
    # hh on the SAME atoms (holo rows ih/jh) gated by the SAME AF3 query pLDDT -> apples-to-apples
    qh1, qn1 = side(holo["p1"].straight, holo["p2"].flipped, ih, jh, qpl_dir1, holo["p2"].n)
    qh2, qn2 = side(holo["p2"].straight, holo["p1"].flipped, jh, ih, qpl_dir2, holo["p1"].n)
    out["hh"] = (np.concatenate([qh1, qh2]), np.concatenate([qn1, qn2]))
    kept = int(np.sum(qpl_dir1 >= thresh) + np.sum(qpl_dir2 >= thresh))
    total = len(ia) + len(ja)
    out["_kept"] = (kept, total)
    return out


def run(args):
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        try:
            r = M.build_record(cid)
        except Exception as e:
            log(f"  {cid}: build FAIL {e}")
            continue
        if r and r["af3"] is not None and len(r["inter"]) >= args.min_pos:
            recs.append(r)
    log(f"usable: {len(recs)}")

    table = []
    for T in THRESHOLDS:
        af3_pos, af3_neg, hh_pos, hh_neg = [], [], [], []
        kept, total = 0, 0
        for r in recs:
            g = gated_dists(r, T, seed=0)
            if g is None:
                continue
            af3_pos.append(g["af3_holo"][0]); af3_neg.append(g["af3_holo"][1])
            hh_pos.append(g["hh"][0]); hh_neg.append(g["hh"][1])
            kept += g["_kept"][0]; total += g["_kept"][1]
        ap = np.concatenate(af3_pos) if af3_pos else np.zeros(0)
        an = np.concatenate(af3_neg) if af3_neg else np.zeros(0)
        hp = np.concatenate(hh_pos) if hh_pos else np.zeros(0)
        hn = np.concatenate(hh_neg) if hh_neg else np.zeros(0)
        af3_auc = separation_auc(ap, an)
        hh_auc = separation_auc(hp, hn)
        row = {"thresh": T, "kept_frac": kept / total if total else float("nan"),
               "af3_holo_auc": af3_auc, "hh_auc": hh_auc, "gap": hh_auc - af3_auc,
               "n_pos": int(len(ap))}
        table.append(row)
        log(f"  pLDDT>={T:3d}: kept {row['kept_frac']*100:4.0f}% ({row['n_pos']} pos) | "
            f"af3_holo AUC {af3_auc:.3f} | hh AUC {hh_auc:.3f} | gap {row['gap']:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump({"n": len(recs), "table": table},
              open(os.path.join(args.out, "m2_plddt.json"), "w"), indent=2)
    # verdict heuristic
    base = next(r for r in table if r["thresh"] == 0)
    hi = [r for r in table if r["kept_frac"] >= 0.3]
    best = min(hi, key=lambda r: r["gap"]) if hi else base
    log("=" * 70)
    log(f"M2 lever-0 (pLDDT gate): baseline gap {base['gap']:+.3f} (af3 {base['af3_holo_auc']:.3f}); "
        f"best-gate gap {best['gap']:+.3f} at pLDDT>={best['thresh']} "
        f"(af3 {best['af3_holo_auc']:.3f}, keep {best['kept_frac']*100:.0f}%)")
    log(f"  -> pLDDT-gating {'REDUCES' if best['gap'] < base['gap'] - 0.01 else 'does NOT reduce'} "
        f"the gap; absolute AF3 AUC {'rises' if best['af3_holo_auc'] > base['af3_holo_auc'] + 0.01 else 'flat'}")
    log(f"results -> {os.path.join(args.out, 'm2_plddt.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
