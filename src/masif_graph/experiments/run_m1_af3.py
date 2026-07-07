"""M1: measure the holo->AF3 descriptor-separation gap.

For each complex we compare the frozen mean-pooled MaSIF descriptor in three regimes, on the SAME
intersection-positive set (atoms that are surface atoms in BOTH holo and the AF3 model, identity-
mapped by (chain,resseq,name) after relabelling AF3 to holo numbering):

  hh        holo query  x holo db     (the ceiling)
  af3_holo  AF3  query  x holo db     (the DEPLOYMENT scenario: query is an AF3 model, DB is holo)
  af3_af3   AF3  query  x AF3  db      (both predicted; secondary)

"query" contributes p1-straight, "db" contributes p2-flipped (reference flip-trick convention).
Both chain-directions (C1-as-query and C2-as-query) are pooled per complex. Negatives are frame-free
(the holo and AF3 frames differ, so coord-based negmix-hard is ill-defined across states):
  randneg   positive p1 vs a random same-complex p2 entity   (clean apples-to-apples, Phase-1 scheme)
  cross     positive p1 vs a random OTHER-complex p2 entity   (deployment retrieval flavour)

Absolute AUC per regime is the honest metric (Phase-2 lesson: the holo->X differential alone is
confounded). Reports pooled + per-complex median + spread + shuffled control (~0.5).

Usage: python -m masif_graph.experiments.run_m1_af3 --ids <file> --out <dir> [--seeds 3] [--min-pos 8]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np

from masif_graph.io.reference import load_complex, complex_is_available
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts
from masif_graph.metrics.separation import pair_distances, separation_auc, shuffled_label_auc


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def af3_id(holo_id: str) -> str:
    pdb, c1, c2 = holo_id.split("_")
    return f"{pdb}AF_{c1}_{c2}"


class ChainState:
    """Mean-pooled surface-atom descriptors + identity keys for one chain in one state."""
    def __init__(self, ch):
        surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                                   ch.desc_straight, ch.desc_flipped, ops=("mean",))
        self.straight = surf.emb_straight["mean"]
        self.flipped = surf.emb_flipped["mean"]
        self.coord = surf.coord
        keys = []
        for r in surf.atom_idx:
            chn, seq, _rn = ch.atom_resid[r].split(":")
            keys.append((chn, seq, str(ch.atom_name[r])))
        self.keys = keys
        self.key2row = {k: i for i, k in enumerate(keys)}
        self.n = self.straight.shape[0]


def load_state(state_id):
    if not complex_is_available(state_id):
        return None
    p1, p2 = load_complex(state_id)
    return {"p1": ChainState(p1), "p2": ChainState(p2)}


def build_record(holo_id, sc_band=(0.5, 1.0)):
    """Load holo + AF3, compute holo sc-positives, and the intersection positive set (holo rows +
    af3 rows) valid in both states. Returns dict or None."""
    if not complex_is_available(holo_id):
        return None
    holo = load_state(holo_id)
    if holo is None:
        return None
    af3 = load_state(af3_id(holo_id))

    # holo sc-filtered vertex contacts -> owner surface-atom positive pairs (p1 row, p2 row)
    p1c, p2c = load_complex(holo_id)
    s1 = build_surface_atoms(p1c.verts, p1c.atom_coords, p1c.atom_element, p1c.atom_resid,
                             p1c.desc_straight, p1c.desc_flipped, ops=("mean",))
    s2 = build_surface_atoms(p2c.verts, p2c.atom_coords, p2c.atom_element, p2c.atom_resid,
                             p2c.desc_straight, p2c.desc_flipped, ops=("mean",))
    vpairs, _ = vertex_contacts(p1c.verts, p2c.verts, pos_cutoff=1.0, sc1=p1c.sc, sc_band=sc_band)
    holo_pos = atom_positives_from_vertex_contacts(vpairs, s1.vertex_surf_idx, s2.vertex_surf_idx)

    inter = []   # (i_holo, j_holo) present in both holo and af3
    if af3 is not None and len(holo_pos) > 0:
        h1, h2 = holo["p1"], holo["p2"]
        a1, a2 = af3["p1"], af3["p2"]
        for i, j in holo_pos:
            ki, kj = h1.keys[i], h2.keys[j]
            if ki in a1.key2row and kj in a2.key2row:
                inter.append((i, j, a1.key2row[ki], a2.key2row[kj]))
    inter = np.array(inter, dtype=np.int64) if inter else np.zeros((0, 4), np.int64)
    # interface retention: fraction of holo positive PAIRS whose BOTH atoms remain AF3 surface atoms.
    # This is the "surface divergence" gap component (Phase-2's FASPR froze backbone -> ~1.0 always).
    retention = (len(inter) / len(holo_pos)) if len(holo_pos) > 0 else float("nan")
    return {"holo_id": holo_id, "holo": holo, "af3": af3,
            "n_holo_pos": len(holo_pos), "inter": inter, "retention": retention}


def _dir_dists(qs, ds, pos_q, pos_d, cross_pool_d, rng):
    """Positive + randneg + cross distances for one direction.
    qs/ds: ChainState for query(straight)/db(flipped). pos_q/pos_d: aligned positive rows.
    cross_pool_d: list of (id, flipped_desc) for cross-complex negatives (db state)."""
    pos = np.stack([pos_q, pos_d], axis=1)
    pos_d_arr = pair_distances(qs.straight, ds.flipped, pos)
    P = len(pos)
    # randneg: query atom vs random db atom (not the true partner)
    randneg = np.zeros(0)
    if P > 0 and ds.n >= 2:
        rj = rng.integers(ds.n, size=P)
        clash = rj == pos_d
        rj[clash] = (rj[clash] + 1) % ds.n
        diff = qs.straight[pos_q] - ds.flipped[rj]
        randneg = np.sqrt(np.sum(diff * diff, axis=1))
    # cross: query atom vs a random OTHER-complex db atom
    cross = np.zeros(0)
    others = [(cid, f) for (cid, f) in cross_pool_d if len(f) > 0]
    if P > 0 and others:
        cd = np.empty(P)
        pick = rng.integers(len(others), size=P)
        for k in range(P):
            _cid, f = others[pick[k]]
            jj = int(rng.integers(len(f)))
            d = qs.straight[pos_q[k]] - f[jj]
            cd[k] = np.sqrt(np.dot(d, d))
        cross = cd
    return pos_d_arr, randneg, cross


def score_complex(rec, regime, cross_pools, seed):
    """Pool both chain-directions for a regime. regime in {hh, af3_holo, af3_af3}.
    Returns dict of pooled pos/randneg/cross distance arrays (or None if no positives)."""
    inter = rec["inter"]
    if len(inter) == 0:
        return None
    holo, af3 = rec["holo"], rec["af3"]
    # pick states per regime: (query_state, db_state)
    qstate, dstate = {"hh": ("holo", "holo"), "af3_holo": ("af3", "holo"),
                      "af3_af3": ("af3", "af3")}[regime]
    Q = {"holo": holo, "af3": af3}[qstate]
    D = {"holo": holo, "af3": af3}[dstate]
    if Q is None or D is None:
        return None
    rng = np.random.default_rng(seed)
    ih, jh, ia, ja = inter[:, 0], inter[:, 1], inter[:, 2], inter[:, 3]
    # rows depend on state (holo rows vs af3 rows)
    qrow_p1 = ih if qstate == "holo" else ia   # C1 rows in query state
    drow_p2 = jh if dstate == "holo" else ja   # C2 rows in db state
    qrow_p2 = jh if qstate == "holo" else ja   # C2 rows in query state (for swap direction)
    drow_p1 = ih if dstate == "holo" else ia   # C1 rows in db state
    cpd = cross_pools[dstate]
    # direction 1: C1 query x C2 db
    p1, r1, c1 = _dir_dists(Q["p1"], D["p2"], qrow_p1, drow_p2, cpd["p2"], rng)
    # direction 2: C2 query x C1 db
    p2, r2, c2 = _dir_dists(Q["p2"], D["p1"], qrow_p2, drow_p1, cpd["p1"], rng)
    return {"pos": np.concatenate([p1, p2]),
            "randneg": np.concatenate([r1, r2]),
            "cross": np.concatenate([c1, c2])}


def build_cross_pools(recs):
    """cross_pools[state]['p1'/'p2'] = list of (holo_id, flipped_desc) over all complexes."""
    pools = {"holo": {"p1": [], "p2": []}, "af3": {"p1": [], "p2": []}}
    for r in recs:
        for state in ("holo", "af3"):
            st = r[state]
            if st is None:
                continue
            pools[state]["p1"].append((r["holo_id"], st["p1"].flipped))
            pools[state]["p2"].append((r["holo_id"], st["p2"].flipped))
    return pools


def aggregate(per_complex, neg_key):
    """pooled AUC + per-complex median/spread + shuffled, for a given negative scheme."""
    all_pos, all_neg = [], []
    per = []
    rng = np.random.default_rng(0)
    for cid, d in per_complex.items():
        pos, neg = d["pos"], d[neg_key]
        if len(pos) == 0 or len(neg) == 0:
            continue
        all_pos.append(pos); all_neg.append(neg)
        per.append((cid, separation_auc(pos, neg), len(pos)))
    if not all_pos:
        return None
    ap = np.concatenate(all_pos); an = np.concatenate(all_neg)
    pooled = separation_auc(ap, an)
    shuf = shuffled_label_auc(ap, an, rng)
    per_aucs = np.array([a for _c, a, _n in per if not np.isnan(a)])
    return {"pooled": pooled, "shuffled": shuf, "n_complexes": len(per),
            "per_complex_median": float(np.median(per_aucs)) if len(per_aucs) else float("nan"),
            "per_complex_mean": float(np.mean(per_aucs)) if len(per_aucs) else float("nan"),
            "per_complex_min": float(np.min(per_aucs)) if len(per_aucs) else float("nan"),
            "per_complex_max": float(np.max(per_aucs)) if len(per_aucs) else float("nan"),
            "per_complex": {c: {"auc": a, "n_pos": n} for c, a, n in per}}


def run(args):
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    retention_rows = []  # (cid, n_holo_pos, retention) for ALL af3-available complexes
    for cid in ids:
        try:
            r = build_record(cid)
        except Exception as e:
            log(f"  {cid}: build FAIL {type(e).__name__}:{e}")
            continue
        if r is None:
            log(f"  {cid}: holo unavailable")
            continue
        has_af3 = r["af3"] is not None
        ninter = len(r["inter"])
        if has_af3:
            retention_rows.append((cid, r["n_holo_pos"], r["retention"]))
        log(f"  {cid}: holo_pos={r['n_holo_pos']} af3={'Y' if has_af3 else 'N'} "
            f"intersection={ninter} retention={r['retention']:.2f}")
        if has_af3 and ninter >= args.min_pos:
            recs.append(r)
    log(f"usable records (af3 + >= {args.min_pos} intersection positives): {len(recs)}")
    # interface-retention summary (the surface-divergence gap component; includes excluded complexes)
    if retention_rows:
        rets = np.array([x[2] for x in retention_rows], float)
        # weighted by n_holo_pos (pair-count-weighted retention across all af3 complexes)
        w = np.array([x[1] for x in retention_rows], float)
        atom_ret = float(np.sum(rets * w) / np.sum(w)) if np.sum(w) > 0 else float("nan")
        log(f"INTERFACE RETENTION (af3 complexes n={len(rets)}): "
            f"mean={float(np.mean(rets)):.2f} median={float(np.median(rets)):.2f} "
            f"atom-weighted={atom_ret:.2f} min={float(np.min(rets)):.2f} "
            f"(#complexes with retention<0.5: {int(np.sum(rets < 0.5))})")
    if len(recs) < 2:
        log("FATAL: too few usable records")
        os.makedirs(args.out, exist_ok=True)
        json.dump({"error": "too_few", "n": len(recs)}, open(os.path.join(args.out, "m1_results.json"), "w"))
        return

    cross_pools = build_cross_pools(recs)
    regimes = ["hh", "af3_holo", "af3_af3"]
    results = {"ids": [r["holo_id"] for r in recs], "n_complexes": len(recs),
               "seeds": args.seeds, "regimes": {},
               "retention": {"per_complex": {c: {"n_holo_pos": n, "retention": ret}
                                             for c, n, ret in retention_rows}}}
    if retention_rows:
        rr = np.array([x[2] for x in retention_rows], float)
        ww = np.array([x[1] for x in retention_rows], float)
        results["retention"]["summary"] = {
            "n_af3_complexes": len(rr), "mean": float(np.mean(rr)),
            "median": float(np.median(rr)),
            "pair_weighted": float(np.sum(rr * ww) / np.sum(ww)),
            "min": float(np.min(rr)), "n_below_0.5": int(np.sum(rr < 0.5))}
    # average over seeds (randneg/cross sampling)
    seed_tables = {rg: {"randneg": [], "cross": []} for rg in regimes}
    for s in range(args.seeds):
        per_regime = {rg: {} for rg in regimes}
        for r in recs:
            for rg in regimes:
                cseed = 1000 * s + (zlib.crc32(r["holo_id"].encode()) % 100000)
                sc = score_complex(r, rg, cross_pools, seed=cseed)
                if sc is not None:
                    per_regime[rg][r["holo_id"]] = sc
        for rg in regimes:
            for nk in ("randneg", "cross"):
                agg = aggregate(per_regime[rg], nk)
                if agg is not None:
                    seed_tables[rg][nk].append(agg)
        # keep the per-complex detail from seed 0 for reporting
        if s == 0:
            results["per_complex_seed0"] = {
                rg: {nk: aggregate(per_regime[rg], nk) for nk in ("randneg", "cross")}
                for rg in regimes}

    def mean_sd(vals):
        v = np.array(vals, dtype=float)
        return float(np.mean(v)), float(np.std(v))

    for rg in regimes:
        entry = {}
        for nk in ("randneg", "cross"):
            tabs = seed_tables[rg][nk]
            if not tabs:
                continue
            pooled_m, pooled_s = mean_sd([t["pooled"] for t in tabs])
            med_m, med_s = mean_sd([t["per_complex_median"] for t in tabs])
            shuf_m, _ = mean_sd([t["shuffled"] for t in tabs])
            entry[nk] = {"pooled_mean": pooled_m, "pooled_sd": pooled_s,
                         "per_complex_median_mean": med_m, "per_complex_median_sd": med_s,
                         "shuffled_mean": shuf_m, "n_complexes": tabs[0]["n_complexes"]}
        results["regimes"][rg] = entry

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "m1_results.json"), "w"), indent=2)

    # summary
    log("=" * 78)
    log("M1 RESULTS — descriptor-separation AUC (mean over seeds); ABSOLUTE per regime")
    for nk in ("randneg", "cross"):
        log(f"--- negatives = {nk} ---")
        for rg in regimes:
            e = results["regimes"][rg].get(nk)
            if e:
                log(f"  {rg:9s}: pooled {e['pooled_mean']:.3f}±{e['pooled_sd']:.3f} | "
                    f"per-cplx median {e['per_complex_median_mean']:.3f} | shuf {e['shuffled_mean']:.2f} "
                    f"(n={e['n_complexes']})")
        hh = results["regimes"]["hh"].get(nk)
        ah = results["regimes"]["af3_holo"].get(nk)
        if hh and ah:
            log(f"  >>> GAP holo->AF3 ({nk}): pooled {hh['pooled_mean'] - ah['pooled_mean']:+.3f} "
                f"(hh {hh['pooled_mean']:.3f} - af3_holo {ah['pooled_mean']:.3f}); "
                f"absolute af3_holo = {ah['pooled_mean']:.3f}")
    log(f"results -> {os.path.join(args.out, 'm1_results.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
