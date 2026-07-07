"""M2 lever-1: multi-sample ENSEMBLE matching (soft-min over AF3 diffusion samples).

Motivation (M1): AF3's 5 diffusion samples span the conformational uncertainty exactly where the gap
is worst (uncertain chains: inter-sample CA-RMSD up to ~15 A; confident chains ~0.1 A). Hypothesis: a
training-free ensemble match — represent the AF3 query atom by the BEST-matching sample (min descriptor
distance to the target) — recovers the hard cases by using AF3's own uncertainty.

For fairness the soft-min is applied to BOTH positives and negatives (the query-target distance is
min over the query's samples). Compares ensemble AF3->holo AUC vs single-sample AF3->holo vs holo ceiling.
Absolute AUC is the headline. NO training -> no leakage. Sample ids: {PDBID}AS{s}_{C1}_{C2}.

Usage: python -m masif_graph.experiments.run_m2_ensemble --ids <file> --out <dir> --samples 0,1,2 [--min-pos 8]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import masif_graph.experiments.run_m1_af3 as M
from masif_graph.metrics.separation import separation_auc, shuffled_label_auc


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sample_state(holo_id, s):
    pdb, c1, c2 = holo_id.split("_")
    return M.load_state(f"{pdb}AS{s}_{c1}_{c2}")


def build_ensemble_record(holo_id, samples, sc_band=(0.5, 1.0)):
    """holo + a list of AF3 sample states + holo positives. Returns dict or None."""
    holo = M.load_state(holo_id)
    if holo is None:
        return None
    states = []
    for s in samples:
        st = sample_state(holo_id, s)
        if st is not None:
            states.append(st)
    if not states:
        return None
    # holo positives (owner-atom pairs) in holo rows + keys
    from masif_graph.io.reference import load_complex
    from masif_graph.surface.atoms import build_surface_atoms
    from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts
    p1c, p2c = load_complex(holo_id)
    s1 = build_surface_atoms(p1c.verts, p1c.atom_coords, p1c.atom_element, p1c.atom_resid,
                             p1c.desc_straight, p1c.desc_flipped, ops=("mean",))
    s2 = build_surface_atoms(p2c.verts, p2c.atom_coords, p2c.atom_element, p2c.atom_resid,
                             p2c.desc_straight, p2c.desc_flipped, ops=("mean",))
    vp, _ = vertex_contacts(p1c.verts, p2c.verts, pos_cutoff=1.0, sc1=p1c.sc, sc_band=sc_band)
    holo_pos = atom_positives_from_vertex_contacts(vp, s1.vertex_surf_idx, s2.vertex_surf_idx)
    return {"holo_id": holo_id, "holo": holo, "states": states, "holo_pos": holo_pos}


def _ens_query_desc(states, pid, key):
    """list of straight descriptors for `key` across ensemble members where it is a surface atom."""
    out = []
    for st in states:
        cs = st[pid]
        r = cs.key2row.get(key)
        if r is not None:
            out.append(cs.straight[r])
    return out


def ensemble_pos_neg(rec, seed, use_holo_query=False):
    """Soft-min ensemble distances for af3_holo (query=ensemble, db=holo). Both directions pooled.
    If use_holo_query, query is the SINGLE holo surface (the ceiling on this atom subset)."""
    holo = rec["holo"]; states = rec["states"]
    h1, h2 = holo["p1"], holo["p2"]
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    # consider positives where BOTH atoms are usable: p2 in holo db (always, holo), p1 in >=1 sample
    for (i, j) in rec["holo_pos"]:
        ki, kj = h1.keys[i], h2.keys[j]
        # direction 1: query = chain1 (ensemble/holo), db = chain2 (holo, flipped)
        if kj in h2.key2row:
            dbrow = h2.key2row[kj]
            db_f = h2.flipped[dbrow]
            if use_holo_query:
                if ki in h1.key2row:
                    q = h1.straight[h1.key2row[ki]]
                    pos.append(np.linalg.norm(q - db_f))
                    rj = int(rng.integers(h2.n)); rj = (rj + 1) % h2.n if rj == dbrow else rj
                    neg.append(np.linalg.norm(q - h2.flipped[rj]))
            else:
                qs = _ens_query_desc(states, "p1", ki)
                if qs:
                    pos.append(min(np.linalg.norm(q - db_f) for q in qs))
                    rj = int(rng.integers(h2.n)); rj = (rj + 1) % h2.n if rj == dbrow else rj
                    neg.append(min(np.linalg.norm(q - h2.flipped[rj]) for q in qs))
        # direction 2: query = chain2, db = chain1 (holo, flipped)
        if ki in h1.key2row:
            dbrow = h1.key2row[ki]
            db_f = h1.flipped[dbrow]
            if use_holo_query:
                if kj in h2.key2row:
                    q = h2.straight[h2.key2row[kj]]
                    pos.append(np.linalg.norm(q - db_f))
                    rj = int(rng.integers(h1.n)); rj = (rj + 1) % h1.n if rj == dbrow else rj
                    neg.append(np.linalg.norm(q - h1.flipped[rj]))
            else:
                qs = _ens_query_desc(states, "p2", kj)
                if qs:
                    pos.append(min(np.linalg.norm(q - db_f) for q in qs))
                    rj = int(rng.integers(h1.n)); rj = (rj + 1) % h1.n if rj == dbrow else rj
                    neg.append(min(np.linalg.norm(q - h1.flipped[rj]) for q in qs))
    return np.array(pos), np.array(neg)


def single_sample_pos_neg(rec, seed):
    """Single-sample (first ensemble member) af3_holo distances — the lever-1 baseline."""
    holo = rec["holo"]; st = rec["states"][0]
    h1, h2 = holo["p1"], holo["p2"]
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for (i, j) in rec["holo_pos"]:
        ki, kj = h1.keys[i], h2.keys[j]
        if kj in h2.key2row and ki in st["p1"].key2row:
            dbrow = h2.key2row[kj]; q = st["p1"].straight[st["p1"].key2row[ki]]
            pos.append(np.linalg.norm(q - h2.flipped[dbrow]))
            rj = int(rng.integers(h2.n)); rj = (rj + 1) % h2.n if rj == dbrow else rj
            neg.append(np.linalg.norm(q - h2.flipped[rj]))
        if ki in h1.key2row and kj in st["p2"].key2row:
            dbrow = h1.key2row[ki]; q = st["p2"].straight[st["p2"].key2row[kj]]
            pos.append(np.linalg.norm(q - h1.flipped[dbrow]))
            rj = int(rng.integers(h1.n)); rj = (rj + 1) % h1.n if rj == dbrow else rj
            neg.append(np.linalg.norm(q - h1.flipped[rj]))
    return np.array(pos), np.array(neg)


def run(args):
    samples = [int(s) for s in args.samples.split(",")]
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        try:
            r = build_ensemble_record(cid, samples)
        except Exception as e:
            log(f"  {cid}: build FAIL {e}")
            continue
        if r is None:
            log(f"  {cid}: no ensemble samples")
            continue
        log(f"  {cid}: {len(r['states'])} samples, holo_pos={len(r['holo_pos'])}")
        recs.append(r)
    log(f"usable ensemble records: {len(recs)} (samples {samples})")

    def pooled(fn, **kw):
        P, N = [], []
        for r in recs:
            p, n = fn(r, seed=0, **kw)
            if len(p) >= args.min_pos:
                P.append(p); N.append(n)
        if not P:
            return float("nan"), 0
        return separation_auc(np.concatenate(P), np.concatenate(N)), len(P)

    single_auc, n1 = pooled(single_sample_pos_neg)
    ens_auc, ne = pooled(ensemble_pos_neg, use_holo_query=False)
    hh_auc, nh = pooled(ensemble_pos_neg, use_holo_query=True)
    result = {"samples": samples, "n_complexes": len(recs),
              "single_sample_af3_holo_auc": single_auc,
              "ensemble_af3_holo_auc": ens_auc,
              "holo_ceiling_auc": hh_auc}
    os.makedirs(args.out, exist_ok=True)
    json.dump(result, open(os.path.join(args.out, "m2_ensemble.json"), "w"), indent=2)
    log("=" * 70)
    log(f"M2 lever-1 (ensemble soft-min over samples {samples}, n={len(recs)}):")
    log(f"  holo ceiling      AUC {hh_auc:.3f}")
    log(f"  single-sample AF3 AUC {single_auc:.3f}")
    log(f"  ENSEMBLE AF3      AUC {ens_auc:.3f}  (delta vs single {ens_auc - single_auc:+.3f})")
    log(f"  -> ensemble {'HELPS' if ens_auc > single_auc + 0.01 else 'does NOT help'} "
        f"(gap to ceiling: single {hh_auc - single_auc:+.3f} -> ensemble {hh_auc - ens_auc:+.3f})")
    log(f"results -> {os.path.join(args.out, 'm2_ensemble.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default="0,1,2")
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
