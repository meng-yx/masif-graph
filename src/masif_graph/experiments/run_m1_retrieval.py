"""M1 secondary metric: top-k RETRIEVAL — the deployment-shaped test.

Scenario: an AF3-model query chain is searched against a database of holo chains; does its true
holo partner rank near the top by surface complementarity? Compares AF3-query vs holo-query (the
ceiling) retrieval over the same database.

Each chain carries an interface patch = its holo sc-positive atoms (the known binding site). Query =
that chain's interface descriptors (straight); DB entry = a holo chain's interface descriptors
(flipped). Match score S(Q,D) = median over query interface atoms of the min complementarity distance
to D's interface atoms (lower = better). Rank DB chains ascending; report the true partner's rank →
top-1 / top-5 recall + mean reciprocal rank. Frame-free (descriptor-only). No training.

Usage: python -m masif_graph.experiments.run_m1_retrieval --ids <file> --out <dir> [--min-pos 8]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import masif_graph.experiments.run_m1_af3 as M


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def chain_iface(rec):
    """Per-complex interface descriptors for each chain in each state, keyed by role.
    Returns {'holo':{'p1':(straight,flipped),'p2':...}, 'af3':{...}} restricted to intersection
    interface atoms (present in both states) so query/DB use the same physical patch."""
    inter = rec["inter"]
    if len(inter) == 0:
        return None
    ih, jh, ia, ja = inter[:, 0], inter[:, 1], inter[:, 2], inter[:, 3]
    # unique interface atoms per chain (dedup rows)
    h1i = np.unique(ih); h2i = np.unique(jh)
    a1i = np.unique(ia); a2i = np.unique(ja)
    holo, af3 = rec["holo"], rec["af3"]
    return {
        "holo": {"p1": (holo["p1"].straight[h1i], holo["p1"].flipped[h1i]),
                 "p2": (holo["p2"].straight[h2i], holo["p2"].flipped[h2i])},
        "af3": {"p1": (af3["p1"].straight[a1i], af3["p1"].flipped[a1i]),
                "p2": (af3["p2"].straight[a2i], af3["p2"].flipped[a2i])},
    }


def match_score(q_straight, d_flipped):
    """median over query atoms of min complementarity distance to DB atoms (lower = better)."""
    if len(q_straight) == 0 or len(d_flipped) == 0:
        return np.inf
    # pairwise distances q_straight (nq,80) vs d_flipped (nd,80)
    d = np.sqrt(((q_straight[:, None, :] - d_flipped[None, :, :]) ** 2).sum(-1))
    return float(np.median(d.min(axis=1)))


def run(args):
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        try:
            r = M.build_record(cid)
        except Exception:
            continue
        if r and r["af3"] is not None and len(r["inter"]) >= args.min_pos:
            ci = chain_iface(r)
            if ci:
                recs.append((cid, ci))
    log(f"usable complexes: {len(recs)}")

    # Build the DB: every holo chain (complex, role) with its flipped interface descriptors.
    db = []   # (complex, role, flipped_desc)
    for cid, ci in recs:
        db.append((cid, "p1", ci["holo"]["p1"][1]))
        db.append((cid, "p2", ci["holo"]["p2"][1]))

    def retrieve(query_state):
        """For each chain as a query (in query_state), rank its true holo partner over the DB."""
        ranks = []
        for cid, ci in recs:
            for qrole, prole in (("p1", "p2"), ("p2", "p1")):
                q_straight = ci[query_state][qrole][0]
                if len(q_straight) == 0:
                    continue
                scores = []
                for (dcid, drole, d_flipped) in db:
                    # a chain never retrieves itself (same complex+role); the true partner is the
                    # OTHER role of the same complex.
                    if dcid == cid and drole == qrole:
                        continue
                    scores.append(((dcid, drole), match_score(q_straight, d_flipped)))
                scores.sort(key=lambda x: x[1])
                order = [k for k, _ in scores]
                true = (cid, prole)
                if true in order:
                    ranks.append(order.index(true) + 1)
        return ranks

    out = {}
    for state, label in (("holo", "holo_query"), ("af3", "af3_query")):
        ranks = retrieve(state)
        if not ranks:
            continue
        ranks = np.array(ranks)
        out[label] = {
            "n_queries": len(ranks),
            "top1": float(np.mean(ranks <= 1)),
            "top5": float(np.mean(ranks <= 5)),
            "top10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "db_size": len(db),
        }
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "m1_retrieval.json"), "w"), indent=2)
    log("=" * 70)
    log(f"M1 RETRIEVAL (AF3-query -> holo-DB), DB size {len(db)} chains:")
    for label in ("holo_query", "af3_query"):
        e = out.get(label)
        if e:
            log(f"  {label:10s}: top1 {e['top1']:.2f} top5 {e['top5']:.2f} top10 {e['top10']:.2f} "
                f"MRR {e['mrr']:.2f} medRank {e['median_rank']:.0f} (n={e['n_queries']})")
    if "holo_query" in out and "af3_query" in out:
        log(f"  >>> retrieval drop AF3 vs holo: top1 {out['holo_query']['top1'] - out['af3_query']['top1']:+.2f}, "
            f"top5 {out['holo_query']['top5'] - out['af3_query']['top5']:+.2f}, "
            f"MRR {out['holo_query']['mrr'] - out['af3_query']['mrr']:+.2f}")
    log(f"results -> {os.path.join(args.out, 'm1_retrieval.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pos", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
