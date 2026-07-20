"""Phase-4 M2 deployment test: top-k interface-patch RETRIEVAL, AF3 query vs holo DB.

The B.0/B.1 finding is that the from-scratch encoder is conformation-invariant at the descriptor level.
This asks the deployment question: does that invariance convert to better *retrieval* — given an AF3-model
query chain, does its true holo partner rank near the top of a holo database? Phase-3 measured this for the
FROZEN descriptor (AF3-query top-5 recall 0.64 vs holo 0.78); here we run the identical protocol with the
learned (z, T) and recompute frozen on the SAME interface patches, so learned-vs-frozen and holo-vs-AF3 are
both exact.

Protocol (mirrors experiments/run_m1_retrieval.py):
  - each chain's patch = its intersection interface atoms (present in both holo & AF3);
  - query = patch embeddings in the query state (holo or af3); DB = every holo chain's patch;
  - learned score S(Q,D) = median over query atoms of the MAX bilinear complementarity zQ^T T zD (higher =
    better; design §5.2 max-inner-product), rank DB descending; frozen = median-of-min descriptor distance
    (lower = better), rank ascending — the exact Phase-3 score;
  - true partner = the OTHER role of the same complex; a chain never retrieves its own role.
Metrics: top-1/5/10 recall, MRR, median rank. Reports the AF3-vs-holo drop per method — small drop = the
invariance paid off at retrieval time.

Usage:
  python -m masif_graph.p4.retrieval_af3 --data <npz> --ids <file> --ckpt vicreg_sc_best_seed0.pt \
      --out ret.json [--min-pos 8] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from masif_graph.p4.eval_af3 import Rec, build_encoder, encode_all  # noqa: E402


def metrics(ranks):
    r = np.array(ranks, float)
    return {"n_queries": len(r), "top1": float(np.mean(r <= 1)), "top5": float(np.mean(r <= 5)),
            "top10": float(np.mean(r <= 10)), "mrr": float(np.mean(1.0 / r)),
            "median_rank": float(np.median(r))}


def run(args):
    device = args.device
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        r = Rec(args.data, cid, "pos_sc", device)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos:
            recs.append(r)
    print(f"usable complexes (af3 + >= {args.min_pos} intersection): {len(recs)}", flush=True)

    enc, comp, src = build_encoder(recs, args.ckpt, device)
    enc.eval()
    emb = encode_all(enc, recs, device)   # normalized z per chain/state
    T = comp.T

    # per-chain interface patches: learned z + frozen desc, on the identical intersection atom set.
    patches = {}   # cid -> {'p1'/'p2': {'z_holo','z_af3','fs_holo','fs_af3','fd_holo','fd_af3'}}
    for r in recs:
        ih = np.unique(r.inter[:, 0]); jh = np.unique(r.inter[:, 1])
        ia = np.unique(r.inter[:, 2]); ja = np.unique(r.inter[:, 3])
        e = emb[r.cid]
        patches[r.cid] = {
            "p1": {"z_holo": e["h1"][ih], "z_af3": e["a1"][ia],
                   "fs_holo": r.hg1["desc_straight"][ih], "fs_af3": r.ag1["desc_straight"][ia],
                   "fd_holo": r.hg1["desc_flipped"][ih], "fd_af3": r.ag1["desc_flipped"][ia]},
            "p2": {"z_holo": e["h2"][jh], "z_af3": e["a2"][ja],
                   "fs_holo": r.hg2["desc_straight"][jh], "fs_af3": r.ag2["desc_straight"][ja],
                   "fd_holo": r.hg2["desc_flipped"][jh], "fd_af3": r.ag2["desc_flipped"][ja]},
        }

    # DB = every holo chain patch (learned z_holo + frozen flipped desc)
    db = [(cid, role, patches[cid][role]) for cid in patches for role in ("p1", "p2")]

    def learned_score(zq, zd):
        if zq.shape[0] == 0 or zd.shape[0] == 0:
            return -1e9
        s = zq @ T @ zd.t()                 # (nq, nd) bilinear complementarity
        return float(s.max(dim=1).values.median())

    def frozen_score(qs, df):
        if qs.shape[0] == 0 or df.shape[0] == 0:
            return 1e9
        d = torch.sqrt(((qs[:, None, :] - df[None, :, :]) ** 2).sum(-1) + 1e-12)
        return float(d.min(dim=1).values.median())

    def retrieve(state, method):
        ranks = []
        for cid in patches:
            for qrole, prole in (("p1", "p2"), ("p2", "p1")):
                qp = patches[cid][qrole]
                scored = []
                for (dcid, drole, dp) in db:
                    if dcid == cid and drole == qrole:      # never retrieve own role
                        continue
                    if method == "learned":
                        sc = learned_score(qp["z_af3" if state == "af3" else "z_holo"], dp["z_holo"])
                        scored.append(((dcid, drole), -sc))  # higher better -> negate for ascending sort
                    else:
                        sc = frozen_score(qp["fs_af3" if state == "af3" else "fs_holo"], dp["fd_holo"])
                        scored.append(((dcid, drole), sc))
                scored.sort(key=lambda x: x[1])
                order = [k for k, _ in scored]
                true = (cid, prole)
                if true in order:
                    ranks.append(order.index(true) + 1)
        return ranks

    out = {"src": src, "n_complexes": len(recs), "db_size": len(db), "results": {}}
    for method in ("learned", "frozen"):
        for state in ("holo", "af3"):
            out["results"][f"{method}_{state}"] = metrics(retrieve(state, method))
    for method in ("learned", "frozen"):
        h, a = out["results"][f"{method}_holo"], out["results"][f"{method}_af3"]
        out["results"][f"{method}_af3_drop"] = {
            "top1": h["top1"] - a["top1"], "top5": h["top5"] - a["top5"], "mrr": h["mrr"] - a["mrr"]}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    print("=" * 74)
    print(f"RETRIEVAL  src={os.path.basename(src)}  DB={len(db)} chains  n={len(recs)}")
    print(f"{'method_state':18s} {'top1':>6} {'top5':>6} {'top10':>6} {'MRR':>6} {'medRank':>8}")
    for k in ("frozen_holo", "frozen_af3", "learned_holo", "learned_af3"):
        e = out["results"][k]
        print(f"{k:18s} {e['top1']:6.2f} {e['top5']:6.2f} {e['top10']:6.2f} {e['mrr']:6.2f} {e['median_rank']:8.0f}")
    for m in ("frozen", "learned"):
        d = out["results"][f"{m}_af3_drop"]
        print(f"  >>> {m} AF3-vs-holo drop: top1 {d['top1']:+.2f}  top5 {d['top5']:+.2f}  MRR {d['mrr']:+.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
