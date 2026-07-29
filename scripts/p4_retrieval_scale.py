"""Phase-4 M2 — retrieval at SCALE: does frozen's small-DB edge survive a large, hard decoy pool?

The §22 verdict (learned ~ frozen; frozen slightly ahead) was measured on a 36-chain DB. This asks
the decisive question: with a large pool of held-out decoy chains, does frozen's advantage shrink or
reverse — as it should if it comes from holo-overfitting that a robust encoder resists?

Design (all held-out; 0 train leak):
  - QUERIES: the 31 m2 eval complexes' chains (AF3 = apo-proxy, and holo for the drop reference).
  - DB     : every query chain's true holo partner + a large pool of decoy holo chains drawn from the
             OTHER held-out test complexes (none seen in training, none a query, no shared PDB stem).
  - PATCH  : DENSE holo interface (`pos`, median ~60 atoms) — pos_sc is too sparse (median 3) to give
             a stable large-DB metric. Query holo-vs-AF3 uses the intersection subset so the drop is exact.
  - CENTER : the checkpoint was trained with DC-offset centering, so we subtract the global mean over
             the whole embedded DB before normalizing (docs/10 §22) — the deployment-realistic offset.
  - SCORE  : learned = median-over-query-atoms of MAX bilinear zQ^T T zD (rank desc); frozen = median
             of MIN descriptor distance straight-vs-flipped (rank asc) — identical to §20/retrieval_af3.

Reports learned vs frozen and AF3-vs-holo drop, at the small DB (queries only) AND the large DB, so
small->large shows whether the gap moves.

Usage:
  python scripts/p4_retrieval_scale.py \
    --query-data logs/phase4/m2_npz --query-ids /work/upthomae/Meng/phase4/m2_eval_ids.txt \
    --decoy-data logs/phase4/m2_ret_scale/npz --decoy-ids logs/phase4/m2_ret_scale/decoy_built.txt \
    --ckpt /work/upthomae/Meng/phase4/ret_full_ctr_best.pt --center --min-pos 8 \
    --out logs/phase4/m2_ret_scale/scale_result.json --device cpu
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from masif_graph.p4.eval_af3 import Rec, build_encoder, normalize  # noqa: E402


def _uniq(a):
    return np.unique(a) if len(a) else np.zeros(0, dtype=np.int64)


def metrics(ranks):
    r = np.array(ranks, float)
    if len(r) == 0:
        return {"n": 0}
    return {"n": len(r), "top1": float(np.mean(r <= 1)), "top5": float(np.mean(r <= 5)),
            "top10": float(np.mean(r <= 10)), "mrr": float(np.mean(1.0 / r)),
            "median_rank": float(np.median(r))}


@torch.no_grad()
def run(args):
    device = args.device
    q_ids = [l.strip() for l in open(args.query_ids) if l.strip() and not l.startswith("#")]
    q_stems = {c[:4] for c in q_ids}

    # ---- load query complexes (need holo + af3 + dense contacts) ----
    q_recs = []
    for cid in q_ids:
        r = Rec(args.query_data, cid, args.pos_key, device)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos:
            q_recs.append(r)
    # ---- load decoy complexes (holo + contacts only; both chains need an interface) ----
    md = args.min_pos_decoy if args.min_pos_decoy is not None else args.min_pos
    d_ids = [l.strip() for l in open(args.decoy_ids) if l.strip() and not l.startswith("#")]
    d_recs = []
    for cid in d_ids:
        if cid[:4] in q_stems:                       # drop any shared-PDB-stem decoy (near-dup guard)
            continue
        r = Rec(args.decoy_data, cid, args.pos_key, device)
        if not r.ok:
            continue
        if len(_uniq(r.pos[:, 0])) >= md and len(_uniq(r.pos[:, 1])) >= md:
            d_recs.append(r)
    print(f"queries: {len(q_recs)} complexes | decoys: {len(d_recs)} complexes "
          f"(DB chains = {2*(len(q_recs)+len(d_recs))})", flush=True)

    enc, comp, src = build_encoder(q_recs, args.ckpt, device)
    enc.eval()
    T = comp.T

    # ---- encode every chain RAW (holo for all; af3 for queries), then DC-offset center globally ----
    raw = {}   # cid -> {'h1','h2', optionally 'a1','a2'}  (raw, unnormalized)
    allrows = []
    for r in q_recs + d_recs:
        e = {"h1": enc(r.hg1), "h2": enc(r.hg2)}
        if getattr(r, "has_af3", False):
            e["a1"] = enc(r.ag1); e["a2"] = enc(r.ag2)
        raw[r.cid] = e
        allrows.append(torch.cat(list(e.values()), 0))
    if args.center:
        mu = torch.cat(allrows, 0).mean(0, keepdim=True)
        emb = {cid: {k: normalize(v - mu) for k, v in e.items()} for cid, e in raw.items()}
    else:
        emb = {cid: {k: normalize(v) for k, v in e.items()} for cid, e in raw.items()}

    # ---- DB: every chain's DENSE holo interface patch (learned z + frozen flipped desc) ----
    def holo_patch(r, pid):
        col = 0 if pid == "p1" else 1
        I = _uniq(r.pos[:, col])
        g = r.hg1 if pid == "p1" else r.hg2
        z = emb[r.cid]["h1" if pid == "p1" else "h2"][I]
        return {"z": z, "df": g["desc_flipped"][I]}

    db = []
    for r in q_recs + d_recs:
        for pid in ("p1", "p2"):
            db.append((r.cid, pid, holo_patch(r, pid)))

    # ---- query patches for the 31: holo & af3 on the SAME (intersection) atoms so drop is exact ----
    qpat = {}
    for r in q_recs:
        ih, jh = _uniq(r.inter[:, 0]), _uniq(r.inter[:, 1])   # holo rows (surviving in af3)
        ia, ja = _uniq(r.inter[:, 2]), _uniq(r.inter[:, 3])   # matching af3 rows
        e = emb[r.cid]
        qpat[r.cid] = {
            "p1": {"zh": e["h1"][ih], "za": e["a1"][ia],
                   "sh": r.hg1["desc_straight"][ih], "sa": r.ag1["desc_straight"][ia]},
            "p2": {"zh": e["h2"][jh], "za": e["a2"][ja],
                   "sh": r.hg2["desc_straight"][jh], "sa": r.ag2["desc_straight"][ja]},
        }

    def learned_score(zq, zd):
        if zq.shape[0] == 0 or zd.shape[0] == 0:
            return -1e9
        return float((zq @ T @ zd.t()).max(1).values.median())

    def frozen_score(qs, df):
        if qs.shape[0] == 0 or df.shape[0] == 0:
            return 1e9
        d = torch.sqrt(((qs[:, None, :] - df[None, :, :]) ** 2).sum(-1) + 1e-12)
        return float(d.min(1).values.median())

    def retrieve(state, method, db_subset):
        ranks = []
        for cid in qpat:
            for qrole, prole in (("p1", "p2"), ("p2", "p1")):
                qp = qpat[cid][qrole]
                scored = []
                for (dcid, drole, dp) in db_subset:
                    if dcid == cid and drole == qrole:
                        continue
                    if method == "learned":
                        sc = -learned_score(qp["za"] if state == "af3" else qp["zh"], dp["z"])
                    else:
                        sc = frozen_score(qp["sa"] if state == "af3" else qp["sh"], dp["df"])
                    scored.append(((dcid, drole), sc))
                scored.sort(key=lambda x: x[1])
                order = [k for k, _ in scored]
                if (cid, prole) in order:
                    ranks.append(order.index((cid, prole)) + 1)
        return ranks

    q_cids = {r.cid for r in q_recs}
    db_small = [e for e in db if e[0] in q_cids]     # queries-only DB (dense-protocol baseline)
    out = {"src": os.path.basename(src), "n_query": len(q_recs), "n_decoy": len(d_recs),
           "center": bool(args.center), "pos_key": args.pos_key, "min_pos": args.min_pos,
           "db_small": len(db_small), "db_large": len(db), "results": {}}
    for tag, dbs in (("small", db_small), ("large", db)):
        for method in ("frozen", "learned"):
            for state in ("holo", "af3"):
                out["results"][f"{tag}_{method}_{state}"] = metrics(retrieve(state, method, dbs))

    def show(tag):
        print(f"\n===== DB={tag.upper()} ({out['db_'+tag]} chains) =====", flush=True)
        print(f"{'method_state':20s} {'top1':>6} {'top5':>6} {'top10':>6} {'MRR':>6} {'medRank':>8}")
        for k in (f"{tag}_frozen_holo", f"{tag}_frozen_af3", f"{tag}_learned_holo", f"{tag}_learned_af3"):
            e = out["results"][k]
            print(f"{k:20s} {e.get('top1',0):6.2f} {e.get('top5',0):6.2f} {e.get('top10',0):6.2f} "
                  f"{e.get('mrr',0):6.2f} {e.get('median_rank',0):8.0f}")
        for m in ("frozen", "learned"):
            h, a = out["results"][f"{tag}_{m}_holo"], out["results"][f"{tag}_{m}_af3"]
            print(f"  >>> {m} holo->AF3 drop: top5 {h.get('top5',0)-a.get('top5',0):+.3f}  "
                  f"MRR {h.get('mrr',0)-a.get('mrr',0):+.3f}")

    show("small"); show("large")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-data", required=True)
    ap.add_argument("--query-ids", required=True)
    ap.add_argument("--decoy-data", required=True)
    ap.add_argument("--decoy-ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos",
                    help="interface patch definition: dense 'pos' or sc-gated 'pos_sc' (MaSIF-native)")
    ap.add_argument("--min-pos", type=int, default=8, help="min interface atoms for QUERY chains")
    ap.add_argument("--min-pos-decoy", type=int, default=None,
                    help="min interface atoms for DECOY chains (default: --min-pos)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
