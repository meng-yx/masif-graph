"""Phase-5 gate: the 4-cell (query-state x DB-state) binder-retrieval benchmark (VECTORIZED).

For each eval complex we hold BOTH partners' interface patches in BOTH states (holo & AF3-apo) on the
IDENTICAL intersection atom set, so learned-vs-frozen and holo-vs-AF3 are exact. DB = every eval chain
(both roles). A query chain must rank its true partner near the top.

Cells (query x DB) x {learned, frozen}: HH holo->holo (floor), AH af3->holo, HA holo->af3,
AA af3->af3 (deployment headline). Robustness = degradation from HH. Controls: shuffled-partner (chance),
DC-offset centering (--center, mandatory for learned; else collapse, doc10 §24), frozen on identical patches.

Scoring (vectorized): learned S(q,d)=median_i max_j (z_qi^T T z_dj); frozen=median_i min_j ||ds_qi - df_dj||.
Segment max/min over DB atoms via scatter_reduce -> one matmul + one scatter per query.

Usage: python -m masif_graph.p5.retrieval_bench --data <npz_dir> --ids <ids> \
   --ckpt /work/upthomae/Meng/phase4/ret_full_ctr_best.pt --center --pos-key pos --out logs/phase5/gate.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
from masif_graph.p4.eval_af3 import Rec, build_encoder, normalize  # noqa: E402

CELLS = {"HH": ("holo", "holo"), "AH": ("af3", "holo"), "HA": ("holo", "af3"), "AA": ("af3", "af3")}


def _U(a): return np.unique(a) if len(a) else np.zeros(0, np.int64)


def metrics(ranks):
    r = np.array(ranks, float)
    if len(r) == 0: return {"n": 0}
    return {"n": len(r), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "top10": float((r <= 10).mean()), "mrr": float((1/r).mean()),
            "median_rank": float(np.median(r)), "ranks": [int(x) for x in r]}


@torch.no_grad()
def run(args):
    dev = args.device
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        r = Rec(args.data, cid, args.pos_key, dev)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos:
            recs.append(r)
    print(f"usable complexes (holo+af3+>= {args.min_pos} '{args.pos_key}'): {len(recs)}/{len(ids)}"
          f"  -> DB {2*len(recs)} chains", flush=True)
    if len(recs) < 5:
        print("TOO FEW; abort"); return None

    enc, comp, src = build_encoder(recs, args.ckpt, dev); enc.eval()
    T = comp.T
    raw, allrows = {}, []
    for r in recs:
        e = {"h1": enc(r.hg1), "h2": enc(r.hg2), "a1": enc(r.ag1), "a2": enc(r.ag2)}
        raw[r.cid] = e; allrows.append(torch.cat(list(e.values()), 0))
    mu = torch.cat(allrows, 0).mean(0, keepdim=True) if args.center else 0.0
    emb = {c: {k: normalize(v - mu) for k, v in e.items()} for c, e in raw.items()}
    zstd = float(torch.cat([torch.cat(list(e.values()), 0) for e in emb.values()], 0).std(0).mean())
    print(f"z_std(post-center)={zstd:.4f}", flush=True)

    # patch per chain/role/state on the intersection atom set (identical across states)
    mp = args.max_patch
    rng0 = np.random.default_rng(0)
    pat = {}
    for r in recs:
        inter = r.inter
        if mp and len(inter) > mp:
            inter = inter[rng0.choice(len(inter), mp, replace=False)]
        ih, jh, ia, ja = _U(inter[:, 0]), _U(inter[:, 1]), _U(inter[:, 2]), _U(inter[:, 3])
        e = emb[r.cid]
        pat[r.cid] = {
            "p1": {"holo": {"z": e["h1"][ih], "ds": r.hg1["desc_straight"][ih], "df": r.hg1["desc_flipped"][ih]},
                   "af3":  {"z": e["a1"][ia], "ds": r.ag1["desc_straight"][ia], "df": r.ag1["desc_flipped"][ia]}},
            "p2": {"holo": {"z": e["h2"][jh], "ds": r.hg2["desc_straight"][jh], "df": r.hg2["desc_flipped"][jh]},
                   "af3":  {"z": e["a2"][ja], "ds": r.ag2["desc_straight"][ja], "df": r.ag2["desc_flipped"][ja]}},
        }
    dbk = [(c, role) for c in pat for role in ("p1", "p2")]
    idx_of = {k: i for i, k in enumerate(dbk)}
    n_db = len(dbk)

    def build_db(dstate, method):
        mats, seg = [], []
        for i, (c, r) in enumerate(dbk):
            p = pat[c][r][dstate]
            m = p["z"] if method == "learned" else p["df"]
            mats.append(m); seg.append(torch.full((m.shape[0],), i, dtype=torch.long))
        return torch.cat(mats, 0), torch.cat(seg, 0)

    def retrieve(qstate, dstate, method, shuffle=False):
        Mdb, seg = build_db(dstate, method)           # (Ntot,dim), (Ntot,)
        TZ = (T @ Mdb.t()) if method == "learned" else None   # (d,Ntot)
        rng = np.random.default_rng(0)
        ranks = []
        for (cid, qr) in dbk:
            pr = "p2" if qr == "p1" else "p1"
            qp = pat[cid][qr][qstate]
            q = qp["z"] if method == "learned" else qp["ds"]
            if q.shape[0] == 0:
                continue
            if method == "learned":
                Mq = q @ TZ                                   # (nq,Ntot) higher=better
                S = torch.full((q.shape[0], n_db), float("-inf"))
                S.scatter_reduce_(1, seg.expand(q.shape[0], -1), Mq, reduce="amax", include_self=True)
                score = S.median(0).values                    # (n_db,) higher=better
                score[idx_of[(cid, qr)]] = float("-inf")
                order = torch.argsort(score, descending=True).tolist()
            else:
                dist = torch.cdist(q, Mdb)                     # (nq,Ntot) lower=better
                S = torch.full((q.shape[0], n_db), float("inf"))
                S.scatter_reduce_(1, seg.expand(q.shape[0], -1), dist, reduce="amin", include_self=True)
                score = S.median(0).values
                score[idx_of[(cid, qr)]] = float("inf")
                order = torch.argsort(score, descending=False).tolist()
            true = idx_of[(cid, pr)]
            if shuffle:
                true = order[rng.integers(len(order))]
            ranks.append(order.index(true) + 1)
        return ranks

    out = {"src": os.path.basename(src), "n": len(recs), "db_chains": n_db,
           "center": bool(args.center), "pos_key": args.pos_key, "max_patch": args.max_patch, "z_std": zstd, "results": {}}
    for cell, (qs, ds) in CELLS.items():
        for method in ("frozen", "learned"):
            out["results"][f"{cell}_{method}"] = metrics(retrieve(qs, ds, method))
            print(f"  done {cell}_{method}", flush=True)
    for method in ("frozen", "learned"):
        out["results"][f"HH_{method}_shuffled"] = metrics(retrieve("holo", "holo", method, shuffle=True))

    out["robustness"] = {}
    for method in ("frozen", "learned"):
        hh = out["results"][f"HH_{method}"]
        for cell in ("AH", "HA", "AA"):
            c = out["results"][f"{cell}_{method}"]
            out["robustness"][f"{method}_{cell}_drop"] = {
                "top5": round(hh.get("top5", 0) - c.get("top5", 0), 4),
                "mrr": round(hh.get("mrr", 0) - c.get("mrr", 0), 4)}

    print("=" * 78)
    print(f"PHASE-5 GATE  src={out['src']}  DB={n_db}  n={out['n']}  patch={args.pos_key}  center={args.center}")
    print(f"{'cell_method':16s} {'top1':>6} {'top5':>6} {'top10':>6} {'MRR':>6} {'medRank':>8}")
    for cell in ("HH", "AH", "HA", "AA"):
        for method in ("frozen", "learned"):
            e = out["results"][f"{cell}_{method}"]
            print(f"{cell+'_'+method:16s} {e.get('top1',0):6.2f} {e.get('top5',0):6.2f} "
                  f"{e.get('top10',0):6.2f} {e.get('mrr',0):6.2f} {e.get('median_rank',0):8.0f}")
    print(f"-- shuffled control (chance top5~{5.0/n_db:.3f}) --")
    for method in ("frozen", "learned"):
        e = out["results"][f"HH_{method}_shuffled"]
        print(f"  HH_{method}_shuffled top5={e.get('top5',0):.3f} medRank={e.get('median_rank',0):.0f}")
    print("-- robustness (HH-AF3cell; smaller=more robust) --")
    for method in ("frozen", "learned"):
        for cell in ("AH", "HA", "AA"):
            d = out["robustness"][f"{method}_{cell}_drop"]
            print(f"  {method:7s} {cell}: top5 {d['top5']:+.3f}  MRR {d['mrr']:+.3f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--center", action="store_true")
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--min-pos", type=int, default=8); ap.add_argument("--out", default=None)
    ap.add_argument("--max-patch", type=int, default=128, help="cap interface atoms/chain (memory)")
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
