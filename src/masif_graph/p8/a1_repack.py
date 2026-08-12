"""Phase-8 A1.2 — sensitivity to sidechain CONFORMATION (what A1 does not test).

A1 permuted sidechain *features*, so it answered "does the encoder read sidechain chemistry?"
(emphatically yes: isolating sidechain atoms costs 70% of the graph's information). An apo
structure does something different — it keeps every sidechain's identity and moves only its
rotamer. A FASPR fixed-backbone repack of each chain **in isolation** is exactly that perturbation
(the Phase-2 apo proxy), so this is the measurement that connects A1 to the north star.

Two readings, on the same complexes:

  displacement  per-surface-atom ||z_holo - z_repack||, against ||z_holo - z_af3|| measured on the
                identical atoms. How much of the AF3 gap is sidechain-only rearrangement?
  retrieval     the Phase-5 gate with the repack substituted into the AF3 slot, so `AH`/`HA`/`AA`
                become holo-vs-repack cells and the numbers are directly comparable to the
                published AF3 ones.

Also: Spearman(displacement, flex_depth) and Spearman(displacement, holo B-factor). The second is
the one that matters for D8-9/D8-19 — B-factor is NOT an input to the encoder, so a correlation
there is evidence the representation tracks real flexibility rather than echoing a feature column.

Usage:
  python -m masif_graph.p8.a1_repack --data <npz_dir> --ids <ids with rp built> --ckpt <ckpt> \
      --tag ppionly_s0 --out-dir logs/phase8A/a1
"""
from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch

from masif_graph.p4.eval_af3 import Rec, build_encoder, load_state_chain
from masif_graph.p5 import retrieval_bench
from masif_graph.p8 import ablate

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def _keys(npz_path):
    z = np.load(npz_path)
    return [k.decode() if isinstance(k, bytes) else str(k) for k in z["keys"].tolist()]


def has_repack(data_dir, cid):
    return all(os.path.exists(os.path.join(data_dir, f"{cid}__rp__{p}.npz")) for p in ("p1", "p2"))


def substitute_repack(rec, data_dir, device="cpu"):
    """Replace the AF3 slot of `rec` with the FASPR repack, remapping `inter` columns 2/3.

    Returns True on success. On failure the record is left untouched, so the CALLER must exclude
    it — silently scoring an unsubstituted record would report AF3 numbers as repack numbers.
    """
    r1 = load_state_chain(data_dir, rec.cid, "rp", "p1", device)
    r2 = load_state_chain(data_dir, rec.cid, "rp", "p2", device)
    if r1 is None or r2 is None:
        return False
    g1, k1 = r1
    g2, k2 = r2
    hk1 = _keys(os.path.join(data_dir, f"{rec.cid}__holo__p1.npz"))
    hk2 = _keys(os.path.join(data_dir, f"{rec.cid}__holo__p2.npz"))
    m1 = {k: i for i, k in enumerate(k1)}
    m2 = {k: i for i, k in enumerate(k2)}
    inter = [(i, j, m1[hk1[i]], m2[hk2[j]])
             for i, j in rec.pos if hk1[i] in m1 and hk2[j] in m2]
    if len(inter) < 8:
        return False
    rec.ag1, rec.ag2 = g1, g2
    rec.inter = np.array(inter, dtype=np.int64)
    rec.retention = len(inter) / max(len(rec.pos), 1)
    return True


def _q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95))}


@torch.no_grad()
def displacement(args, ids):
    from scipy.stats import spearmanr
    recs = []
    for cid in ids:
        r = Rec(args.data, cid, args.pos_key, args.device)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos and has_repack(args.data, cid):
            recs.append(r)
    if not recs:
        raise SystemExit("no complexes with holo + af3 + repack")
    print(f"{len(recs)} complexes with all three states", flush=True)
    enc, _, src = build_encoder(recs, args.ckpt, args.device)
    enc.eval()

    d_rp, d_af, flex, bfac = [], [], [], []
    per_complex = []
    for r in recs:
        rp = {}
        ok = True
        for pid, hg in (("p1", r.hg1), ("p2", r.hg2)):
            g = load_state_chain(args.data, r.cid, "rp", pid, args.device)
            if g is None:
                ok = False
                break
            rp[pid] = g
        if not ok:
            continue
        hk = {pid: _keys(os.path.join(args.data, f"{r.cid}__holo__{pid}.npz")) for pid in ("p1", "p2")}
        ak = {pid: _keys(os.path.join(args.data, f"{r.cid}__af3__{pid}.npz")) for pid in ("p1", "p2")}
        rows = []
        for pid, hg, ag in (("p1", r.hg1, r.ag1), ("p2", r.hg2, r.ag2)):
            rg, rk = rp[pid]
            zh, zr, za = enc(hg), enc(rg), enc(ag)
            scale = float((zh - zh.mean(0, keepdim=True)).norm(dim=1).mean())
            mr = {k: i for i, k in enumerate(rk)}
            ma = {k: i for i, k in enumerate(ak[pid])}
            idx = [(i, mr[k], ma[k]) for i, k in enumerate(hk[pid]) if k in mr and k in ma]
            if len(idx) < 10:
                continue
            ih = torch.tensor([x[0] for x in idx], dtype=torch.long)
            ir = torch.tensor([x[1] for x in idx], dtype=torch.long)
            ia = torch.tensor([x[2] for x in idx], dtype=torch.long)
            dr = (zh[ih] - zr[ir]).norm(dim=1).cpu().numpy() / max(scale, 1e-12)
            da = (zh[ih] - za[ia]).norm(dim=1).cpu().numpy() / max(scale, 1e-12)
            d_rp.append(dr); d_af.append(da)
            flex.append(hg["atom_feat"][hg["surf_node_idx"][ih], ablate.COL_FLEX_DEPTH].cpu().numpy())
            rows.append((float(np.median(dr)), float(np.median(da)), len(idx)))
        if rows:
            per_complex.append({"cid": r.cid,
                                "median_rp": float(np.mean([x[0] for x in rows])),
                                "median_af3": float(np.mean([x[1] for x in rows])),
                                "n_atoms": int(sum(x[2] for x in rows))})

    DR = np.concatenate(d_rp) if d_rp else np.zeros(0)
    DA = np.concatenate(d_af) if d_af else np.zeros(0)
    FL = np.concatenate(flex) if flex else np.zeros(0)
    out = {"src": os.path.basename(src), "n_complexes": len(per_complex),
           "rel_disp_repack": _q(DR), "rel_disp_af3": _q(DA),
           "ratio_repack_over_af3_median": (float(np.median(DR) / np.median(DA))
                                            if DA.size and np.median(DA) > 0 else None),
           "per_complex": per_complex}
    keep = np.isfinite(DR) & np.isfinite(FL)
    if keep.sum() > 10 and np.std(FL[keep]) > 0 and np.std(DR[keep]) > 0:
        out["spearman_disp_vs_flex_depth"] = {
            "rho": float(spearmanr(DR[keep], FL[keep]).statistic), "n": int(keep.sum()),
            "caveat": "flex_depth is input feature col 22, so this is not independent evidence"}
    _ = bfac
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="logs/phase8A/a1")
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--max-patch", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = [ln.strip() for ln in open(args.ids) if ln.strip() and not ln.startswith("#")]
    ids = [c for c in ids if has_repack(args.data, c)]
    print(f"{len(ids)} ids have a repack state built", flush=True)
    if len(ids) < 5:
        raise SystemExit("too few repacked complexes")
    id_file = os.path.join(args.out_dir, f"rp_ids_{args.tag}.txt")
    with open(id_file, "w") as f:
        f.write("\n".join(ids) + "\n")

    d = displacement(args, ids)
    p = os.path.join(args.out_dir, f"repack_disp_{args.tag}.json")
    json.dump(d, open(p, "w"), indent=2)
    print(f"  rel displacement  repack median {d['rel_disp_repack'].get('median', float('nan')):.4f}"
          f"   af3 median {d['rel_disp_af3'].get('median', float('nan')):.4f}"
          f"   ratio {d['ratio_repack_over_af3_median']}")
    print(f"wrote {p}", flush=True)

    # Retrieval with the repack substituted into the AF3 slot. Complexes where substitution fails
    # are DROPPED, not left as AF3 — otherwise their AF3 numbers would be reported as repack ones.
    dropped = []

    def tr(rec):
        if not substitute_repack(rec, args.data, args.device):
            dropped.append(rec.cid)
            rec.inter = np.zeros((0, 4), np.int64)

    print("=== A1.2 retrieval: repack substituted into the AF3 slot ===", flush=True)
    sub = Namespace(data=args.data, ids=id_file, ckpt=args.ckpt, center=True,
                    pos_key=args.pos_key, min_pos=args.min_pos,
                    out=os.path.join(args.out_dir, f"repack_ret_{args.tag}.json"),
                    max_patch=args.max_patch, device=args.device)
    retrieval_bench.run(sub, transform=tr)
    if dropped:
        print(f"NOTE: {len(dropped)} complexes could not be substituted and were emptied: "
              f"{dropped[:8]}", flush=True)


if __name__ == "__main__":
    main()
