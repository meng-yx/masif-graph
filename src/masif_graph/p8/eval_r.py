"""Stage R / R4 — the pre-registered gate (docs/26 §4).

Three checks on a trained Stage-R checkpoint, all on the held-out 287-clean eval set:

  PRIMARY   median top-1 spatial error   < 5 A          (Stage-A baseline: 19.4 A)
  DO-NO-HARM chain retrieval HH top-5    >= 0.644 - 0.02 (Phase-7 §2 for the PPI-only models)
  INVARIANCE end-to-end rotation maxdiff ~ float epsilon (the Phase-4 M0 property must survive R1)

The primary metric is deliberately not the training loss and not chain retrieval: Stage A showed
retrieval can be 0.644 while the top-1 partner is 19.4 A away, so retrieval cannot detect the
failure Stage R exists to fix.

Rotation is checked END TO END, not just on the context block: R1 computes context from `coord` at
load time, so the composition (context -> encoder) is what must be invariant, and checking only the
context features would leave that unverified.

Usage:
  MASIF_GLOBAL_CTX=1 python -m masif_graph.p8.eval_r --data <npz_eval> --ids <ids> \
      --ckpt <r.pt> --tag r_s0 --out logs/phase8R/gate_r_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch

from masif_graph.p4.eval_af3 import Rec, build_encoder
from masif_graph.p4.objective import normalize
from masif_graph.p5 import retrieval_bench
from masif_graph.p8.context import attach_context, context_features
from masif_graph.p8.objective_r import DustbinScore, top1_spatial_error

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

GATE_SPATIAL, GATE_RETRIEVAL_DROP, BASELINE_RETRIEVAL = 5.0, 0.02, 0.644


@torch.no_grad()
def rotation_end_to_end(enc, g, seed=0):
    """Rotate the chain, recompute global context from the rotated coords, re-encode, compare."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = rng.uniform(-40, 40, size=3)
    z0 = enc(g)
    n_ctx = 0
    base = dict(g)
    if g["atom_feat"].shape[1] > 26:                    # strip the context block, then rebuild it
        n_ctx = g["atom_feat"].shape[1] - 26
        base["atom_feat"] = g["atom_feat"][:, :26]
    rc = torch.as_tensor(g["coord"].cpu().numpy() @ q.T + t, dtype=g["coord"].dtype)
    base["coord"] = rc
    g_rot = attach_context(base) if n_ctx else base
    return float((z0 - enc(g_rot)).abs().max()), n_ctx


@torch.no_grad()
def spatial_gate(args, ids):
    recs = []
    for cid in ids:
        r = Rec(args.data, cid, args.pos_key, args.device)
        if r.ok and len(r.pos) >= args.min_pos:
            recs.append(r)
        if args.max_n and len(recs) >= args.max_n:
            break
    if not recs:
        raise SystemExit("no usable complexes")
    enc, comp, src = build_encoder(recs, args.ckpt, args.device)
    enc.eval()
    scorer = DustbinScore(comp)
    ck = torch.load(args.ckpt, map_location=args.device)
    if "bin_score" in ck:
        scorer.bin_score.data = torch.as_tensor(ck["bin_score"])

    errs, per = [], []
    for r in recs:
        z1, z2 = normalize(enc(r.hg1)), normalize(enc(r.hg2))
        P = torch.as_tensor(r.pos, dtype=torch.long, device=args.device)
        e = top1_spatial_error(z1, z2, P, r.hg2["coord"], scorer)
        if e.numel():
            errs.append(e.cpu().numpy())
            per.append({"cid": r.cid, "median": float(np.median(e.cpu().numpy())),
                        "n_queries": int(e.numel())})
    a = np.concatenate(errs)
    rot, n_ctx = rotation_end_to_end(enc, recs[0].hg1)
    ctx_diag = context_features(recs[0].hg1["coord"].cpu().numpy())[1]
    return {
        "src": os.path.basename(src), "n_complexes": len(recs), "n_queries": int(a.size),
        "n_context_dims": n_ctx,
        "top1_spatial_median": float(np.median(a)),
        "top1_spatial_mean": float(a.mean()),
        "top1_spatial_p25": float(np.percentile(a, 25)),
        "top1_spatial_p75": float(np.percentile(a, 75)),
        "frac_within_5A": float((a < 5).mean()),
        "frac_within_10A": float((a < 10).mean()),
        "rotation_maxdiff_end_to_end": rot,
        "context_eig_gaps": {k: v for k, v in ctx_diag.items() if k != "degenerate"},
        "per_complex": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=120)
    ap.add_argument("--max-patch", type=int, default=128)
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    out = {"tag": args.tag, "gate": {"spatial_below": GATE_SPATIAL,
                                     "retrieval_at_least": BASELINE_RETRIEVAL - GATE_RETRIEVAL_DROP,
                                     "pre_registered": "docs/26-phase8R-design.md §4"}}
    out["spatial"] = spatial_gate(args, ids)

    if not args.skip_retrieval:
        rp = args.out.replace(".json", "_retrieval.json")
        sub = Namespace(data=args.data, ids=args.ids, ckpt=args.ckpt, center=True, pos_key="pos",
                        min_pos=args.min_pos, out=rp, max_patch=args.max_patch, device=args.device)
        r = retrieval_bench.run(sub)
        if r:
            out["retrieval"] = {"HH_top5": r["results"]["HH_learned"]["top5"],
                                "AA_top5": r["results"]["AA_learned"]["top5"],
                                "HH_median_rank": r["results"]["HH_learned"]["median_rank"],
                                "n": r["n"], "db_chains": r["db_chains"]}

    s = out["spatial"]
    checks = {
        "spatial_below_5A": s["top1_spatial_median"] < GATE_SPATIAL,
        "rotation_invariant": s["rotation_maxdiff_end_to_end"] < 1e-3,
    }
    if "retrieval" in out:
        checks["retrieval_preserved"] = (out["retrieval"]["HH_top5"]
                                         >= BASELINE_RETRIEVAL - GATE_RETRIEVAL_DROP)
    out["checks"] = checks
    out["GATE_PASSED"] = all(checks.values())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("=" * 78)
    print(f"R4 GATE  {args.tag}  src={s['src']}  n={s['n_complexes']} complexes, "
          f"{s['n_queries']} queries, ctx dims={s['n_context_dims']}")
    print(f"  top-1 spatial error   median {s['top1_spatial_median']:7.2f} A   "
          f"[p25 {s['top1_spatial_p25']:.2f}, p75 {s['top1_spatial_p75']:.2f}]   "
          f"(baseline 19.4, gate < {GATE_SPATIAL})")
    print(f"  within 5 A / 10 A     {s['frac_within_5A']:.3f} / {s['frac_within_10A']:.3f}")
    print(f"  rotation maxdiff      {s['rotation_maxdiff_end_to_end']:.3e}")
    if "retrieval" in out:
        print(f"  chain retrieval HH    {out['retrieval']['HH_top5']:.3f} "
              f"(gate >= {BASELINE_RETRIEVAL - GATE_RETRIEVAL_DROP:.3f})   "
              f"AA {out['retrieval']['AA_top5']:.3f}")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"  >>> GATE {'PASSED' if out['GATE_PASSED'] else 'NOT PASSED'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
