"""Phase-8 A1 — is the Stage-1 encoder modelling sidechains, or ignoring them?

Why this matters (docs/23 §6, docs/24 §3): Phases 4-7 found the encoder is conformation-robust —
swapping a crystal chain for an AF3 model costs almost nothing in retrieval. That is the project's
headline, and it has an unflattering alternative explanation: the encoder may be robust to
sidechains because it never reads them. If so, the north star ("learn per atom how much mismatch is
tolerable") is unreachable with this encoder and D8-14 (apo-substituted training) is promoted from
augmentation to prerequisite.

Two measurements on the SAME ablation ladder (`p8.ablate`), both row-matched:

  (1) embedding displacement — how far does z move when a channel is destroyed? Reported
      separately for surface atoms that are sidechain atoms vs backbone atoms, because a
      sidechain-blind encoder should barely move even its own sidechain rows.
  (2) retrieval — the Phase-5 gate (287-clean, --center) rerun on ablated graphs.

Read the two together. Displacement says whether the information reaches the embedding at all;
retrieval says whether the part that reaches it is the part the task uses.

Positive control: `all_feat` destroys every node feature in the graph. If retrieval survives it,
this harness is not measuring what it claims and NO conclusion may be drawn from the other rows.

Usage:
  python -m masif_graph.p8.a1_sidechain --data <npz_eval> --ids <ids> --ckpt <ret_ppionly*.pt> \
      --tag ppionly_s0 --out-dir logs/phase8A/a1
"""
from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch

from masif_graph.p4.eval_af3 import Rec, build_encoder
from masif_graph.p5 import retrieval_bench
from masif_graph.p8 import ablate

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def _q(a):
    """Distribution, not just a mean — per the guardrails."""
    a = np.asarray(a, float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95))}


@torch.no_grad()
def displacement(args):
    """Per-surface-atom embedding displacement under each ablation, split by atom class."""
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids[: args.max_complexes]:
        r = Rec(args.data, cid, args.pos_key, args.device)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos:
            recs.append(r)
    if not recs:
        raise SystemExit("no usable complexes for the displacement probe")
    enc, _, src = build_encoder(recs, args.ckpt, args.device)
    enc.eval()

    # baseline embeddings + the class of each surface row, per (complex, chain)
    base, is_sc, flex = {}, {}, {}
    for r in recs:
        for attr in ("hg1", "hg2"):
            g = getattr(r, attr)
            k = (r.cid, attr)
            base[k] = enc(g)
            sidx = g["surf_node_idx"]
            is_sc[k] = ablate.sidechain_mask(g)[sidx].cpu().numpy()
            flex[k] = g["atom_feat"][sidx, ablate.COL_FLEX_DEPTH].cpu().numpy()

    # scale: the natural spread of z across surface atoms, so displacement is dimensionless
    allz = torch.cat([v for v in base.values()], 0)
    scale = float((allz - allz.mean(0, keepdim=True)).norm(dim=1).mean())

    out = {"src": os.path.basename(src), "n_complexes": len(recs), "z_scale": scale, "ablations": {}}
    for kind in ablate.ABLATIONS:
        if kind == "none":
            continue
        rel_sc, rel_bb, cos_sc, cos_bb, flex_rows, rel_rows = [], [], [], [], [], []
        for r in recs:
            for i, attr in enumerate(("hg1", "hg2")):
                g = getattr(r, attr)
                k = (r.cid, attr)
                za = enc(ablate.ablate_graph(g, kind, seed=args.seed * 977 + i))
                z0 = base[k]
                d = (z0 - za).norm(dim=1).cpu().numpy() / max(scale, 1e-12)
                c = torch.nn.functional.cosine_similarity(z0, za, dim=1).cpu().numpy()
                m = is_sc[k].astype(bool)
                rel_sc.append(d[m]); rel_bb.append(d[~m])
                cos_sc.append(c[m]); cos_bb.append(c[~m])
                if kind == "sc_all":
                    flex_rows.append(flex[k]); rel_rows.append(d)
        e = {"rel_disp_sidechain_rows": _q(np.concatenate(rel_sc) if rel_sc else []),
             "rel_disp_backbone_rows": _q(np.concatenate(rel_bb) if rel_bb else []),
             "cos_sidechain_rows": _q(np.concatenate(cos_sc) if cos_sc else []),
             "cos_backbone_rows": _q(np.concatenate(cos_bb) if cos_bb else [])}
        if kind == "sc_all" and flex_rows:
            # does the encoder move MORE where the structure is more flexible? the closest thing
            # we have to an implicit per-atom sigma (D8-9 / D8-19).
            f = np.concatenate(flex_rows); d = np.concatenate(rel_rows)
            keep = np.isfinite(f) & np.isfinite(d)
            if keep.sum() > 10 and np.std(f[keep]) > 0 and np.std(d[keep]) > 0:
                from scipy.stats import spearmanr
                rho, p = spearmanr(f[keep], d[keep])
                e["spearman_disp_vs_flex_depth"] = {"rho": float(rho), "p": float(p),
                                                    "n": int(keep.sum())}
        out["ablations"][kind] = e
        print(f"  disp {kind:10s} sc_med={e['rel_disp_sidechain_rows'].get('median', float('nan')):.4f} "
              f"bb_med={e['rel_disp_backbone_rows'].get('median', float('nan')):.4f}", flush=True)

    # what each ablation actually removed, on the first chain (a no-op ablation must be visible)
    g0 = recs[0].hg1
    out["ablation_effect_example"] = {k: ablate.ablation_stats(g0, k, seed=args.seed)
                                      for k in ablate.ABLATIONS if k != "none"}
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
    ap.add_argument("--max-complexes", type=int, default=60,
                    help="displacement probe only; retrieval always uses the full id list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="comma-separated ablation subset")
    ap.add_argument("--skip-displacement", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_displacement:
        print(f"=== A1 displacement probe ({args.tag}) ===", flush=True)
        d = displacement(args)
        p = os.path.join(args.out_dir, f"disp_{args.tag}.json")
        json.dump(d, open(p, "w"), indent=2)
        print(f"wrote {p}", flush=True)

    kinds = args.only.split(",") if args.only else list(ablate.ABLATIONS)
    for kind in kinds:
        p = os.path.join(args.out_dir, f"ret_{args.tag}_{kind}.json")
        if os.path.exists(p):
            print(f"=== A1 retrieval {kind}: exists, skip ===", flush=True)
            continue
        print(f"=== A1 retrieval under ablation={kind} ({args.tag}) ===", flush=True)
        sub = Namespace(data=args.data, ids=args.ids, ckpt=args.ckpt, center=True,
                        pos_key=args.pos_key, min_pos=args.min_pos, out=p,
                        max_patch=args.max_patch, device=args.device)
        tr = None if kind == "none" else (lambda r, k=kind: ablate.ablate_rec(r, k, seed=args.seed))
        retrieval_bench.run(sub, transform=tr)


if __name__ == "__main__":
    main()
