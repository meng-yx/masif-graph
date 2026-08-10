"""Phase-7 S5 — holo→AF3-apo robustness on the LIGAND axis (the north star, applied to protein-ligand).

Phase 5 measured conformational robustness for protein–protein. This is the protein–ligand analogue.
Because **a ligand has no apo state** (project rule: the protein varies, the ligand stays at its
experimental pose), the Phase-5 4-cell query×DB matrix collapses to four *role* cells rather than
four conformational ones:

    P(holo) -> ligand DB      "given this pocket, find its ligand"      the deployment question
    P(af3)  -> ligand DB      the same, from a predicted apo pocket     the robustness question
    ligand  -> P(holo) DB     "given this ligand, find its protein"
    ligand  -> P(af3) DB      the same against a predicted-protein DB

Robustness = the drop from the holo row to the AF3 row. Everything is centred together so all four
cells sit in one embedding space, and the shuffled-partner control gives the chance line.

The protein patch is the contact set *of that state*: holo contacts for the holo protein, contacts
recomputed against the same crystal ligand for the AF3 protein. Reusing holo rows would be wrong —
the AF3 surface has its own atom rows — and the change in the contact set is itself part of what is
being measured (`contact_ratio_af3_over_holo` in the build report).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from masif_graph.p4.dataset import D_AA, D_VV, D_VA, load_chain_graph
from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, normalize


def _u(a):
    return np.unique(a) if len(a) else np.zeros(0, np.int64)


def metrics(ranks, n_db):
    r = np.array(ranks, float)
    if len(r) == 0:
        return {"n": 0}
    return {"n": len(r), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "top10": float((r <= 10).mean()), "mrr": float((1 / r).mean()),
            "median_rank": float(np.median(r)), "db": int(n_db),
            "chance_top5": round(5.0 / max(n_db - 1, 1), 4)}


@torch.no_grad()
def run(args):
    dev = args.device
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs = []
    for cid in ids:
        f = {k: os.path.join(args.data, f"{cid}__{k}.npz")
             for k in ("holo__p1", "holo__p2", "contacts", "af3__p1", "af3contacts")}
        if not all(os.path.exists(p) for p in f.values()):
            continue
        ph = np.load(f["contacts"])["pos"].reshape(-1, 2)
        pa = np.load(f["af3contacts"])["pos"].reshape(-1, 2)
        if len(ph) < args.min_pos or len(pa) < args.min_pos:
            continue
        recs.append((cid, f, ph, pa))
    print(f"usable (holo+af3+>= {args.min_pos} contacts): {len(recs)}/{len(ids)}", flush=True)
    if len(recs) < 5:
        print("TOO FEW; abort")
        return None

    ck = torch.load(args.ckpt, map_location=dev)
    cfg = ck.get("cfg", {})
    probe = load_chain_graph(recs[0][1]["holo__p1"], dev)
    enc = HeteroEncoder(probe["atom_feat"].shape[1], 4, D_AA, D_VV, D_VA, d=cfg.get("d", 64),
                        d_out=cfg.get("d_out", 32), n_layers=cfg.get("layers", 4)).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()
    comp = Complementarity(cfg.get("d_out", 32)).to(dev)
    comp.load_state_dict(ck["comp"])

    raw, patches = {}, {}
    for cid, f, ph, pa in recs:
        zh = enc(load_chain_graph(f["holo__p1"], dev))
        za = enc(load_chain_graph(f["af3__p1"], dev))
        zl = enc(load_chain_graph(f["holo__p2"], dev))
        raw[(cid, "ph")], raw[(cid, "pa")], raw[(cid, "lig")] = zh, za, zl
        patches[cid] = {"ph": _u(ph[:, 0]), "pa": _u(pa[:, 0]), "lig": _u(ph[:, 1])}
    mu = torch.cat(list(raw.values()), 0).mean(0, keepdim=True) if args.center else 0.0
    emb = {k: normalize(v - mu) for k, v in raw.items()}
    z_std = float(torch.cat(list(emb.values()), 0).std(0).mean())
    print(f"z_std(post-center)={z_std:.4f}", flush=True)

    rng0 = np.random.default_rng(0)

    def pat(cid, role):
        idx = patches[cid][role]
        if args.max_patch and len(idx) > args.max_patch:
            idx = np.sort(rng0.choice(idx, args.max_patch, replace=False))
        z = emb[(cid, role)]
        return z[torch.as_tensor(idx, dtype=torch.long, device=z.device)]

    cids = [c for c, _f, _h, _a in recs]

    def retrieve(q_role, d_role, shuffle=False):
        mats, seg = [], []
        for i, c in enumerate(cids):
            m = pat(c, d_role)
            mats.append(m)
            seg.append(torch.full((m.shape[0],), i, dtype=torch.long, device=m.device))
        Mdb, seg_t = torch.cat(mats, 0), torch.cat(seg, 0)
        TZ = comp.T @ Mdb.t()
        rng = np.random.default_rng(0)
        ranks = []
        for i, c in enumerate(cids):
            q = pat(c, q_role)
            if q.shape[0] == 0:
                continue
            S = torch.full((q.shape[0], len(cids)), float("-inf"), device=q.device)
            S.scatter_reduce_(1, seg_t.expand(q.shape[0], -1), q @ TZ, reduce="amax",
                              include_self=True)
            order = torch.argsort(S.median(0).values, descending=True).tolist()
            true = rng.integers(len(cids)) if shuffle else i
            ranks.append(order.index(true) + 1)
        return metrics(ranks, len(cids))

    out = {"ckpt": os.path.basename(args.ckpt), "n": len(recs), "z_std": z_std,
           "center": bool(args.center), "results": {}}
    cells = {"Pholo_to_lig": ("ph", "lig"), "Paf3_to_lig": ("pa", "lig"),
             "lig_to_Pholo": ("lig", "ph"), "lig_to_Paf3": ("lig", "pa")}
    for name, (q, d) in cells.items():
        out["results"][name] = retrieve(q, d)
    out["results"]["Pholo_to_lig_shuffled"] = retrieve("ph", "lig", shuffle=True)
    a, b = out["results"]["Pholo_to_lig"], out["results"]["Paf3_to_lig"]
    c, d = out["results"]["lig_to_Pholo"], out["results"]["lig_to_Paf3"]
    out["robustness"] = {
        "protein_query_top5_drop": round(a.get("top5", 0) - b.get("top5", 0), 4),
        "protein_query_mrr_drop": round(a.get("mrr", 0) - b.get("mrr", 0), 4),
        "protein_db_top5_drop": round(c.get("top5", 0) - d.get("top5", 0), 4),
        "protein_db_mrr_drop": round(c.get("mrr", 0) - d.get("mrr", 0), 4)}

    print("=" * 74)
    print(f"LIGAND-AXIS ROBUSTNESS  src={out['ckpt']}  n={out['n']}  DB={len(cids)}")
    print(f"{'cell':22s} {'top1':>6} {'top5':>6} {'MRR':>6} {'medR':>7}")
    for k in list(cells) + ["Pholo_to_lig_shuffled"]:
        e = out["results"][k]
        print(f"{k:22s} {e.get('top1',0):6.3f} {e.get('top5',0):6.3f} {e.get('mrr',0):6.3f} "
              f"{e.get('median_rank',0):7.0f}")
    print(f"-- holo->AF3 drop: protein-query top5 {out['robustness']['protein_query_top5_drop']:+.3f}"
          f"  MRR {out['robustness']['protein_query_mrr_drop']:+.3f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--center", action="store_true")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--max-patch", type=int, default=128)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
