"""Phase-6 C(c).3 axis 3 — does the encoder recover a drug-induced binding partner?

For each of the 14 ligand-induced ternary systems, each subunit is one query (28 cases):

  query patch  = that subunit's surface atoms within `--lig-radius` of the drug
                 **+ the drug's own atom embeddings** (Path B: the drug is a graph of atoms)
  DB entry     = the FULL surface-atom embedding set of a chain
  score        = median_i max_j (z_qi)^T T (z_dj)          (the Phase-4/5 form)
  target       = the other subunit of the same system

Two deliberate choices keep the number honest:
* the query uses **no knowledge of the partner** — the patch is defined by the drug alone, which is
  exactly what "neosurface" means and what deployment would have. (The Phase-5 gate, by contrast,
  uses interface patches on both sides.)
* DB entries are **whole surfaces**, not interface patches, for the same reason — at deployment
  nobody tells you which part of a database protein is the interface.

The control that decides whether the signal is a *neosurface* signal at all is `--no-ligand`: the
identical run with the drug's atom embeddings dropped from the query. If ranking does not improve
when the drug is present, the model is just doing protein-surface matching and the ligand adds
nothing — a negative result worth reporting plainly.

n = 28. That is small, so per-system ranks are always emitted alongside the pooled numbers.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from masif_graph.p4.dataset import D_AA, D_VV, D_VA, load_chain_graph
from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, normalize


def systems(npz_dir):
    return sorted({f[:-len("__lig.npz")] for f in os.listdir(npz_dir) if f.endswith("__lig.npz")})


def decoy_chains(npz_dir, ids_file, limit=None):
    out = []
    ids = [l.strip() for l in open(ids_file) if l.strip() and not l.startswith("#")]
    for cid in ids:
        for pid in ("p1", "p2"):
            p = os.path.join(npz_dir, f"{cid}__holo__{pid}.npz")
            if os.path.exists(p):
                out.append((f"{cid}:{pid}", p))
    return out[:limit] if limit else out


@torch.no_grad()
def run(args):
    dev = args.device
    ck = torch.load(args.ckpt, map_location=dev)
    cfg = ck.get("cfg", {})

    sys_ids = systems(args.neosurf_data)
    probe = load_chain_graph(os.path.join(args.neosurf_data, f"{sys_ids[0]}__holo__p1.npz"), dev)
    enc = HeteroEncoder(probe["atom_feat"].shape[1], 4, D_AA, D_VV, D_VA, d=cfg.get("d", 64),
                        d_out=cfg.get("d_out", 32), n_layers=cfg.get("layers", 4)).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()
    comp = Complementarity(cfg.get("d_out", 32)).to(dev)
    comp.load_state_dict(ck["comp"])

    rng = np.random.default_rng(0)
    raw, meta, dsc = {}, {}, {}
    for s in sys_ids:
        g1 = load_chain_graph(os.path.join(args.neosurf_data, f"{s}__holo__p1.npz"), dev)
        g2 = load_chain_graph(os.path.join(args.neosurf_data, f"{s}__holo__p2.npz"), dev)
        gl = load_chain_graph(os.path.join(args.neosurf_data, f"{s}__lig.npz"), dev)
        raw[(s, "p1")], raw[(s, "p2")], raw[(s, "lig")] = enc(g1), enc(g2), enc(gl)
        dsc[(s, "p1")], dsc[(s, "p2")] = g1, g2
        lig_xyz = gl["coord"].cpu().numpy()
        tree = cKDTree(lig_xyz)
        meta[s] = {}
        for pid, g in (("p1", g1), ("p2", g2)):
            c = g["coord"].cpu().numpy()
            near = np.flatnonzero(tree.query_ball_point(c, args.lig_radius, return_length=True) > 0)
            meta[s][pid] = {"near": near, "n_surf": c.shape[0]}
    # Phase-7: the COMPOSITE protein+drug surface as an alternative query representation. Loaded
    # into the same centering pool so it sits on one scale with everything else, but deliberately
    # NOT added to the DB -- a composite entry would trivially match itself.
    cmp_meta = {}
    if getattr(args, "composite_data", None):
        from masif_graph.p6.neosurf import parse_benchmark
        chain_of = {}
        for e in parse_benchmark(args.bench):
            chain_of[(e["sys"], "p1")] = e["c1"]; chain_of[(e["sys"], "p2")] = e["c2"]
        for s_ in sys_ids:
            lig_xyz = load_chain_graph(os.path.join(args.neosurf_data, f"{s_}__lig.npz"),
                                       dev)["coord"].cpu().numpy()
            tree = cKDTree(lig_xyz)
            for qr in ("p1", "p2"):
                ch = chain_of.get((s_, qr))
                p = os.path.join(args.composite_data, f"{s_}_{ch}__cmp.npz")
                if ch is None or not os.path.exists(p):
                    continue
                g = load_chain_graph(p, dev)
                raw[("cmp", s_, qr)] = enc(g)
                zc = np.load(p)
                is_lig = zc["readout_is_lig"].astype(bool)
                co = g["coord"].cpu().numpy()
                near = np.flatnonzero((tree.query_ball_point(co, args.lig_radius,
                                                             return_length=True) > 0) & ~is_lig)
                cmp_meta[(s_, qr)] = {"near": near, "is_lig": np.flatnonzero(is_lig),
                                      "n_readout": len(co)}
    decoys = decoy_chains(args.decoy_data, args.decoy_ids, args.max_decoys)
    for name, path in decoys:
        g = load_chain_graph(path, dev)
        raw[("decoy", name)] = enc(g)
        dsc[("decoy", name)] = g

    mu = torch.cat([v for v in raw.values()], 0).mean(0, keepdim=True) if args.center else 0.0
    emb = {k: normalize(v - mu) for k, v in raw.items()}
    z_std = float(torch.cat(list(emb.values()), 0).std(0).mean())
    # frozen MaSIF is only scoreable where the descriptor net actually ran; the neosurface build
    # skips it by default, so report availability rather than silently scoring rows of zeros.
    have_desc = {k: bool(v["desc_straight"].abs().sum() > 0) for k, v in dsc.items()}
    frozen_ok = all(have_desc.get((s, p), False) for s in sys_ids for p in ("p1", "p2")) and \
        all(have_desc.get(("decoy", n), False) for n, _ in decoys)

    # ---- DB: every subunit + every decoy, as a whole surface (subsampled for memory) ----
    def cap(z):
        if args.max_db_atoms and z.shape[0] > args.max_db_atoms:
            k = np.sort(rng.choice(z.shape[0], args.max_db_atoms, False))
            return z[torch.as_tensor(k, dtype=torch.long, device=z.device)]
        return z

    dbk, mats, seg = [], [], []
    for key in list(raw):
        if key[0] == "cmp":
            continue
        if key[0] == "decoy" or key[1] in ("p1", "p2"):
            m = cap(emb[key])
            if m.shape[0] == 0:
                continue
            seg.append(torch.full((m.shape[0],), len(dbk), dtype=torch.long, device=m.device))
            mats.append(m); dbk.append(key)
    Mdb, seg = torch.cat(mats, 0), torch.cat(seg, 0)
    idx_of = {k: i for i, k in enumerate(dbk)}
    TZ = comp.T @ Mdb.t()
    n_db = len(dbk)

    def rank_case(s, qr, with_ligand):
        near = meta[s][qr]["near"]
        if len(near) == 0:
            return None
        zq = emb[(s, qr)]
        parts = [zq[torch.as_tensor(near, dtype=torch.long, device=zq.device)]]
        if with_ligand:
            parts.append(emb[(s, "lig")])
        q = torch.cat(parts, 0)
        S = torch.full((q.shape[0], n_db), float("-inf"), device=q.device)
        S.scatter_reduce_(1, seg.expand(q.shape[0], -1), q @ TZ, reduce="amax", include_self=True)
        score = S.median(0).values
        score[idx_of[(s, qr)]] = float("-inf")            # never retrieve the query itself
        order = torch.argsort(score, descending=True).tolist()
        target = idx_of[(s, "p2" if qr == "p1" else "p1")]
        return order.index(target) + 1

    def rank_case_cmp(s_, qr, with_ligand):
        m = cmp_meta.get((s_, qr))
        if m is None or len(m["near"]) == 0:
            return None
        z = emb[("cmp", s_, qr)]
        idx = m["near"] if not with_ligand else np.concatenate([m["near"], m["is_lig"]])
        q = z[torch.as_tensor(idx, dtype=torch.long, device=z.device)]
        S = torch.full((q.shape[0], n_db), float("-inf"), device=q.device)
        S.scatter_reduce_(1, seg.expand(q.shape[0], -1), q @ TZ, reduce="amax", include_self=True)
        score = S.median(0).values
        score[idx_of[(s_, qr)]] = float("-inf")
        order = torch.argsort(score, descending=True).tolist()
        return order.index(idx_of[(s_, "p2" if qr == "p1" else "p1")]) + 1

    out = {"ckpt": os.path.basename(args.ckpt), "n_systems": len(sys_ids), "db_chains": n_db,
           "n_decoys": len(decoys), "lig_radius": args.lig_radius, "center": args.center,
           "z_std": z_std, "cases": []}
    for s in sys_ids:
        for qr in ("p1", "p2"):
            r_with = rank_case(s, qr, True)
            r_without = rank_case(s, qr, False)
            if r_with is None:
                continue
            case = {"system": s, "query": qr, "n_near": int(len(meta[s][qr]["near"])),
                    "n_lig": int(emb[(s, "lig")].shape[0]),
                    "rank_with_ligand": r_with, "rank_no_ligand": r_without}
            if (s, qr) in cmp_meta:
                case["rank_composite"] = rank_case_cmp(s, qr, True)
                case["rank_composite_noligand"] = rank_case_cmp(s, qr, False)
                case["n_near_composite"] = int(len(cmp_meta[(s, qr)]["near"]))
            out["cases"].append(case)

    def summ(key):
        r = np.array([c[key] for c in out["cases"] if c.get(key) is not None], float)
        if len(r) == 0:
            return {"n": 0}
        return {"n": len(r), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
                "top10": float((r <= 10).mean()), "top20": float((r <= 20).mean()),
                "mrr": float((1 / r).mean()), "median_rank": float(np.median(r))}

    # ---- frozen MaSIF on the IDENTICAL patches (the published representation as a baseline) ----
    # It scores by nearest descriptor: S(q,d) = median_i min_j ||ds_qi - df_dj||, lower = better.
    # It has no ligand arm by construction — Path B's protein surface is built WITHOUT the drug, so
    # the frozen descriptor is ligand-blind. That makes it comparable to the `no_ligand` arm only.
    out["frozen_available"] = bool(frozen_ok)
    if frozen_ok:
        fmats, fseg = [], []
        for i, key in enumerate(dbk):
            m = cap(dsc[key]["desc_flipped"])
            fseg.append(torch.full((m.shape[0],), i, dtype=torch.long, device=m.device))
            fmats.append(m)
        Fdb, fseg = torch.cat(fmats, 0), torch.cat(fseg, 0)
        for c in out["cases"]:
            s, qr = c["system"], c["query"]
            near = meta[s][qr]["near"]
            q = dsc[(s, qr)]["desc_straight"][torch.as_tensor(near, dtype=torch.long)]
            S = torch.full((q.shape[0], n_db), float("inf"), device=q.device)
            S.scatter_reduce_(1, fseg.expand(q.shape[0], -1), torch.cdist(q, Fdb),
                              reduce="amin", include_self=True)
            score = S.median(0).values
            score[idx_of[(s, qr)]] = float("inf")
            order = torch.argsort(score).tolist()
            c["rank_frozen"] = order.index(idx_of[(s, "p2" if qr == "p1" else "p1")]) + 1

    out["with_ligand"] = summ("rank_with_ligand")
    out["no_ligand"] = summ("rank_no_ligand")
    if cmp_meta:
        # composite = the drug and the protein share ONE surface (a real neosurface);
        # composite_noligand isolates "the drug reshaped the protein surface" from
        # "the drug contributed its own embeddings".
        out["composite"] = summ("rank_composite")
        out["composite_noligand"] = summ("rank_composite_noligand")
    if frozen_ok:
        out["frozen"] = summ("rank_frozen")
    out["chance_top5"] = round(5.0 / max(n_db - 1, 1), 5)
    a = np.array([c["rank_with_ligand"] for c in out["cases"]], float)
    b = np.array([c["rank_no_ligand"] for c in out["cases"]], float)
    out["ligand_effect"] = {"n_better_with_ligand": int((a < b).sum()),
                            "n_worse": int((a > b).sum()), "n_tied": int((a == b).sum()),
                            "median_rank_delta": float(np.median(a - b))}

    print(f"DB={n_db} chains ({len(decoys)} decoy) | z_std={z_std:.4f} | chance top5={out['chance_top5']:.4f}")
    for k in ("with_ligand", "no_ligand") + (("frozen",) if frozen_ok else ()):
        m = out[k]
        print(f"  {k:12s} n={m['n']:3d} top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"top10={m['top10']:.3f} MRR={m['mrr']:.3f} medRank={m['median_rank']:.0f}")
    print(f"  ligand effect: better {out['ligand_effect']['n_better_with_ligand']} / "
          f"worse {out['ligand_effect']['n_worse']} / tied {out['ligand_effect']['n_tied']} "
          f"(median rank delta {out['ligand_effect']['median_rank_delta']:+.0f})")
    print("  per-case ranks (with/without ligand):")
    for c in out["cases"]:
        fr = f" / frozen {c['rank_frozen']:5d}" if "rank_frozen" in c else ""
        print(f"    {c['system']:10s} {c['query']} near={c['n_near']:4d} lig={c['n_lig']:3d} "
              f"{c['rank_with_ligand']:5d} / {c['rank_no_ligand']:5d}{fr}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neosurf-data", required=True)
    ap.add_argument("--decoy-data", required=True)
    ap.add_argument("--decoy-ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--lig-radius", type=float, default=6.0)
    ap.add_argument("--composite-data", default=None,
                    help="Phase-7 composite protein+drug graphs (query side only)")
    ap.add_argument("--bench", default="/scratch/ymeng/masif-graph/masif-neosurf-af2/"
                                       "computational_benchmark/benchmark_pdbs.txt")
    ap.add_argument("--max-decoys", type=int, default=0)
    ap.add_argument("--max-db-atoms", type=int, default=1500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
