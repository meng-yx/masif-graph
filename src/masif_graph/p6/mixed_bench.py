"""Phase-6 C — chain-level retrieval over a MIXED corpus (PPI + protein-ligand).

The Phase-5 gate (`p5/retrieval_bench.py`) needs holo+AF3 states and is PPI-only. The mixed
held-out axis needs something simpler and type-aware: one held-out DB containing both protein
chains and ligands, and for every entry the question "does its true partner rank at the top?".

Score is the Phase-4/5 form, unchanged:  S(q,d) = median_i max_j (z_qi)^T T (z_dj),
with **DC-offset centering mandatory** (`--center`) — Phase-4 §21 found the raw embeddings share a
mean ~32x larger than the per-chain variation, so plain L2-normalisation collapses every chain onto
one direction and retrieval dies.

Reported per type (`ppi` / `pl`) and, for protein-ligand, per query role — "given this pocket find
its ligand" (p1) and "given this ligand find its pocket" (p2) are different questions and the
pooled number can hide one of them. The shuffled-partner control gives the chance line; a decoy
pool mixing proteins and ligands is *easier* than a same-type pool, so `--same-type-db` restricts
each query to compete only against DB entries of its own role — the honest version of the number.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from masif_graph.p4.dataset import ComplexP4
from masif_graph.p4.objective import normalize


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
def build_patches(enc, comp, recs, device="cpu", center=True, pos_key="pos", max_patch=128, seed=0):
    """recs: list of (cid, kind, ComplexP4) -> (patches dict, z_std). Encodes every chain once."""
    enc.eval()
    raw, allz = {}, []
    for cid, _kind, c in recs:
        e = {"p1": enc(c.p1), "p2": enc(c.p2)}
        raw[cid] = e
        allz.append(torch.cat(list(e.values()), 0))
    mu = torch.cat(allz, 0).mean(0, keepdim=True) if center else 0.0
    emb = {cid: {k: normalize(v - mu) for k, v in e.items()} for cid, e in raw.items()}
    z_std = float(torch.cat([torch.cat(list(e.values()), 0) for e in emb.values()], 0).std(0).mean())

    rng = np.random.default_rng(seed)
    pat = {}
    for cid, kind, c in recs:
        pos = getattr(c, pos_key).detach().cpu().numpy().reshape(-1, 2)
        if max_patch and len(pos) > max_patch:
            pos = pos[rng.choice(len(pos), max_patch, replace=False)]
        i1, i2 = _u(pos[:, 0]), _u(pos[:, 1])
        gi = lambda a: torch.as_tensor(a, dtype=torch.long, device=emb[cid]["p1"].device)
        pat[cid] = {"kind": kind, "p1": emb[cid]["p1"][gi(i1)], "p2": emb[cid]["p2"][gi(i2)]}
    return pat, z_std


@torch.no_grad()
def retrieve(pat, comp, same_type_db=True, shuffle=False, seed=0):
    """Rank every chain's true partner in the pooled DB. Returns list of (cid, kind, role, rank)."""
    dbk = [(c, r) for c in pat for r in ("p1", "p2") if pat[c][r].shape[0] > 0]
    idx_of = {k: i for i, k in enumerate(dbk)}
    dev = pat[dbk[0][0]][dbk[0][1]].device
    mats, seg = [], []
    for i, (c, r) in enumerate(dbk):
        m = pat[c][r]
        mats.append(m)
        seg.append(torch.full((m.shape[0],), i, dtype=torch.long, device=dev))
    Mdb, seg = torch.cat(mats, 0), torch.cat(seg, 0)
    TZ = comp.T @ Mdb.t()                                    # (d, Ntot)
    n_db = len(dbk)
    kinds = np.array([pat[c]["kind"] for c, _ in dbk])
    roles = np.array([r for _, r in dbk])

    rng = np.random.default_rng(seed)
    out = []
    for (cid, qr) in dbk:
        pr = "p2" if qr == "p1" else "p1"
        if (cid, pr) not in idx_of:
            continue
        q = pat[cid][qr]
        S = torch.full((q.shape[0], n_db), float("-inf"), device=dev)
        S.scatter_reduce_(1, seg.expand(q.shape[0], -1), q @ TZ, reduce="amax", include_self=True)
        score = S.median(0).values
        score[idx_of[(cid, qr)]] = float("-inf")             # never retrieve yourself
        if same_type_db:
            # a protein query beating a *ligand* decoy is not evidence of binder discrimination;
            # restrict the pool to the DB entries that play the same structural role.
            mask = (kinds == pat[cid]["kind"]) & (roles == pr)
            score = score.masked_fill(~torch.as_tensor(mask, device=dev), float("-inf"))
        order = torch.argsort(score, descending=True).tolist()
        true = order[rng.integers(int((score > float("-inf")).sum()))] if shuffle else idx_of[(cid, pr)]
        out.append((cid, pat[cid]["kind"], qr, order.index(true) + 1,
                    int((score > float("-inf")).sum()) + 1))
    return out


def summarize(rows, tag=""):
    res = {}
    groups = {"all": lambda r: True,
              "ppi": lambda r: r[1] == "ppi",
              "pl": lambda r: r[1] == "pl",
              "pl_query_protein": lambda r: r[1] == "pl" and r[2] == "p1",
              "pl_query_ligand": lambda r: r[1] == "pl" and r[2] == "p2",
              "ppi_query_p1": lambda r: r[1] == "ppi" and r[2] == "p1"}
    for name, f in groups.items():
        sel = [r for r in rows if f(r)]
        if sel:
            res[tag + name] = metrics([r[3] for r in sel], int(np.median([r[4] for r in sel])))
    return res


def load_recs(data_dirs, id_files, kinds, device="cpu", min_pos=8):
    """[(dir, ids_file, kind)] -> list of (cid, kind, ComplexP4); skips incomplete/empty complexes."""
    recs, missing = [], 0
    for d, idf, kind in zip(data_dirs, id_files, kinds):
        for cid in [l.strip() for l in open(idf) if l.strip() and not l.startswith("#")]:
            try:
                c = ComplexP4(d, cid, device)
            except (FileNotFoundError, OSError):
                missing += 1
                continue
            if c.pos.shape[0] < min_pos:
                missing += 1
                continue
            recs.append((cid, kind, c))
    return recs, missing


def main():
    import argparse

    from masif_graph.p4.encoder import HeteroEncoder
    from masif_graph.p4.dataset import D_AA, D_VV, D_VA
    from masif_graph.p4.objective import Complementarity

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi-data"); ap.add_argument("--ppi-ids")
    ap.add_argument("--pl-data"); ap.add_argument("--pl-ids")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--max-patch", type=int, default=128)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--mixed-db", action="store_true", help="let proteins and ligands share one pool")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dirs, ids, kinds = [], [], []
    if a.ppi_ids:
        dirs.append(a.ppi_data); ids.append(a.ppi_ids); kinds.append("ppi")
    if a.pl_ids:
        dirs.append(a.pl_data); ids.append(a.pl_ids); kinds.append("pl")
    recs, missing = load_recs(dirs, ids, kinds, a.device, a.min_pos)
    print(f"loaded {len(recs)} complexes ({missing} skipped) -> DB {2*len(recs)} chains", flush=True)

    ck = torch.load(a.ckpt, map_location=a.device)
    f_atom = recs[0][2].p1["atom_feat"].shape[1]
    cfg = ck.get("cfg", {})
    enc = HeteroEncoder(f_atom, 4, D_AA, D_VV, D_VA, d=cfg.get("d", 64),
                        d_out=cfg.get("d_out", 32), n_layers=cfg.get("layers", 4)).to(a.device)
    enc.load_state_dict(ck["enc"])
    comp = Complementarity(cfg.get("d_out", 32)).to(a.device)
    comp.load_state_dict(ck["comp"])

    pat, z_std = build_patches(enc, comp, recs, a.device, a.center, a.pos_key, a.max_patch)
    print(f"z_std(post-center)={z_std:.4f}", flush=True)
    rows = retrieve(pat, comp, same_type_db=not a.mixed_db)
    out = {"ckpt": os.path.basename(a.ckpt), "n": len(recs), "z_std": z_std,
           "center": a.center, "pos_key": a.pos_key, "same_type_db": not a.mixed_db,
           "results": summarize(rows)}
    out["results"].update(summarize(retrieve(pat, comp, not a.mixed_db, shuffle=True), "shuffled_"))
    print(json.dumps(out["results"], indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
