"""Phase-8 A4 — add Stage-1 embedding features to mined interfaces, for the probe's third arm.

Per interface, score every contacting surface-atom pair with the learned bilinear form
`s = z_i^T T z_j` (the same quantity chain retrieval aggregates) and summarise the distribution.
The probe then asks whether those summaries add anything over BSA and simple geometry.

Centering is over the WHOLE pool of chains scored here, matching how `p5.retrieval_bench` does it
(the DC-offset de-collapse, docs/10 §21/§24) — a per-interface mean would make scores incomparable
across interfaces, which is exactly the comparison the probe needs.

Usage:
  python -m masif_graph.p8.a4_embed --interfaces logs/phase8A/a4/interfaces.json \
      --npz <dir> --ckpt <ckpt> --out logs/phase8A/a4/interfaces_embed.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from masif_graph.p4.dataset import load_chain_graph
from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, normalize
from masif_graph.p4.dataset import D_AA, D_VA, D_VV

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def _load(npz_dir, cid, pid, device):
    p = os.path.join(npz_dir, f"{cid}__holo__{pid}.npz")
    return load_chain_graph(p, device) if os.path.exists(p) else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfaces", default="logs/phase8A/a4/interfaces.json")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="logs/phase8A/a4/interfaces_embed.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = json.load(open(args.interfaces))
    iface = d["interfaces"]

    # which interfaces have graphs built?
    usable = []
    for x in iface:
        cid = f"{x['pdb']}_{x['c1']}_{x['c2']}"
        if (os.path.exists(os.path.join(args.npz, f"{cid}__holo__p1.npz"))
                and os.path.exists(os.path.join(args.npz, f"{cid}__holo__p2.npz"))
                and os.path.exists(os.path.join(args.npz, f"{cid}__contacts.npz"))):
            usable.append((cid, x))
    print(f"{len(usable)}/{len(iface)} mined interfaces have graphs + contacts built", flush=True)
    if not usable:
        raise SystemExit("no graphs built yet")

    g0 = load_chain_graph(os.path.join(args.npz, f"{usable[0][0]}__holo__p1.npz"), args.device)
    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = ck.get("cfg", {})
    enc = HeteroEncoder(g0["atom_feat"].shape[1], g0["vert_feat"].shape[1], D_AA, D_VV, D_VA,
                        d=cfg.get("d", 64), d_out=cfg.get("d_out", 32),
                        n_layers=cfg.get("layers", 4)).to(args.device)
    comp = Complementarity(cfg.get("d_out", 32), tau_init=cfg.get("tau", 0.1)).to(args.device)
    enc.load_state_dict(ck["enc"]); comp.load_state_dict(ck["comp"])
    enc.eval()

    raw, graphs = {}, {}
    for cid, _ in usable:
        for pid in ("p1", "p2"):
            g = _load(args.npz, cid, pid, args.device)
            graphs[(cid, pid)] = g
            raw[(cid, pid)] = enc(g)
    mu = torch.cat(list(raw.values()), 0).mean(0, keepdim=True)
    z = {k: normalize(v - mu) for k, v in raw.items()}
    print(f"encoded {len(raw)} chains; z_std {float(torch.cat(list(z.values()),0).std(0).mean()):.4f}",
          flush=True)

    out_iface, n_scored = [], 0
    for cid, x in usable:
        pos = np.load(os.path.join(args.npz, f"{cid}__contacts.npz"))["pos"].reshape(-1, 2)
        if len(pos) < 10:
            out_iface.append(x)
            continue
        z1, z2 = z[(cid, "p1")], z[(cid, "p2")]
        i = torch.tensor(pos[:, 0], dtype=torch.long)
        j = torch.tensor(pos[:, 1], dtype=torch.long)
        if int(i.max()) >= z1.shape[0] or int(j.max()) >= z2.shape[0]:
            out_iface.append(x)
            continue
        s = ((z1[i] @ comp.T) * z2[j]).sum(1).cpu().numpy()
        y = dict(x)
        y.update({"emb_score_mean": float(s.mean()), "emb_score_max": float(s.max()),
                  "emb_score_median": float(np.median(s)),
                  "emb_score_p90": float(np.percentile(s, 90)),
                  "emb_n_contact_pairs": int(len(s))})
        out_iface.append(y)
        n_scored += 1

    d["interfaces"] = out_iface
    d["embedding"] = {"ckpt": os.path.basename(args.ckpt), "n_scored": n_scored,
                      "n_total": len(iface)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(d, open(args.out, "w"), indent=2)
    print(f"scored {n_scored}/{len(iface)} interfaces; wrote {args.out}")


if __name__ == "__main__":
    main()
