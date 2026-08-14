"""Stage R trainer — redesigned Stage 1 + the distogram that replaces Stage 2 (docs/26).

Four losses, and each one is there because Stage A measured a specific failure:

  R2 soft-target InfoNCE   the old hard single-label CE demanded an identity match that is not
                           identifiable from local chemistry (true partner at median rank 109/854).
                           Distance-decayed targets ask for a REGION instead.
  R2 dustbin               the old anchor was always a true contacting atom, so "is this atom at an
                           interface" was never supervised — measured as worth 7-24x precision.
  R3 distogram             replaces real-space pose: no alignment (so no alignment bias), ~10^5-10^6
                           labels per complex instead of one pose, and it back-propagates a
                           SPATIALLY MEANINGFUL gradient into the encoder, which R2 alone cannot.
  chain InfoNCE + VICReg   kept from Phase 4. Retrieval (0.644) is the do-no-harm side of the R4
                           gate, and VICReg is what stopped the Phase-4 collapse.

Conformer augmentation (D8-14): with `--apo-prob p` each chain is independently drawn from its
FASPR-repacked graph instead of holo. A repack reproduces 91% of the AF3 perturbation the encoder
feels (docs/25 §6) at zero GPU cost, so this is the cheap apo arm of the D8-12 hybrid.

The gate metric (top-1 spatial error, target <5 A from 19.4 A) is evaluated every `--eval-every`
epochs on held-out complexes and is what `--select-by` uses to checkpoint — deliberately NOT the
training loss, because Stage A showed chain retrieval can look excellent while this is catastrophic.

Usage:
  MASIF_GLOBAL_CTX=1 python -m masif_graph.p8.train_r --data <npz> --train-ids <f> --val-ids <f> \
      --out <ckpt.pt> --epochs 20 --seed 0 --apo-prob 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from masif_graph.p4.dataset import D_AA, D_VA, D_VV, load_chain_graph
from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import (Complementarity, chain_retrieval_loss, normalize,
                                      vicreg_terms)
from masif_graph.p8.distogram import DistogramHead, distogram_loss
from masif_graph.p8.objective_r import DustbinScore, soft_target_infonce, top1_spatial_error

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def _ids(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]


class Corpus:
    """Lazily loads (holo, rp) graphs + contacts per complex, caching what it has touched."""

    def __init__(self, data, ids, device="cpu", max_cache=400):
        self.data, self.device, self.max_cache = data, device, max_cache
        self.cache = {}
        self.ids = [c for c in ids if os.path.exists(os.path.join(data, f"{c}__contacts.npz"))
                    and os.path.exists(os.path.join(data, f"{c}__holo__p1.npz"))
                    and os.path.exists(os.path.join(data, f"{c}__holo__p2.npz"))]
        self.has_rp = {c for c in self.ids
                       if os.path.exists(os.path.join(data, f"{c}__rp__p1.npz"))
                       and os.path.exists(os.path.join(data, f"{c}__rp__p2.npz"))}

    def get(self, cid, states):
        key = (cid, states)
        if key in self.cache:
            return self.cache[key]
        g = []
        for pid, st in zip(("p1", "p2"), states):
            p = os.path.join(self.data, f"{cid}__{st}__{pid}.npz")
            g.append(load_chain_graph(p, self.device))
        pos = np.load(os.path.join(self.data, f"{cid}__contacts.npz"))["pos"].reshape(-1, 2)
        # An rp graph has its own surface-atom rows; the holo contact indices do NOT transfer, so a
        # substituted chain must have its contacts remapped by identity key. Handled by the caller
        # via `remap_pos`; here we only carry the holo indices.
        val = (g[0], g[1], pos)
        if len(self.cache) < self.max_cache:
            self.cache[key] = val
        return val


def _keys(path):
    z = np.load(path)
    return [k.decode() if isinstance(k, bytes) else str(k) for k in z["keys"].tolist()]


def remap_pos(data, cid, pos, states):
    """Remap holo contact rows onto whichever state each chain is in, by (chain,resseq,name) key.

    Rows whose atom is absent in the substituted state are DROPPED — the same identity-join the
    Phase-3/5 evaluation uses, so a repacked chain never contributes an invented positive.
    """
    if states == ("holo", "holo"):
        return pos
    cols = []
    for pid, st in zip(("p1", "p2"), states):
        hk = _keys(os.path.join(data, f"{cid}__holo__{pid}.npz"))
        if st == "holo":
            cols.append(None)
            continue
        sk = {k: i for i, k in enumerate(_keys(os.path.join(data, f"{cid}__{st}__{pid}.npz")))}
        cols.append((hk, sk))
    out = []
    for i, j in pos:
        a, b = i, j
        if cols[0] is not None:
            hk, sk = cols[0]
            if hk[i] not in sk:
                continue
            a = sk[hk[i]]
        if cols[1] is not None:
            hk, sk = cols[1]
            if hk[j] not in sk:
                continue
            b = sk[hk[j]]
        out.append((a, b))
    return np.asarray(out, np.int64).reshape(-1, 2)


@torch.no_grad()
def gate_metrics(enc, comp, scorer, corpus, ids, device, max_n=80):
    """R4 primary metric on held-out complexes: median top-1 spatial error (A)."""
    enc.eval()
    errs, n = [], 0
    for cid in ids:
        if n >= max_n:
            break
        try:
            g1, g2, pos = corpus.get(cid, ("holo", "holo"))
        except Exception:                                               # noqa: BLE001
            continue
        if len(pos) < 8:
            continue
        z1, z2 = normalize(enc(g1)), normalize(enc(g2))
        e = top1_spatial_error(z1, z2, torch.as_tensor(pos, dtype=torch.long, device=device),
                               g2["coord"], scorer)
        if e.numel():
            errs.append(e.cpu().numpy())
            n += 1
    enc.train()
    if not errs:
        return {"n": 0, "top1_spatial_median": float("nan")}
    a = np.concatenate(errs)
    return {"n": n, "n_queries": int(a.size),
            "top1_spatial_median": float(np.median(a)),
            "top1_spatial_mean": float(a.mean()),
            "frac_within_5A": float((a < 5).mean()),
            "frac_within_10A": float((a < 10).mean())}


def run(args):
    dev = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr = Corpus(args.data, _ids(args.train_ids), dev)
    va = Corpus(args.data, _ids(args.val_ids), dev)
    print(f"train {len(tr.ids)} complexes ({len(tr.has_rp)} with a repack state) | "
          f"val {len(va.ids)}", flush=True)
    if not tr.ids:
        raise SystemExit("no training complexes")

    g0, _, _ = tr.get(tr.ids[0], ("holo", "holo"))
    f_atom, f_vert = g0["atom_feat"].shape[1], g0["vert_feat"].shape[1]
    print(f"atom_feat dim = {f_atom} (global context {'ON' if f_atom > 26 else 'OFF'})", flush=True)
    enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA, d=args.d, d_out=args.d_out,
                        n_layers=args.layers).to(dev)
    comp = Complementarity(args.d_out, tau_init=args.tau).to(dev)
    scorer = DustbinScore(comp).to(dev)
    head = DistogramHead(args.d_out).to(dev)
    params = list(enc.parameters()) + list(comp.parameters()) + list(head.parameters()) + [scorer.bin_score]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rng = np.random.default_rng(args.seed)
    best, hist = float("inf"), []
    for ep in range(args.epochs):
        t0 = time.time()
        order = rng.permutation(len(tr.ids))
        agg = {}
        nb = 0
        for s in range(0, len(order) - args.batch + 1, args.batch):
            sel = [tr.ids[i] for i in order[s:s + args.batch]]
            raws, patches, partner, per = [], [], [], []
            for cid in sel:
                states = tuple("rp" if (cid in tr.has_rp and rng.random() < args.apo_prob)
                               else "holo" for _ in range(2))
                try:
                    g1, g2, pos = tr.get(cid, states)
                    pos = remap_pos(args.data, cid, pos, states)
                except Exception:                                       # noqa: BLE001
                    continue
                if len(pos) < 8:
                    continue
                r1, r2 = enc(g1), enc(g2)
                per.append((r1, r2, g1["coord"], g2["coord"], pos))
                raws += [r1, r2]
            if len(per) < 2:
                continue
            mu = torch.cat(raws, 0).mean(0, keepdim=True)               # DC-offset centering
            loss = 0.0
            st_acc = {}
            for (r1, r2, c1, c2, pos) in per:
                z1, z2 = normalize(r1 - mu), normalize(r2 - mu)
                P = torch.as_tensor(pos, dtype=torch.long, device=dev)
                la, _ = soft_target_infonce(z1, z2, P, c1, c2, scorer, sigma=args.sigma,
                                            n_neg_query=args.n_neg_query, seed=args.seed + ep)
                ld, sd = distogram_loss(head, comp, z1, z2, c1, c2, pos, n_neg=None,
                                        neg_per_pos=args.neg_per_pos, seed=args.seed + ep)
                loss = loss + args.w_atom * la + args.w_disto * ld
                for k, v in sd.items():
                    st_acc.setdefault(k, []).append(v)
                # interface patches for the chain-level term
                ii = np.unique(pos[:, 0]); jj = np.unique(pos[:, 1])
                patches += [z1[torch.as_tensor(ii, dtype=torch.long, device=dev)],
                            z2[torch.as_tensor(jj, dtype=torch.long, device=dev)]]
            loss = loss / len(per)
            n_ch = len(patches)
            partner = torch.tensor([k + 1 if k % 2 == 0 else k - 1 for k in range(n_ch)], device=dev)
            if args.w_chain > 0 and n_ch >= 4:
                loss = loss + args.w_chain * chain_retrieval_loss(patches, partner, comp,
                                                                 tau_c=args.tau_c)
            zc = torch.cat(raws, 0)
            v, c = vicreg_terms(zc)
            loss = loss + args.vicreg_var * v + args.vicreg_cov * c
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            nb += 1
            agg.setdefault("loss", []).append(float(loss))
            agg.setdefault("z_std", []).append(float(zc.std(0).mean()))
            for k, vv in st_acc.items():
                agg.setdefault(k, []).append(float(np.mean(vv)))
            if args.max_batches and nb >= args.max_batches:
                break
        sched.step()
        row = {"epoch": ep, "batches": nb, "secs": round(time.time() - t0, 1),
               **{k: round(float(np.mean(v)), 4) for k, v in agg.items()}}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            row.update(gate_metrics(enc, comp, scorer, va, va.ids, dev, max_n=args.val_n))
            m = row.get("top1_spatial_median", float("inf"))
            if np.isfinite(m) and m < best:
                best = m
                torch.save({"enc": enc.state_dict(), "comp": comp.state_dict(),
                            "head": head.state_dict(), "bin_score": scorer.bin_score.detach(),
                            "cfg": {"d": args.d, "d_out": args.d_out, "layers": args.layers,
                                    "tau": args.tau, "f_atom": f_atom, "global_ctx": f_atom > 26,
                                    "sigma": args.sigma, "apo_prob": args.apo_prob}},
                           args.out)
                row["saved"] = True
        hist.append(row)
        print(json.dumps(row), flush=True)
        if args.log:
            os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
            json.dump(hist, open(args.log, "w"), indent=2)
    print(f"BEST top1_spatial_median = {best:.2f} A  -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d-out", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--tau-c", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=4.0, help="soft-target width, angstroms")
    ap.add_argument("--w-atom", type=float, default=1.0)
    ap.add_argument("--w-disto", type=float, default=1.0)
    ap.add_argument("--w-chain", type=float, default=0.5)
    ap.add_argument("--n-neg-query", type=int, default=256, help="dustbin queries per chain")
    ap.add_argument("--neg-per-pos", type=int, default=4,
                    help="random non-contact pairs per true contact (class balance)")
    ap.add_argument("--vicreg-var", type=float, default=2.0)
    ap.add_argument("--vicreg-cov", type=float, default=0.04)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--apo-prob", type=float, default=0.0, help="P(draw a chain from its repack)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
