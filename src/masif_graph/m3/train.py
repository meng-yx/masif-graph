"""M3 contrastive training + eval (atomsurf_h100 env).

Trains the M3 fused encoder (learnable DiffusionNet surface ⊕ invariant chem graph) so that:
  - COMPLEMENTARITY (preserve the useful signal): true holo contacts have close p1-straight/p2-flipped
    embeddings; random pairs are pushed apart by a margin.
  - INVARIANCE (the new objective): an atom's AF3 embedding is pulled to its holo embedding
    (identity-matched), so an AF3-query descriptor matches the holo database like the holo one would.

Eval = the M1 metric (AF3-query straight vs holo-db flipped, both directions, randneg) on HELD-OUT
complexes, vs the frozen mean-pooled baseline. Absolute AF3->holo AUC is the headline (Phase-2 lesson).
Complex-level split; structural-mismatch complexes excluded from training positives (user guidance).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from masif_graph.m3.encoder import M3Encoder
from masif_graph.m3.dataset import ComplexData, usable_complexes


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def encode_chain(enc, chain, which, return_reg=False):
    """per-atom embedding for a chain in one 'which' in {'straight','flipped'}."""
    desc = chain.desc_straight if which == "straight" else chain.desc_flipped
    return enc(chain.surf, desc, chain.vertex_surf_idx, chain.n_surf, chain.gt, return_reg=return_reg)


def _pair_d(a, b, idx):
    d = a[idx[:, 0]] - b[idx[:, 1]]
    return torch.sqrt((d * d).sum(1) + 1e-9)


def complementarity_loss(e1s, e2f, contacts, margin, rng):
    """holo contacts: p1-straight vs p2-flipped close; random negatives apart by margin."""
    if len(contacts) == 0:
        return torch.zeros((), device=e1s.device)
    pos = _pair_d(e1s, e2f, contacts)
    n2 = e2f.shape[0]
    rj = torch.randint(n2, (len(contacts),), device=e1s.device)
    neg = torch.sqrt(((e1s[contacts[:, 0]] - e2f[rj]) ** 2).sum(1) + 1e-9)
    return (pos ** 2).mean() + (torch.clamp(margin - neg, min=0) ** 2).mean()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    train_ids = usable_complexes(args.data, [l.strip() for l in open(args.train_ids) if l.strip()])
    eval_ids = usable_complexes(args.data, [l.strip() for l in open(args.eval_ids) if l.strip()])
    mism = set()
    if args.mismatch_json and os.path.exists(args.mismatch_json):
        mism = set(json.load(open(args.mismatch_json)).get("mismatch_ids", []))
    # complex-level train/val split (val = model selection; eval set stays untouched -> no peeking)
    rng = np.random.default_rng(args.seed)
    perm = list(train_ids); rng.shuffle(perm)
    n_val = max(6, int(round(args.val_frac * len(perm))))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    log(f"train={len(tr_ids)} val={len(val_ids)} eval={len(eval_ids)} "
        f"(mismatch excluded from train pos: {len(mism & set(tr_ids))})")

    log("loading complexes into memory...")
    tr_c = {c: ComplexData(args.data, c, device) for c in tr_ids}
    val_c = {c: ComplexData(args.data, c, device) for c in val_ids}
    eval_c = {c: ComplexData(args.data, c, device) for c in eval_ids}

    enc = M3Encoder(desc_dim=80, out_dim=args.out_dim, use_graph=not args.no_graph).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"encoder params: {sum(p.numel() for p in enc.parameters())}")

    # step-0 baseline = frozen-normalized (residual=0 at init): the exact apples-to-apples reference.
    b0v, b0e = evaluate(enc, val_c, device), evaluate(enc, eval_c, device)
    log(f"step 0 (FROZEN-normalized baseline): val af3={b0v['af3']:.3f} hh={b0v['hh']:.3f} | "
        f"EVAL af3={b0e['af3']:.3f} hh={b0e['hh']:.3f}  (M1 raw-frozen af3=0.821)")

    best = {"val_af3": b0v["af3"], "step": 0, "eval_af3": b0e["af3"], "eval_hh": b0e["hh"]}
    torch.save(enc.state_dict(), os.path.join(args.out, "m3_encoder_best.pt"))
    order = list(tr_c)
    for step in range(1, args.steps + 1):
        enc.train()
        cd = tr_c[order[rng.integers(len(order))]]
        cid = cd.cid
        h1s, r1 = encode_chain(enc, cd.holo["p1"], "straight", return_reg=True)
        h2f, r2 = encode_chain(enc, cd.holo["p2"], "flipped", return_reg=True)
        loss = complementarity_loss(h1s, h2f, cd.contacts, args.margin, rng)
        reg = r1 + r2
        inv = torch.zeros((), device=device)
        if cd.af3 is not None and cid not in mism:
            for p, which, hemb in (("p1", "straight", h1s), ("p2", "flipped", h2f)):
                hr, ar = cd.match[p]
                if len(hr) >= 3:
                    aemb, ra = encode_chain(enc, cd.af3[p], which, return_reg=True)
                    inv = inv + ((aemb[ar] - hemb[hr].detach()) ** 2).sum(1).mean()
                    reg = reg + ra
        (loss + args.inv_weight * inv + args.reg_weight * reg).backward()
        opt.step(); opt.zero_grad()

        if step % args.eval_every == 0 or step == args.steps:
            au = evaluate(enc, val_c, device)
            marker = ""
            if au["af3"] > best["val_af3"]:
                ev = evaluate(enc, eval_c, device)
                best = {"val_af3": au["af3"], "step": step, "eval_af3": ev["af3"], "eval_hh": ev["hh"]}
                torch.save(enc.state_dict(), os.path.join(args.out, "m3_encoder_best.pt"))
                marker = " *best"
            log(f"step {step}: comp={loss.item():.3f} inv={float(inv):.3f} | val af3={au['af3']:.3f} "
                f"hh={au['hh']:.3f}{marker}")
    summary = {"frozen_norm_baseline": {"eval_af3": b0e["af3"], "eval_hh": b0e["hh"]},
               "m1_raw_frozen_eval_af3": 0.821, "best": best,
               "delta_eval_af3_vs_frozennorm": best["eval_af3"] - b0e["af3"]}
    json.dump(summary, open(os.path.join(args.out, "m3_train_summary.json"), "w"), indent=2)
    log(f"BEST (val-selected @step {best['step']}): EVAL af3={best['eval_af3']:.3f} hh={best['eval_hh']:.3f} "
        f"| frozen-norm eval af3={b0e['af3']:.3f} -> delta {best['eval_af3']-b0e['af3']:+.3f}")
    return summary


def _remap(match):
    """holo-row -> alt(af3)-row dict from a (holo_rows, af3_rows) match tensor pair."""
    h, a = match
    return {int(x): int(y) for x, y in zip(h.tolist(), a.tolist())}


@torch.no_grad()
def evaluate(enc, eval_c, device, seed=0):
    """AF3-query straight vs holo-db flipped (both directions), randneg — the M1 af3_holo metric,
    plus holo->holo. Encodes each complex ONCE; vectorized contact scoring. Pools over complexes."""
    enc.eval()
    rng = np.random.default_rng(seed)
    acc = {"af3": ([], []), "hh": ([], [])}   # regime -> (pos_list, neg_list)
    for cd in eval_c.values():
        if cd.af3 is None:
            continue
        E = {(st, p, w): encode_chain(enc, src[p], w)
             for st, src in (("holo", cd.holo), ("af3", cd.af3))
             for p in ("p1", "p2") for w in ("straight", "flipped")}
        h2a = {"p1": _remap(cd.match["p1"]), "p2": _remap(cd.match["p2"])}
        cont = cd.contacts.tolist()
        for qs in ("af3", "hh"):
            qstate = "af3" if qs == "af3" else "holo"
            pos, neg = acc[qs]
            # direction 1: query chain1 straight vs holo chain2 flipped; direction 2: swapped
            for (qp, dp, ci) in (("p1", "p2", 0), ("p2", "p1", 1)):
                dbf = E[("holo", dp, "flipped")]
                qmat = E[(qstate, qp, "straight")]
                for row in cont:
                    di = row[1 - ci]                 # db atom = the OTHER chain's holo row
                    qi = row[ci] if qstate == "holo" else h2a[qp].get(row[ci])
                    if qi is None:
                        continue
                    q = qmat[qi]
                    pos.append(float(torch.linalg.vector_norm(q - dbf[di])))
                    rj = int(rng.integers(dbf.shape[0]))
                    neg.append(float(torch.linalg.vector_norm(q - dbf[rj])))
    out = {}
    for qs, (pos, neg) in acc.items():
        if not pos or not neg:
            out[qs] = float("nan"); continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = 1.0 / (np.r_[pos, neg] + 1e-9)
        out[qs] = float(roc_auc_score(y, s))
    return {"af3": out["af3"], "hh": out["hh"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--eval-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mismatch-json", default="logs/phase3/m1_full/m1_mismatch.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--inv-weight", type=float, default=1.0)
    ap.add_argument("--out-dim", type=int, default=80)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--reg-weight", type=float, default=0.5,
                    help="penalty on refinement magnitude (anchor to frozen baseline; anti-overfit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graph", action="store_true", help="ablate the chem-graph branch (surface-only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
