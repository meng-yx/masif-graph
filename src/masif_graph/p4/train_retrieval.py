"""Phase-4 M2 fix: retrain with hard decoy-partner-CHAIN negatives to earn RETRIEVAL (docs/10 §20).

The retrieval failure was a training-objective gap: Stage-A InfoNCE ranked the true partner ATOM against
random/bank atoms, never the true partner CHAIN against decoy partner chains — so the encoder learned
"contact-like" not "complements THIS partner." This fine-tunes the (invariant) Stage-A encoder with a
chain-level contrastive loss over interface patches (`objective.chain_retrieval_loss`): each chain must
rank its true partner above every other chain in the batch (the in-batch chains ARE the hard decoy-partner
negatives). A small atom-level InfoNCE + VICReg are kept so local correspondence and anti-collapse survive.

Eval = the deployment retrieval metric itself (learned holo & AF3 top-k vs the frozen ceiling on identical
patches), on a held-out DB, every few epochs — plus train-set retrieval to separate "can't learn" from
"can't generalize."

Usage (CPU proof, then Kuma GPU for full-set):
  python -m masif_graph.p4.train_retrieval --data <npz> --train-ids f --val-ids f \
     --init-ckpt vicreg_sc_best_seed0.pt --epochs 40 --batch 32 --patch dense \
     --w-atom 0.5 --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --tau-c 0.1 \
     --lr 5e-4 --grad-clip 1.0 --cosine --device cpu --out r.json --save best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from masif_graph.p4.encoder import HeteroEncoder  # noqa: E402
from masif_graph.p4.objective import (Complementarity, normalize, vicreg_terms,  # noqa: E402
                                      info_nce_complex, chain_retrieval_loss, chain_patch_score_matrix)
from masif_graph.p4.dataset import ComplexP4, usable_complexes, D_AA, D_VV, D_VA  # noqa: E402
from masif_graph.p4 import eval_af3  # noqa: E402


def iface_idx(pos, col):
    return torch.unique(pos[:, col]) if pos.shape[0] else torch.zeros(0, dtype=torch.long)


# ---------------------------------------------------------------------------------------------
# retrieval eval (the target metric) — learned & frozen on identical interface patches, held-out DB
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def retrieval_metrics(enc, comp, recs, device, patch="sc"):
    enc.eval()
    # Rec restricts to the sc intersection patch (matches the docs/10 §20 retrieval test); `patch` arg
    # only controls the *training* patch, not this eval.
    emb = eval_af3.encode_all(enc, recs, device)
    T = comp.T
    pat = {}
    for r in recs:
        ih = np.unique(r.inter[:, 0]); jh = np.unique(r.inter[:, 1])
        ia = np.unique(r.inter[:, 2]); ja = np.unique(r.inter[:, 3])
        e = emb[r.cid]
        # per chain: zh/za = learned holo/af3 patch; sh/sa = frozen straight (query) holo/af3;
        # fd = frozen flipped (DB) holo. All on the same intersection interface atoms.
        pat[r.cid] = {
            "p1": {"zh": e["h1"][ih], "za": e["a1"][ia], "sh": r.hg1["desc_straight"][ih],
                   "sa": r.ag1["desc_straight"][ia], "fd": r.hg1["desc_flipped"][ih]},
            "p2": {"zh": e["h2"][jh], "za": e["a2"][ja], "sh": r.hg2["desc_straight"][jh],
                   "sa": r.ag2["desc_straight"][ja], "fd": r.hg2["desc_flipped"][jh]},
        }
    db = [(cid, role, pat[cid][role]) for cid in pat for role in ("p1", "p2")]

    def learned(zq, zd):
        if zq.shape[0] == 0 or zd.shape[0] == 0:
            return -1e9
        return float((zq @ T @ zd.t()).max(1).values.median())

    def frozen(qs, df):
        if qs.shape[0] == 0 or df.shape[0] == 0:
            return 1e9
        d = torch.sqrt(((qs[:, None, :] - df[None, :, :]) ** 2).sum(-1) + 1e-12)
        return float(d.min(1).values.median())

    def rank(state, method):
        ranks = []
        for cid in pat:
            for q, p in (("p1", "p2"), ("p2", "p1")):
                qp = pat[cid][q]; scored = []
                for (dc, dr, dp) in db:
                    if dc == cid and dr == q:
                        continue
                    if method == "learned":
                        sc = learned(qp["za"] if state == "af3" else qp["zh"], dp["zh"])
                        scored.append(((dc, dr), -sc))
                    else:
                        sc = frozen(qp["sa"] if state == "af3" else qp["sh"], dp["fd"])
                        scored.append(((dc, dr), sc))
                scored.sort(key=lambda x: x[1])
                order = [k for k, _ in scored]
                if (cid, p) in order:
                    ranks.append(order.index((cid, p)) + 1)
        r = np.array(ranks, float)
        return {"top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
                "mrr": float((1 / r).mean()), "med": float(np.median(r)), "n": len(r)}

    return {f"{m}_{s}": rank(s, m) for m in ("learned", "frozen") for s in ("holo", "af3")}


# ---------------------------------------------------------------------------------------------
def train(args):
    device = args.device
    pos_attr = "pos_sc" if args.patch == "sc" else "pos"
    val_data = args.val_data or args.data
    train_ids = usable_complexes(args.data, [l.strip() for l in open(args.train_ids) if l.strip()])
    train_c = [ComplexP4(args.data, c, device) for c in train_ids]
    # keep only complexes with non-empty interface patches on both chains
    train_c = [c for c in train_c if iface_idx(getattr(c, pos_attr), 0).numel() > 0
               and iface_idx(getattr(c, pos_attr), 1).numel() > 0]
    val_recs = []
    for cid in [l.strip() for l in open(args.val_ids) if l.strip()]:
        r = eval_af3.Rec(val_data, cid, "pos_sc", device)
        if r.ok and r.has_af3 and len(r.inter) >= 8:
            val_recs.append(r)
    print(f"train chains={2*len(train_c)} val DB={2*len(val_recs)} chains; patch={args.patch}", flush=True)

    f_atom = train_c[0].p1["atom_feat"].shape[1]; f_vert = train_c[0].p1["vert_feat"].shape[1]
    enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA, d=args.d, d_out=args.d_out, n_layers=args.layers).to(device)
    comp = Complementarity(args.d_out, tau_init=args.tau).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        enc.load_state_dict(ck["enc"]); comp.load_state_dict(ck["comp"])
        print(f"init from {args.init_ckpt}", flush=True)
    if args.freeze_tau:
        with torch.no_grad():
            comp.log_tau.fill_(math.log(args.tau))
        comp.log_tau.requires_grad_(False)

    opt = torch.optim.Adam(
        [{"params": list(enc.parameters()), "weight_decay": 1e-5},
         {"params": list(comp.parameters()), "weight_decay": args.t_wd if args.t_wd > 0 else 1e-5}], lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
             if args.cosine else None)
    rng = np.random.default_rng(args.seed)

    def show(tag, m):
        lh, la, fh, fa = m["learned_holo"], m["learned_af3"], m["frozen_holo"], m["frozen_af3"]
        print(f"  [{tag}] learned holo top5={lh['top5']:.2f} med={lh['med']:.0f} | af3 top5={la['top5']:.2f} "
              f"med={la['med']:.0f} (drop {lh['top5']-la['top5']:+.2f}) || frozen holo {fh['top5']:.2f} "
              f"af3 {fa['top5']:.2f}", flush=True)

    base = retrieval_metrics(enc, comp, val_recs, device, args.patch)
    print("[init]", flush=True); show("val", base)

    history, best = [], -1.0
    torch.manual_seed(args.seed)
    for ep in range(args.epochs):
        enc.train()
        order = rng.permutation(len(train_c))
        losses, zstds = [], []
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            patches, raws, partner = [], [], []
            for k in idx:
                c = train_c[k]
                for pid, col in (("p1", 0), ("p2", 1)):
                    zr = enc(getattr(c, pid))            # raw (n,d)
                    ii = iface_idx(getattr(c, pos_attr), col).to(device)
                    raws.append(zr[ii]); patches.append(normalize(zr)[ii])
                # partner indices: the two chains just appended are partners
                b = len(patches)
                partner += [b - 1, b - 2]
            if len(patches) < 4:
                continue
            partner = torch.tensor(partner, device=device)
            loss = chain_retrieval_loss(patches, partner, comp, tau_c=args.tau_c, tau_atom=args.tau_atom)
            # keep local correspondence: small atom-level InfoNCE on each complex in the batch
            if args.w_atom > 0:
                al = 0.0; na = 0
                for k in idx:
                    c = train_c[k]
                    z1 = normalize(enc(c.p1)); z2 = normalize(enc(c.p2))
                    tp = getattr(c, pos_attr)
                    if tp.shape[0] > 0:
                        al = al + info_nce_complex(z1, z2, tp, comp); na += 1
                if na:
                    loss = loss + args.w_atom * (al / na)
            if args.vicreg_var > 0 or args.vicreg_cov > 0:
                zc = torch.cat(raws, 0)
                v, cc = vicreg_terms(zc)
                loss = loss + args.vicreg_var * v + args.vicreg_cov * cc
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(comp.parameters()), args.grad_clip)
            opt.step()
            losses.append(float(loss))
            zstds.append(float(torch.cat(patches, 0).std(0).mean()))
        if sched:
            sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = retrieval_metrics(enc, comp, val_recs, device, args.patch)
            m["epoch"] = ep + 1; m["loss"] = float(np.mean(losses)) if losses else 0.0
            m["z_std"] = float(np.mean(zstds)) if zstds else 0.0
            history.append(m)
            print(f"[ep {ep+1:3d}] loss={m['loss']:.3f} z_std={m['z_std']:.4f}", flush=True)
            show("val", m)
            score = m["learned_af3"]["top5"]
            if score > best:
                best = score
                if args.save:
                    torch.save({"enc": enc.state_dict(), "comp": comp.state_dict(), "cfg": vars(args),
                                "metric": m}, args.save)

    out = {"init": base, "history": history, "best_learned_af3_top5": best,
           "cfg": {k: getattr(args, k) for k in ("d", "d_out", "layers", "lr", "epochs", "batch", "seed",
                    "patch", "w_atom", "tau_c", "tau_atom", "vicreg_var", "vicreg_cov", "freeze_tau",
                    "tau", "t_wd", "cosine", "init_ckpt")}}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    print(f"best held-out learned AF3 top-5 = {best:.3f} (frozen AF3 ceiling 0.64)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="holo npz dir for TRAINING complexes (ComplexP4)")
    ap.add_argument("--val-data", default=None, help="npz dir with holo+af3 for eval (default: --data)")
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32, help="complexes per batch (2x = chains = decoy pool)")
    ap.add_argument("--patch", choices=["sc", "dense"], default="dense")
    ap.add_argument("--w-atom", type=float, default=0.5, help="weight on the auxiliary atom-level InfoNCE")
    ap.add_argument("--tau-c", type=float, default=0.1, help="chain-level InfoNCE temperature")
    ap.add_argument("--tau-atom", type=float, default=0.1, help="atom soft-max temperature in the chain score")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d-out", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--vicreg-var", type=float, default=2.0)
    ap.add_argument("--vicreg-cov", type=float, default=0.04)
    ap.add_argument("--freeze-tau", action="store_true")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--t-wd", type=float, default=1e-3)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
