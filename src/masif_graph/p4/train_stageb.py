"""Phase-4 Stage-B (M2) trainer: conformer-augmented invariance fine-tune.

Starts from a holo-only Stage-A checkpoint (the stabilized VICReg encoder + T) and fine-tunes with
the query embedding drawn from a random conformer c in {holo, AF3} (design §5.3), so the InfoNCE +
learned-T objective must recover the true contact regardless of which conformer produced the query.
Invariance falls out of the task; VICReg + frozen-tau + T weight-decay keep the representation alive
(the docs/10 §16 anti-collapse recipe, carried over unchanged — do NOT re-tune the optimizer).

Held-out eval is the AF3->holo separation AUC (p4.eval_af3), reported every eval step alongside the
holo->holo do-no-harm floor and the frozen ceiling on identical pairs, so we watch the north-star
Delta_robustness = learned(af3_holo) - frozen(af3_holo) converge — not a proxy.

Ablations wired as first-class flags:
  --no-atom-graph : drop atom-atom covalent edges (tests whether bond connectivity earns robustness;
                    the Phase-3 open question — the chem graph added ~nothing when only unfreezing).
  --two-conformer : match BOTH query conformers (holo & AF3) to the holo target each step (stronger push).

Usage (Kuma H100):
  python -m masif_graph.p4.train_stageb --data <npz> --train-ids f --val-ids f \
     --init-ckpt vicreg_sc_best_seed0.pt --epochs 40 --af3-prob 0.5 --train-pos sc \
     --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 --lr 5e-4 \
     --grad-clip 1.0 --cosine --device cuda --out res.json --save best.pt
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

from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, info_nce_complex, normalize, vicreg_terms
from masif_graph.p4.dataset import ComplexP4B, D_AA, D_VV, D_VA
from masif_graph.p4 import eval_af3


def _drop_atom_graph(g):
    """Ablation: remove atom-atom covalent edges (atoms stay as nodes, lose connectivity messages)."""
    g["aa_edge"] = torch.zeros(2, 0, dtype=torch.long, device=g["aa_edge"].device)
    g["aa_feat"] = torch.zeros(0, D_AA, device=g["aa_feat"].device if torch.is_tensor(g["aa_feat"]) else "cpu")
    return g


def _ablate(c, on):
    if not on:
        return
    for name in ("p1", "p2", "a1", "a2"):
        g = getattr(c, name, None)
        if g is not None:
            _drop_atom_graph(g)


def evaluate_af3(enc, comp, val_recs, device, seeds=3):
    """AF3->holo + hh separation AUC via the C4 harness (encode once, reuse across seeds/regimes)."""
    enc.eval()
    emb = eval_af3.encode_all(enc, val_recs, device)
    acc = {"hh": [], "af3_holo": []}
    for s in range(seeds):
        for rg in ("hh", "af3_holo"):
            acc[rg].append(eval_af3.evaluate(comp, val_recs, emb, device, rg, seed=1000 + s))

    def mn(vals, k):
        v = np.array([x[k] for x in vals if x[k] is not None], float)
        return float(v.mean()) if len(v) else float("nan")

    hh_l, hh_f = mn(acc["hh"], "learned_randneg"), mn(acc["hh"], "frozen_randneg")
    ah_l, ah_f = mn(acc["af3_holo"], "learned_randneg"), mn(acc["af3_holo"], "frozen_randneg")
    return {"learned_hh": hh_l, "frozen_hh": hh_f, "learned_af3_holo": ah_l, "frozen_af3_holo": ah_f,
            "learned_hh_median": mn(acc["hh"], "learned_percplx_median"),
            "learned_af3_holo_median": mn(acc["af3_holo"], "learned_percplx_median"),
            "shuffled": mn(acc["af3_holo"], "shuffled"),
            "delta_robustness_af3": ah_l - ah_f, "do_no_harm_hh": hh_l - hh_f}


def sample_confs(rng, has_af3, af3_prob):
    if not has_af3:
        return "holo", "holo"
    c1 = "af3" if rng.random() < af3_prob else "holo"
    c2 = "af3" if rng.random() < af3_prob else "holo"
    return c1, c2


def train(args):
    device = args.device
    pos_attr = "pos_sc" if args.train_pos == "sc" else "pos"
    train_ids = [l.strip() for l in open(args.train_ids) if l.strip() and not l.startswith("#")]
    val_ids = [l.strip() for l in open(args.val_ids) if l.strip() and not l.startswith("#")]
    assert not (set(train_ids) & set(val_ids)), "train/val leak!"

    t0 = time.perf_counter()
    train_c = []
    n_af3 = 0
    for cid in train_ids:
        try:
            c = ComplexP4B(args.data, cid, device, min_retention=args.min_retention)
        except FileNotFoundError:
            continue
        if c.positives("holo", "holo", pos_attr).shape[0] == 0:
            continue
        _ablate(c, args.no_atom_graph)
        train_c.append(c)
        n_af3 += int(c.has_af3)
    # held-out eval records (C4 harness). Ablate their graphs too so eval matches the trained model.
    val_recs = []
    for cid in val_ids:
        r = eval_af3.Rec(args.data, cid, "pos_sc", device)
        if r.ok and len(r.inter) >= args.min_pos:
            if args.no_atom_graph:
                for g in (r.hg1, r.hg2, getattr(r, "ag1", None), getattr(r, "ag2", None)):
                    if g is not None:
                        _drop_atom_graph(g)
            val_recs.append(r)
    print(f"train={len(train_c)} (af3-usable {n_af3}) val_recs={len(val_recs)} "
          f"loaded in {time.perf_counter()-t0:.1f}s; ablate_graph={args.no_atom_graph} "
          f"two_conf={args.two_conformer} af3_prob={args.af3_prob} pos={pos_attr}", flush=True)

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
         {"params": list(comp.parameters()), "weight_decay": args.t_wd if args.t_wd > 0 else 1e-5}],
        lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
             if args.cosine else None)
    rng = np.random.default_rng(args.seed)

    base = evaluate_af3(enc, comp, val_recs, device, args.eval_seeds)
    print(f"[init] af3_holo learned={base['learned_af3_holo']:.3f} frozen={base['frozen_af3_holo']:.3f} "
          f"Δrobust={base['delta_robustness_af3']:+.3f} | hh learned={base['learned_hh']:.3f} "
          f"dnh={base['do_no_harm_hh']:+.3f} shuf={base['shuffled']:.2f}", flush=True)

    history, diag = [], []
    best = -1e9
    bank = []
    torch.manual_seed(args.seed)
    for ep in range(args.epochs):
        enc.train()
        order = rng.permutation(len(train_c))
        losses, gnorms, zstds = [], [], []
        for k in order:
            c = train_c[k]
            b2 = torch.cat(bank, 0) if (bank and args.bank > 0) else None
            if args.two_conformer and c.has_af3:
                pairs = [("holo", "holo"), ("af3", "holo")]
            else:
                pairs = [sample_confs(rng, c.has_af3, args.af3_prob)]
            step_loss = 0.0; nsub = 0
            for c1, c2 in pairs:
                pos = c.positives(c1, c2, pos_attr)
                if pos.shape[0] == 0:
                    continue
                z1r = enc(c.graph("p1", c1)); z2r = enc(c.graph("p2", c2))
                z1, z2 = normalize(z1r), normalize(z2r)
                loss = info_nce_complex(z1, z2, pos, comp, bank2=b2, bank1=b2)
                if args.vicreg_var > 0 or args.vicreg_cov > 0:
                    v1, cc1 = vicreg_terms(z1r); v2, cc2 = vicreg_terms(z2r)
                    loss = loss + args.vicreg_var * 0.5 * (v1 + v2) + args.vicreg_cov * 0.5 * (cc1 + cc2)
                step_loss = step_loss + loss; nsub += 1
                zstds.append(float(z1.detach().std(0).mean()))
            if nsub == 0 or not torch.isfinite(step_loss):
                continue
            step_loss = step_loss / nsub
            opt.zero_grad(); step_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(comp.parameters()), args.grad_clip)
            opt.step()
            losses.append(float(step_loss)); gnorms.append(float(gn))
            if args.bank > 0:
                bank.append(z2.detach()[torch.randperm(z2.shape[0], device=device)[:args.bank]])
                if len(bank) > 16:
                    bank.pop(0)
        if sched:
            sched.step()
        with torch.no_grad():
            tau_val = float(comp.log_tau.exp().clamp(1e-2, 1.0))
            t_spec = float(torch.linalg.matrix_norm(comp.T, ord=2))
        dg = {"epoch": ep + 1, "tau": tau_val, "T_specnorm": t_spec,
              "gnorm_max": float(np.max(gnorms)) if gnorms else 0.0,
              "z_std": float(np.mean(zstds)) if zstds else 0.0,
              "loss": float(np.mean(losses)) if losses else 0.0}
        diag.append(dg)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate_af3(enc, comp, val_recs, device, args.eval_seeds)
            m["epoch"] = ep + 1; m["train_loss"] = dg["loss"]
            history.append(m)
            print(f"[ep {ep+1:3d}] loss={dg['loss']:.3f} z_std={dg['z_std']:.4f} |T|2={t_spec:.2f} "
                  f"gmax={dg['gnorm_max']:.0f} | af3 L={m['learned_af3_holo']:.3f} F={m['frozen_af3_holo']:.3f} "
                  f"Δ={m['delta_robustness_af3']:+.3f} | hh L={m['learned_hh']:.3f} dnh={m['do_no_harm_hh']:+.3f} "
                  f"shuf={m['shuffled']:.2f}", flush=True)
            # select on Δrobustness subject to the do-no-harm floor (hh must not regress badly)
            score = m["delta_robustness_af3"] if m["do_no_harm_hh"] > -0.03 else -1e9
            if score > best:
                best = score
                if args.save:
                    torch.save({"enc": enc.state_dict(), "comp": comp.state_dict(),
                                "cfg": vars(args), "metric": m}, args.save)

    out = {"train_ids": [c.cid for c in train_c], "val_ids": [r.cid for r in val_recs],
           "init": base, "history": history, "diag": diag, "best_delta_robustness": best,
           "cfg": {k: getattr(args, k) for k in ("d", "d_out", "layers", "lr", "epochs", "bank", "seed",
                    "cosine", "vicreg_var", "vicreg_cov", "freeze_tau", "tau", "t_wd", "train_pos",
                    "af3_prob", "two_conformer", "no_atom_graph", "min_retention", "init_ckpt")}}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    print(f"best held-out Δrobustness (learned-frozen @ af3_holo, dnh-gated) = {best:+.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz dir with holo + af3 graphs (p4.precompute --state)")
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--init-ckpt", default=None, help="Stage-A checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d-out", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bank", type=int, default=128)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--vicreg-var", type=float, default=2.0)
    ap.add_argument("--vicreg-cov", type=float, default=0.04)
    ap.add_argument("--freeze-tau", action="store_true")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--t-wd", type=float, default=1e-3)
    ap.add_argument("--train-pos", choices=["dense", "sc"], default="sc")
    # M2 conformer-augmentation + ablations
    ap.add_argument("--af3-prob", type=float, default=0.5, help="per-chain prob of drawing the AF3 conformer")
    ap.add_argument("--two-conformer", action="store_true", help="match BOTH holo & AF3 query to holo target")
    ap.add_argument("--no-atom-graph", action="store_true", help="ablate atom-atom covalent edges (chem-graph test)")
    ap.add_argument("--min-retention", type=float, default=0.5, help="drop AF3 conformer if retention < this (C5)")
    ap.add_argument("--min-pos", type=int, default=8, help="min intersection positives for a held-out eval complex")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
