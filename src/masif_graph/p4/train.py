"""Phase-4 Stage-A trainer + held-out holo->holo separation-AUC eval (the M1 feasibility gate).

Trains the from-scratch HeteroEncoder + symmetric bilinear T with InfoNCE (holo-only correspondence)
and measures whether the learned (z, T) separates true interface contacts from negatives on HELD-OUT
complexes as well as MaSIF's frozen descriptor does (~0.90). Absolute AUC is the headline; the frozen
ceiling is computed on the IDENTICAL pos/neg index pairs (exact, not a remembered number); a shuffled
control (~0.5) guards the metric. Complex-level holdout; eval ids never seen in training.

Run (CPU smoke on Jed, or GPU on Kuma):
  python -m masif_graph.p4.train --data <npz_dir> --train-ids f --val-ids f --epochs N --device cpu|cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from masif_graph.p4.encoder import HeteroEncoder, encoder_rotation_maxdiff
from masif_graph.p4.objective import Complementarity, info_nce_complex, normalize, vicreg_terms
from masif_graph.p4.dataset import ComplexP4, usable_complexes, D_AA, D_VV, D_VA


# --------------------------------------------------------------------------------------------
# eval: separation AUC (learned + frozen ceiling on identical pairs) + shuffled control
# --------------------------------------------------------------------------------------------
def _frozen_scores(c, ii, jj, which):
    """1/L2(desc_straight1[i] - desc_flipped2[j]) for pairs (ii,jj). which=(g1,g2) dicts."""
    g1, g2 = which
    d = g1["desc_straight"][ii] - g2["desc_flipped"][jj]
    dist = torch.sqrt((d * d).sum(1) + 1e-12)
    return (1.0 / (dist + 1e-12))


@torch.no_grad()
def evaluate(encoder, comp, complexes, device, seed=0, neg_ratio=1, pos_key="pos"):
    """Held-out separation AUC: learned vs frozen ceiling, randneg + cross-complex, + shuffled control.

    pos_key selects the positive definition: 'pos' = dense all-vertex contacts (training distribution,
    frozen ceiling ~0.69) or 'pos_sc' = MaSIF sc-filtered clean contacts (the ~0.90 frozen-ceiling gate).
    """
    encoder.eval()
    rng = np.random.default_rng(seed)
    # cache embeddings per complex
    emb = {}
    for c in complexes:
        z1 = normalize(encoder(c.p1)); z2 = normalize(encoder(c.p2))
        emb[c.cid] = (z1, z2)

    L_pos, L_neg_r, L_neg_x = [], [], []   # learned scores
    F_pos, F_neg_r, F_neg_x = [], [], []   # frozen scores
    per_cplx = []
    for c in complexes:
        z1, z2 = emb[c.cid]
        n1, n2 = z1.shape[0], z2.shape[0]
        pos = getattr(c, pos_key)
        if pos.shape[0] == 0:
            continue
        i, j = pos[:, 0], pos[:, 1]
        posset = set(map(tuple, pos.tolist()))
        # positives (per-pair bilinear score: (z1_i^T T) · z2_j)
        lp = (z1[i] @ comp.T * z2[j]).sum(1)
        fp = _frozen_scores(c, i, j, (c.p1, c.p2))
        # randneg (same complex, random non-positive cross pairs)
        n_neg = pos.shape[0] * neg_ratio
        ri = torch.tensor(rng.integers(0, n1, size=n_neg), device=device)
        rj = torch.tensor(rng.integers(0, n2, size=n_neg), device=device)
        keep = torch.tensor([ (int(a),int(b)) not in posset for a,b in zip(ri.tolist(), rj.tolist()) ],
                            device=device)
        ri, rj = ri[keep], rj[keep]
        lr = (z1[ri] @ comp.T * z2[rj]).sum(1)
        fr = _frozen_scores(c, ri, rj, (c.p1, c.p2))
        # cross-complex neg (chain1 of c vs chain2 of another complex)
        others = [x for x in complexes if x.cid != c.cid]
        cx = others[rng.integers(0, len(others))] if others else c
        z2x = emb[cx.cid][1]
        m = min(n_neg, n1, z2x.shape[0])
        xi = torch.tensor(rng.integers(0, n1, size=m), device=device)
        xj = torch.tensor(rng.integers(0, z2x.shape[0], size=m), device=device)
        lx = (z1[xi] @ comp.T * z2x[xj]).sum(1)
        fx = _frozen_scores(c, xi, xj, (c.p1, cx.p2))

        L_pos.append(lp); L_neg_r.append(lr); L_neg_x.append(lx)
        F_pos.append(fp); F_neg_r.append(fr); F_neg_x.append(fx)
        # per-complex AUC (learned, randneg)
        if lr.numel() > 0:
            y = np.r_[np.ones(lp.numel()), np.zeros(lr.numel())]
            s = np.r_[lp.cpu().numpy(), lr.cpu().numpy()]
            try:
                per_cplx.append(roc_auc_score(y, s))
            except ValueError:
                pass

    def auc(pos, neg):
        p = torch.cat(pos).cpu().numpy(); n = torch.cat(neg).cpu().numpy()
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]; s = np.r_[p, n]
        return float(roc_auc_score(y, s)), y, s

    la_r, y_r, s_r = auc(L_pos, L_neg_r)
    la_x, _, _ = auc(L_pos, L_neg_x)
    fa_r, _, _ = auc(F_pos, F_neg_r)
    fa_x, _, _ = auc(F_pos, F_neg_x)
    # shuffled control on the learned randneg pooled set
    ysh = y_r.copy(); rng.shuffle(ysh)
    shuf = float(roc_auc_score(ysh, s_r))
    return {
        "learned_randneg": la_r, "learned_cross": la_x,
        "frozen_randneg": fa_r, "frozen_cross": fa_x,
        "learned_percplx_median": float(np.median(per_cplx)) if per_cplx else float("nan"),
        "shuffled": shuf, "n_cplx": len(per_cplx),
    }


# --------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------
def train(args):
    device = args.device
    train_ids = [l.strip() for l in open(args.train_ids) if l.strip() and not l.startswith("#")]
    val_ids = [l.strip() for l in open(args.val_ids) if l.strip() and not l.startswith("#")]
    train_ids = usable_complexes(args.data, train_ids)
    val_ids = usable_complexes(args.data, val_ids)
    assert not (set(train_ids) & set(val_ids)), "train/val leak!"
    print(f"train={len(train_ids)} val={len(val_ids)} complexes; device={device}", flush=True)

    t0 = time.perf_counter()
    stream = getattr(args, "stream", False)
    train_pos_attr = "pos_sc" if getattr(args, "train_pos", "dense") == "sc" else "pos"
    print(f"training positives = {train_pos_attr} "
          f"({'sc-filtered' if train_pos_attr=='pos_sc' else 'dense all-vertex contacts'})", flush=True)
    # val always preloaded (small, reused every eval). Train preloaded unless --stream, in which case
    # keep only ids and load each complex on-demand per step (drop after) so the ~4,700-complex /
    # ~14 GB npz set never sits in memory at once.
    val_c = [ComplexP4(args.data, c, device) for c in val_ids]
    if stream:
        # preload all train complexes to CPU RAM once (~40 GB for 4,700 → needs --mem≈90G), then move
        # each to GPU per step. Load-once (no per-epoch disk I/O) + bounded GPU memory.
        train_c = [ComplexP4(args.data, c, "cpu") for c in train_ids]
        f_atom = train_c[0].p1["atom_feat"].shape[1]; f_vert = train_c[0].p1["vert_feat"].shape[1]
    else:
        train_c = [ComplexP4(args.data, c, device) for c in train_ids]
        f_atom = train_c[0].p1["atom_feat"].shape[1]; f_vert = train_c[0].p1["vert_feat"].shape[1]
    print(f"loaded val={len(val_c)} train={'stream:'+str(len(train_c)) if stream else len(train_c)} "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)
    torch.manual_seed(args.seed)
    enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA, d=args.d, d_out=args.d_out,
                        n_layers=args.layers).to(device)
    comp = Complementarity(args.d_out, tau_init=args.tau).to(device)
    # anti-collapse stabilizers (docs/10 root cause): freeze the learnable temperature (it ran to the
    # 0.01 floor) at a fixed value so it can't amplify a collapsed representation.
    if args.freeze_tau:
        with torch.no_grad():
            comp.log_tau.fill_(math.log(args.tau))
        comp.log_tau.requires_grad_(False)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in comp.parameters())
    print(f"encoder+T params={n_params} (f_atom={f_atom} f_vert={f_vert} d={args.d} d_out={args.d_out} L={args.layers})", flush=True)
    print(f"stabilizers: vicreg_var={args.vicreg_var} vicreg_cov={args.vicreg_cov} "
          f"freeze_tau={args.freeze_tau}@{args.tau} t_wd={args.t_wd}", flush=True)

    # T (bilinear form) gets its own weight-decay group to bound the unbounded ||T|| growth seen in diag.
    opt = torch.optim.Adam(
        [{"params": list(enc.parameters()), "weight_decay": 1e-5},
         {"params": list(comp.parameters()), "weight_decay": args.t_wd if args.t_wd > 0 else 1e-5}],
        lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
             if getattr(args, "cosine", False) else None)
    rng = np.random.default_rng(args.seed)

    def eval_both(tag=""):
        m = {"dense": evaluate(enc, comp, val_c, device, seed=1, pos_key="pos"),
             "sc": evaluate(enc, comp, val_c, device, seed=1, pos_key="pos_sc")}
        return m

    # baseline eval (random init) — frozen ceiling on both pos sets; learned should be ~chance/shuffled
    base = eval_both()
    print(f"[init] dense: learned={base['dense']['learned_randneg']:.3f} frozen={base['dense']['frozen_randneg']:.3f} "
          f"| sc: learned={base['sc']['learned_randneg']:.3f} frozen={base['sc']['frozen_randneg']:.3f} "
          f"shuf={base['dense']['shuffled']:.3f}", flush=True)

    history = []
    diag = []   # per-epoch instability diagnostics (name the divergence trigger)
    best = -1
    bank = []  # FIFO of detached normalized z2 for cross-complex negatives
    step_times = []
    for ep in range(args.epochs):
        enc.train()
        order = rng.permutation(len(train_c))
        losses = []
        gnorms = []   # pre-clip grad norm per step (gradient-explosion signal)
        zstds = []    # embedding spread per step (collapse signal: ->0 = all z identical)
        for k in order:
            c = train_c[k].to(device) if stream else train_c[k]
            ts = time.perf_counter()
            z1r = enc(c.p1); z2r = enc(c.p2)            # raw (pre-normalize) for VICReg anti-collapse
            z1 = normalize(z1r); z2 = normalize(z2r)
            b2 = torch.cat(bank, 0) if (bank and args.bank > 0) else None
            tpos = getattr(c, train_pos_attr)          # --train-pos selects dense (pos) or sc (pos_sc)
            if tpos.shape[0] == 0:                      # sc set can be empty for some complexes
                continue
            loss = info_nce_complex(z1, z2, tpos, comp, bank2=b2, bank1=b2)
            if args.vicreg_var > 0 or args.vicreg_cov > 0:   # anti-collapse (docs/10 root cause)
                v1, cc1 = vicreg_terms(z1r); v2, cc2 = vicreg_terms(z2r)
                loss = loss + args.vicreg_var * 0.5 * (v1 + v2) + args.vicreg_cov * 0.5 * (cc1 + cc2)
            if not torch.isfinite(loss):   # insurance: never let one bad step corrupt the weights
                continue
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(comp.parameters()),
                                                getattr(args, "grad_clip", 5.0))
            opt.step()
            losses.append(float(loss))
            gnorms.append(float(gn))                       # total grad-norm BEFORE clipping
            zstds.append(float(z1.detach().std(0).mean())) # mean per-dim std across chain-1 atoms
            step_times.append(time.perf_counter() - ts)
            if args.bank > 0:
                bank.append(z2.detach()[torch.randperm(z2.shape[0], device=device)[:args.bank]])
                if len(bank) > 16:
                    bank.pop(0)
        # ---- per-epoch instability diagnostic (every epoch, 1-epoch resolution through divergence) ----
        with torch.no_grad():
            tau_val = float(comp.log_tau.exp().clamp(1e-2, 1.0))
            t_spec = float(torch.linalg.matrix_norm(comp.T, ord=2))   # spectral norm of the bilinear form
        dg = {"epoch": ep + 1,
              "tau": tau_val, "T_specnorm": t_spec,
              "gnorm_med": float(np.median(gnorms)) if gnorms else 0.0,
              "gnorm_max": float(np.max(gnorms)) if gnorms else 0.0,
              "z_std": float(np.mean(zstds)) if zstds else 0.0,
              "loss": float(np.mean(losses)) if losses else 0.0}
        diag.append(dg)
        print(f"[diag ep {ep+1:3d}] tau={tau_val:.4f} |T|2={t_spec:.2f} "
              f"gnorm med={dg['gnorm_med']:.2f} max={dg['gnorm_max']:.1f} "
              f"z_std={dg['z_std']:.4f} loss={dg['loss']:.2f}", flush=True)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = eval_both()
            m["epoch"] = ep + 1; m["train_loss"] = float(np.mean(losses))
            history.append(m)
            sc, dn = m["sc"], m["dense"]
            print(f"[ep {ep+1:3d}] loss={m['train_loss']:.3f} "
                  f"| SC learned={sc['learned_randneg']:.3f} frozen={sc['frozen_randneg']:.3f} "
                  f"med={sc['learned_percplx_median']:.3f} "
                  f"| dense learned={dn['learned_randneg']:.3f} frozen={dn['frozen_randneg']:.3f} "
                  f"| shuf={dn['shuffled']:.3f}", flush=True)
            # track best by the SC-filtered gate (the ~0.90 comparison)
            if sc["learned_randneg"] > best:
                best = sc["learned_randneg"]
                if args.save:
                    torch.save({"enc": enc.state_dict(), "comp": comp.state_dict(),
                                "cfg": vars(args), "metric": m}, args.save)

    med_step = float(np.median(step_times))
    out = {
        "train_ids": train_ids, "val_ids": val_ids, "n_params": n_params,
        "init": base, "history": history, "diag": diag, "best_sc_learned_randneg": best,
        "median_step_sec": med_step, "n_steps": len(step_times), "device": device,
        "cfg": {k: getattr(args, k) for k in ("d", "d_out", "layers", "lr", "epochs", "bank", "seed", "cosine",
                                              "vicreg_var", "vicreg_cov", "freeze_tau", "tau", "t_wd", "train_pos")},
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}", flush=True)
    print(f"median step {med_step*1000:.0f} ms; best held-out SC learned_randneg={best:.3f} "
          f"(SC frozen ceiling ~0.90)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d-out", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bank", type=int, default=64, help="cross-complex neg bank size per complex (0=off)")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay to 1%% of peak (stabilizer)")
    ap.add_argument("--grad-clip", type=float, default=5.0, help="grad-norm clip (lower = more stable)")
    # anti-collapse stabilizers (docs/10 root cause: representation collapse + tau-floor + unbounded ||T||)
    ap.add_argument("--vicreg-var", type=float, default=0.0,
                    help="VICReg variance coef (hinge std>=1 per dim); 0=off. THE anti-collapse lever.")
    ap.add_argument("--vicreg-cov", type=float, default=0.0,
                    help="VICReg covariance coef (decorrelate dims); 0=off.")
    ap.add_argument("--freeze-tau", action="store_true",
                    help="freeze temperature at --tau (stops the learned tau running to the 0.01 floor).")
    ap.add_argument("--tau", type=float, default=0.1, help="temperature value (init, or fixed if --freeze-tau).")
    ap.add_argument("--t-wd", type=float, default=0.0,
                    help="extra weight decay on the bilinear T params (bounds ||T|| growth); 0=default 1e-5.")
    ap.add_argument("--stream", action="store_true", help="lazy-load train complexes per step (full-set scale)")
    ap.add_argument("--train-pos", choices=["dense", "sc"], default="dense",
                    help="positive set for the InfoNCE loss: dense all-vertex contacts or sc-filtered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
