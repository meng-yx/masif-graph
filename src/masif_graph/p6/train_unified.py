"""Phase-6 C(c) — retrain the unified 26-D encoder on the COMBINED PPI + protein-ligand corpus.

Same objective and anti-collapse recipe as the Phase-4 scale-up (chain-level InfoNCE over interface
patches + auxiliary atom InfoNCE + VICReg; freeze-tau @0.1, T weight-decay, lr 5e-4 cosine,
d=64/d_out=32/L=4, DC-offset centering). What is new is only the corpus and the validation:

* **corpus** — protein-ligand complexes arrive in the identical two-graphs-plus-contacts form
  (`p6/pl_graph.py`), so they are just extra entries in the id list. Each batch is built to hold
  both types (`--pl-frac`), because the chain-level loss uses the in-batch chains as its hard
  decoy pool: a batch of only-PPI or only-ligands would never ask the model to tell a true partner
  from a plausible wrong one of the other type.
* **validation** — `p6/mixed_bench.py` retrieval on a held-out mixture, reported per type, so
  "PPI still works" and "protein-ligand works" cannot hide inside one pooled number.

Divergences from the Phase-4 recipe, logged as required:
  * `--max-patch` (default 128) subsamples interface atoms per chain. Phase 4 was uncapped, but the
    chain score matrix is O(N^2 * n_a * n_b) and PPI dense patches run to many hundreds of atoms
    while a ligand has ~25 — capping bounds the cost and stops PPI patches from dominating the
    wall-clock. Set 0 to reproduce Phase 4 exactly.
  * model selection is on **mixed** held-out MRR (`all`), not AF3 top-5, since there is no AF3
    state for protein-ligand complexes.
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

from masif_graph.p4.dataset import ComplexP4, D_AA, D_VV, D_VA  # noqa: E402
from masif_graph.p4.encoder import HeteroEncoder  # noqa: E402
from masif_graph.p4.objective import (Complementarity, chain_patch_score_matrix,  # noqa: E402
                                      chain_retrieval_loss, info_nce_complex, normalize,
                                      vicreg_terms)
from masif_graph.p6 import mixed_bench  # noqa: E402


def _read_ids(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def load_split(sources, device="cpu", min_pos=1):
    """sources: [(npz_dir, ids_file, kind)] -> ([(cid, kind, ComplexP4)], n_skipped)."""
    out, skipped = [], 0
    for d, idf, kind in sources:
        for cid in _read_ids(idf):
            try:
                c = ComplexP4(d, cid, device)
            except (FileNotFoundError, OSError, KeyError):
                skipped += 1
                continue
            if c.pos.shape[0] < min_pos:
                skipped += 1
                continue
            out.append((cid, kind, c))
    return out, skipped


def iface_idx(pos, col, max_patch, rng):
    if pos.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    idx = torch.unique(pos[:, col])
    if max_patch and idx.numel() > max_patch:
        keep = rng.choice(idx.numel(), max_patch, replace=False)
        idx = idx[torch.tensor(np.sort(keep))]
    return idx


def make_batches(kinds, batch, pl_frac, rng):
    """Interleave the two corpora so every batch contains both types (hard decoys of each kind).

    An epoch is long enough to cover the LARGER pool once; the smaller one wraps around. Sizing the
    epoch by the smaller pool would silently show the model only a prefix of the larger corpus."""
    ppi = np.flatnonzero(kinds == "ppi")
    pl = np.flatnonzero(kinds == "pl")
    rng.shuffle(ppi); rng.shuffle(pl)
    n_pl = int(round(batch * pl_frac))
    n_ppi = batch - n_pl
    if len(ppi) == 0 or n_ppi <= 0:
        n_ppi, n_pl = 0, batch
    if len(pl) == 0 or n_pl <= 0:
        n_ppi, n_pl = batch, 0
    n_batches = max(1, max(len(ppi) // n_ppi if n_ppi else 0, len(pl) // n_pl if n_pl else 0))

    def take(pool, start, k):
        return [pool[(start + t) % len(pool)] for t in range(k)] if len(pool) else []

    out = []
    for b in range(n_batches):
        idx = take(ppi, b * n_ppi, n_ppi) + take(pl, b * n_pl, n_pl)
        if len(idx) >= 4:
            out.append(np.array(idx))
    return out


def train(args):
    dev = args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    sources = []
    if args.ppi_ids:
        sources.append((args.ppi_data, args.ppi_ids, "ppi"))
    if args.pl_ids:
        sources.append((args.pl_data, args.pl_ids, "pl"))
    train_recs, sk = load_split(sources, "cpu")
    val_sources = []
    if args.val_ppi_ids:
        val_sources.append((args.val_ppi_data or args.ppi_data, args.val_ppi_ids, "ppi"))
    if args.val_pl_ids:
        val_sources.append((args.val_pl_data or args.pl_data, args.val_pl_ids, "pl"))
    val_recs, vsk = load_split(val_sources, dev, min_pos=args.min_val_pos)

    kinds = np.array([k for _, k, _ in train_recs])
    print(f"train: {len(train_recs)} complexes ({int((kinds=='ppi').sum())} ppi / "
          f"{int((kinds=='pl').sum())} pl; {sk} skipped) | val: {len(val_recs)} ({vsk} skipped)",
          flush=True)
    if not train_recs:
        raise SystemExit("no training complexes loaded")

    f_atom = train_recs[0][2].p1["atom_feat"].shape[1]
    enc = HeteroEncoder(f_atom, 4, D_AA, D_VV, D_VA, d=args.d, d_out=args.d_out,
                        n_layers=args.layers).to(dev)
    comp = Complementarity(args.d_out, tau_init=args.tau).to(dev)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=dev)
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
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=args.lr * 0.01)
             if args.cosine else None)

    def evaluate():
        if not val_recs:
            return {}
        pat, z = mixed_bench.build_patches(enc, comp, val_recs, dev, center=args.center,
                                           pos_key=args.patch_key, max_patch=args.val_max_patch)
        r = mixed_bench.summarize(mixed_bench.retrieve(pat, comp, same_type_db=True))
        r["z_std_val"] = z
        return r

    def show(tag, m):
        parts = []
        for g in ("all", "ppi", "pl"):
            if g in m:
                parts.append(f"{g} top5={m[g]['top5']:.3f} mrr={m[g]['mrr']:.3f} "
                             f"med={m[g]['median_rank']:.0f}")
        print(f"  [{tag}] " + " | ".join(parts) + f" | z_std={m.get('z_std_val', 0):.4f}", flush=True)

    base = evaluate()
    print("[init]", flush=True); show("val", base)

    history, best, best_ep = [], -1.0, -1
    for ep in range(args.epochs):
        enc.train()
        t0 = time.time()
        losses, zstds, tr_top1 = [], [], []
        for idx in make_batches(kinds, args.batch, args.pl_frac, rng):
            raws, partner, zfull = [], [], []
            for k in idx:
                c = train_recs[k][2].to(dev)
                pos = getattr(c, args.patch_key)
                i1 = iface_idx(pos, 0, args.max_patch, rng).to(dev)
                i2 = iface_idx(pos, 1, args.max_patch, rng).to(dev)
                if i1.numel() == 0 or i2.numel() == 0:
                    continue
                z1r, z2r = enc(c.p1), enc(c.p2)
                raws.append(z1r[i1]); raws.append(z2r[i2])
                b = len(raws); partner += [b - 1, b - 2]       # the two chains partner each other
                zfull.append((normalize(z1r), normalize(z2r), pos))
            if len(raws) < 4:
                continue
            if args.center:
                mu = torch.cat(raws, 0).mean(0, keepdim=True)
                patches = [normalize(r - mu) for r in raws]
            else:
                patches = [normalize(r) for r in raws]
            partner_t = torch.tensor(partner, device=dev)
            loss = chain_retrieval_loss(patches, partner_t, comp, tau_c=args.tau_c,
                                        tau_atom=args.tau_atom)
            if args.w_atom > 0:
                al, na = 0.0, 0
                for (z1n, z2n, tp) in zfull:
                    if tp.shape[0] > 0:
                        al = al + info_nce_complex(z1n, z2n, tp.to(dev), comp); na += 1
                if na:
                    loss = loss + args.w_atom * (al / na)
            if args.vicreg_var > 0 or args.vicreg_cov > 0:
                v, cc = vicreg_terms(torch.cat(raws, 0))
                loss = loss + args.vicreg_var * v + args.vicreg_cov * cc
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(comp.parameters()),
                                           args.grad_clip)
            opt.step()
            losses.append(float(loss))
            zstds.append(float(torch.cat(patches, 0).std(0).mean()))
            if len(tr_top1) < 4:
                with torch.no_grad():
                    M = chain_patch_score_matrix([p.detach() for p in patches], comp, args.tau_atom)
                    M = M - torch.diag(torch.full((M.shape[0],), float("inf"), device=M.device))
                    tr_top1.append(float((M.argmax(1) == partner_t).float().mean()))
        if sched:
            sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate()
            m.update({"epoch": ep + 1, "loss": float(np.mean(losses)) if losses else 0.0,
                      "z_std": float(np.mean(zstds)) if zstds else 0.0,
                      "train_top1": float(np.mean(tr_top1)) if tr_top1 else 0.0,
                      "T_norm": float(comp.T.norm()), "tau": float(comp.log_tau.exp()),
                      "sec": round(time.time() - t0, 1)})
            history.append(m)
            print(f"[ep {ep+1:3d}] loss={m['loss']:.3f} z_std={m['z_std']:.4f} "
                  f"train_top1={m['train_top1']:.3f} |T|={m['T_norm']:.2f} {m['sec']}s", flush=True)
            show("val", m)
            score = m.get(args.select_on, {}).get("mrr", -1.0) if args.select_on in m else -1.0
            if score > best:
                best, best_ep = score, ep + 1
                if args.save:
                    torch.save({"enc": enc.state_dict(), "comp": comp.state_dict(),
                                "cfg": vars(args), "metric": m}, args.save)

    out = {"init": base, "history": history, "best": best, "best_epoch": best_ep,
           "select_on": args.select_on, "cfg": vars(args)}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print("wrote", args.out, flush=True)
    print(f"best mixed held-out {args.select_on} MRR = {best:.3f} (epoch {best_ep})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi-data"); ap.add_argument("--ppi-ids")
    ap.add_argument("--pl-data"); ap.add_argument("--pl-ids")
    ap.add_argument("--val-ppi-data"); ap.add_argument("--val-ppi-ids")
    ap.add_argument("--val-pl-data"); ap.add_argument("--val-pl-ids")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32, help="complexes/batch (2x = decoy chain pool)")
    ap.add_argument("--pl-frac", type=float, default=0.5, help="fraction of each batch that is P-L")
    ap.add_argument("--patch-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--max-patch", type=int, default=128, help="0 = uncapped (Phase-4 behaviour)")
    ap.add_argument("--val-max-patch", type=int, default=128)
    ap.add_argument("--min-val-pos", type=int, default=8)
    ap.add_argument("--w-atom", type=float, default=0.5)
    ap.add_argument("--tau-c", type=float, default=0.07)
    ap.add_argument("--tau-atom", type=float, default=0.1)
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
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--select-on", default="all", choices=["all", "ppi", "pl"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save", default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
