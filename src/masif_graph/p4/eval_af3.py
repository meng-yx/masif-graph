"""Phase-4 M2 eval: holo->AF3 descriptor-separation for the from-scratch (z, T) encoder.

The M1 eval (`p4.train.evaluate`) measures holo->holo only. This measures the north-star quantity:
does the learned encoder degrade LESS than the frozen MaSIF descriptor when the *query* chain is an
AF3 model (DB stays holo crystal)? It reuses the Phase-3 identity-join (match AF3 surface atoms to
holo contacts by (chain,resseq,name) key) but scores with the learned symmetric bilinear form
s(X,Y)=zX^T T zY instead of frozen descriptor distance, AND recomputes the frozen ceiling on the
IDENTICAL intersection pairs — so learned-vs-frozen is exact in every regime.

Regimes (both chain-directions pooled per complex):
  hh        holo query x holo db   (the do-no-harm ceiling; must be preserved)
  af3_holo  AF3  query x holo db   (the DEPLOYMENT scenario; the robustness number)

Headline: Delta_robustness = learned(af3_holo) - frozen(af3_holo). B.0 asks whether this is >0 with
NO invariance training (the holo-only Stage-A checkpoint already conformer-robust?). M2 gate asks
whether Stage-B lifts it past the Phase-3 M3 +0.016 bar while keeping hh >= frozen-hh - 0.01.

Data: holo + af3 p4 graphs from `p4.precompute --state {holo,af3}` (same holo `cid` prefix), plus the
holo `{cid}__contacts.npz`. Controls: shuffled ~0.5; frozen ceiling on identical pairs; per-complex
spread; complex-level (eval ids must be disjoint from training — the 31 m1_ids are, by construction).

Usage:
  python -m masif_graph.p4.eval_af3 --data <npz_dir> --ids <file> --ckpt <vicreg_*.pt> \
      --out results.json [--pos-key pos_sc] [--seeds 3] [--min-pos 8] [--device cpu]
  (omit --ckpt for a random-init control; frozen ceiling is reported either way.)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, normalize
from masif_graph.p4.dataset import load_chain_graph, D_AA, D_VV, D_VA


def _keys(npz_path):
    z = np.load(npz_path)
    return [k.decode() if isinstance(k, bytes) else str(k) for k in z["keys"].tolist()]


def load_state_chain(data_dir, cid, state, pid, device="cpu"):
    """Return (graph_dict, keys) for one chain in one state, or None if the npz is absent."""
    p = os.path.join(data_dir, f"{cid}__{state}__{pid}.npz")
    if not os.path.exists(p):
        return None
    return load_chain_graph(p, device), _keys(p)


class Rec:
    """One complex: holo + af3 graph dicts (both chains) + the intersection positive set."""

    def __init__(self, data_dir, cid, pos_key, device="cpu"):
        self.cid = cid
        self.ok = False
        h1 = load_state_chain(data_dir, cid, "holo", "p1", device)
        h2 = load_state_chain(data_dir, cid, "holo", "p2", device)
        a1 = load_state_chain(data_dir, cid, "af3", "p1", device)
        a2 = load_state_chain(data_dir, cid, "af3", "p2", device)
        cpath = os.path.join(data_dir, f"{cid}__contacts.npz")
        if h1 is None or h2 is None or not os.path.exists(cpath):
            return
        self.hg1, hk1 = h1
        self.hg2, hk2 = h2
        cz = np.load(cpath)
        pos = cz[pos_key].reshape(-1, 2) if pos_key in cz.files else np.zeros((0, 2), np.int64)
        self.pos = pos
        self.has_af3 = a1 is not None and a2 is not None
        # intersection: holo contact rows whose BOTH atoms remain af3 surface atoms
        inter = []
        if self.has_af3 and len(pos) > 0:
            self.ag1, ak1 = a1
            self.ag2, ak2 = a2
            m1 = {k: r for r, k in enumerate(ak1)}
            m2 = {k: r for r, k in enumerate(ak2)}
            for i, j in pos:
                ki, kj = hk1[i], hk2[j]
                if ki in m1 and kj in m2:
                    inter.append((i, j, m1[ki], m2[kj]))
        self.inter = np.array(inter, dtype=np.int64) if inter else np.zeros((0, 4), np.int64)
        self.retention = (len(inter) / len(pos)) if len(pos) > 0 else float("nan")
        self.ok = True


def _frozen_score(gq, iq, gd, jd):
    """Frozen ceiling: 1/L2(desc_straight_query[iq] - desc_flipped_db[jd]) (reference flip convention)."""
    d = gq["desc_straight"][iq] - gd["desc_flipped"][jd]
    return (1.0 / (torch.sqrt((d * d).sum(1) + 1e-12) + 1e-12))


@torch.no_grad()
def encode_all(encoder, recs, device, normalize_out=True):
    """Per-chain embeddings. normalize_out=True L2-normalizes (default, for legacy callers); set
    False to return RAW embeddings so the caller can mean-center before normalizing (the DC-offset
    de-collapse — see train_retrieval.retrieval_metrics / docs/10 §21)."""
    nrm = normalize if normalize_out else (lambda x: x)
    emb = {}
    for r in recs:
        e = {"h1": nrm(encoder(r.hg1)), "h2": nrm(encoder(r.hg2))}
        if r.has_af3:
            e["a1"] = nrm(encoder(r.ag1)); e["a2"] = nrm(encoder(r.ag2))
        emb[r.cid] = e
    return emb


def _learned(zX, T, zY):
    return (zX @ T * zY).sum(1)


@torch.no_grad()
def evaluate(comp, recs, emb, device, regime, seed=0, neg_ratio=1):
    """Pooled + per-complex separation AUC for one regime, learned (z,T) and frozen, + shuffled control.

    regime: 'hh' (query=holo) or 'af3_holo' (query=af3). db is always holo.
    emb: precomputed {cid: {h1,h2,a1,a2}} normalized embeddings (encode ONCE, reuse across seeds/regimes).
    """
    rng = np.random.default_rng(seed)
    T = comp.T
    Lp, Lr, Lx, Fp, Fr, Fx = [], [], [], [], [], []
    per = []
    # cross-complex db pool: (cid, holo p1 dict, holo p2 dict, z_h1, z_h2)
    for r in recs:
        if len(r.inter) == 0:
            continue
        ih, jh, ia, ja = r.inter[:, 0], r.inter[:, 1], r.inter[:, 2], r.inter[:, 3]
        e = emb[r.cid]
        # query rows/embeddings depend on regime; db is holo
        if regime == "hh":
            zq1, zq2 = e["h1"], e["h2"]; qg1, qg2 = r.hg1, r.hg2
            qi1, qi2 = ih, jh    # C1-query rows, C2-query rows (holo)
        else:  # af3_holo
            if not r.has_af3:
                continue
            zq1, zq2 = e["a1"], e["a2"]; qg1, qg2 = r.ag1, r.ag2
            qi1, qi2 = ia, ja    # af3 rows
        zd1, zd2 = e["h1"], e["h2"]; dg1, dg2 = r.hg1, r.hg2
        di1, di2 = ih, jh        # holo db rows
        n_d2, n_d1 = zd2.shape[0], zd1.shape[0]

        # direction 1: C1 query x C2 db ; direction 2: C2 query x C1 db
        lp = torch.cat([_learned(zq1[qi1], T, zd2[di2]), _learned(zq2[qi2], T, zd1[di1])])
        fp = torch.cat([_frozen_score(qg1, qi1, dg2, di2), _frozen_score(qg2, qi2, dg1, di1)])
        # randneg: query positive atom vs random db atom
        P = len(ih)
        r2 = torch.tensor(rng.integers(0, n_d2, size=P), device=device)
        r1 = torch.tensor(rng.integers(0, n_d1, size=P), device=device)
        lr = torch.cat([_learned(zq1[qi1], T, zd2[r2]), _learned(zq2[qi2], T, zd1[r1])])
        fr = torch.cat([_frozen_score(qg1, qi1, dg2, r2), _frozen_score(qg2, qi2, dg1, r1)])
        # cross-complex neg: query atom vs a random OTHER complex's holo db chain
        others = [x for x in recs if x.cid != r.cid and len(x.inter) > 0]
        if others:
            ox = others[rng.integers(0, len(others))]
            oe = emb[ox.cid]
            oz2, og2 = oe["h2"], ox.hg2
            oz1, og1 = oe["h1"], ox.hg1
            xj2 = torch.tensor(rng.integers(0, oz2.shape[0], size=P), device=device)
            xj1 = torch.tensor(rng.integers(0, oz1.shape[0], size=P), device=device)
            lx = torch.cat([_learned(zq1[qi1], T, oz2[xj2]), _learned(zq2[qi2], T, oz1[xj1])])
            fx = torch.cat([_frozen_score(qg1, qi1, og2, xj2), _frozen_score(qg2, qi2, og1, xj1)])
            Lx.append(lx); Fx.append(fx)
        Lp.append(lp); Lr.append(lr); Fp.append(fp); Fr.append(fr)
        # per-complex learned AUC (randneg)
        y = np.r_[np.ones(lp.numel()), np.zeros(lr.numel())]
        s = np.r_[lp.cpu().numpy(), lr.cpu().numpy()]
        try:
            per.append(roc_auc_score(y, s))
        except ValueError:
            pass

    def pooled(pos, neg):
        if not pos or not neg:
            return None, None, None
        p = torch.cat(pos).cpu().numpy(); n = torch.cat(neg).cpu().numpy()
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]; s = np.r_[p, n]
        return float(roc_auc_score(y, s)), y, s

    la_r, y_r, s_r = pooled(Lp, Lr)
    la_x, _, _ = pooled(Lp, Lx)
    fa_r, _, _ = pooled(Fp, Fr)
    fa_x, _, _ = pooled(Fp, Fx)
    shuf = None
    if y_r is not None:
        ysh = y_r.copy(); rng.shuffle(ysh); shuf = float(roc_auc_score(ysh, s_r))
    return {
        "learned_randneg": la_r, "learned_cross": la_x,
        "frozen_randneg": fa_r, "frozen_cross": fa_x,
        "learned_percplx_median": float(np.median(per)) if per else float("nan"),
        "learned_percplx_mean": float(np.mean(per)) if per else float("nan"),
        "shuffled": shuf, "n_cplx": len(per),
    }


def build_encoder(recs, ckpt_path, device):
    r0 = recs[0]
    f_atom = r0.hg1["atom_feat"].shape[1]; f_vert = r0.hg1["vert_feat"].shape[1]
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device)
        cfg = ck.get("cfg", {})
        d = cfg.get("d", 64); d_out = cfg.get("d_out", 32); layers = cfg.get("layers", 4)
        tau = cfg.get("tau", 0.1)
        enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA, d=d, d_out=d_out, n_layers=layers).to(device)
        comp = Complementarity(d_out, tau_init=tau).to(device)
        enc.load_state_dict(ck["enc"]); comp.load_state_dict(ck["comp"])
        src = ckpt_path
    else:
        enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA).to(device)
        comp = Complementarity(enc.d_out).to(device)
        src = "random_init"
    return enc, comp, src


def run(args):
    device = args.device
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    recs, ret_rows = [], []
    for cid in ids:
        r = Rec(args.data, cid, args.pos_key, device)
        if not r.ok:
            print(f"  {cid}: holo/contacts unavailable — skip", flush=True)
            continue
        if r.has_af3:
            ret_rows.append((cid, len(r.pos), r.retention))
        print(f"  {cid}: holo_pos={len(r.pos)} af3={'Y' if r.has_af3 else 'N'} "
              f"inter={len(r.inter)} retention={r.retention:.2f}", flush=True)
        if len(r.inter) >= args.min_pos:
            recs.append(r)
    print(f"usable records (af3 + >= {args.min_pos} intersection positives): {len(recs)}", flush=True)
    if len(recs) < 2:
        raise SystemExit("too few usable records")

    enc, comp, src = build_encoder(recs, args.ckpt, device)
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"encoder from: {src} (params={n_par}, pos_key={args.pos_key})", flush=True)

    enc.eval()
    emb = encode_all(enc, recs, device)   # encode ONCE, reuse across seeds/regimes
    # average over seeds (negative sampling)
    regimes = ["hh", "af3_holo"]
    acc = {rg: [] for rg in regimes}
    for s in range(args.seeds):
        for rg in regimes:
            acc[rg].append(evaluate(comp, recs, emb, device, rg, seed=1000 + s))

    def ms(vals, key):
        v = np.array([x[key] for x in vals if x[key] is not None], float)
        return (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"), float("nan"))

    out = {"ids": [r.cid for r in recs], "n_complexes": len(recs), "src": src,
           "pos_key": args.pos_key, "seeds": args.seeds, "regimes": {}}
    if ret_rows:
        rr = np.array([x[2] for x in ret_rows], float); ww = np.array([x[1] for x in ret_rows], float)
        out["retention"] = {"mean": float(np.mean(rr)), "median": float(np.median(rr)),
                            "pair_weighted": float(np.sum(rr * ww) / np.sum(ww)),
                            "min": float(np.min(rr)), "n_below_0.5": int(np.sum(rr < 0.5))}
    for rg in regimes:
        e = {}
        for k in ("learned_randneg", "learned_cross", "frozen_randneg", "frozen_cross",
                  "learned_percplx_median", "shuffled"):
            m, sd = ms(acc[rg], k)
            e[k] = m; e[k + "_sd"] = sd
        e["n_cplx"] = acc[rg][0]["n_cplx"]
        out["regimes"][rg] = e

    hh, ah = out["regimes"]["hh"], out["regimes"]["af3_holo"]
    out["summary"] = {
        "learned_hh": hh["learned_randneg"], "frozen_hh": hh["frozen_randneg"],
        "learned_af3_holo": ah["learned_randneg"], "frozen_af3_holo": ah["frozen_randneg"],
        "learned_gap_holo_to_af3": hh["learned_randneg"] - ah["learned_randneg"],
        "frozen_gap_holo_to_af3": hh["frozen_randneg"] - ah["frozen_randneg"],
        "delta_robustness_learned_minus_frozen_af3": ah["learned_randneg"] - ah["frozen_randneg"],
        "do_no_harm_hh_learned_minus_frozen": hh["learned_randneg"] - hh["frozen_randneg"],
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)

    s = out["summary"]
    print("=" * 78)
    print(f"POS={args.pos_key}  src={os.path.basename(src)}  n={len(recs)}")
    print(f"  hh       : learned {hh['learned_randneg']:.3f}  frozen {hh['frozen_randneg']:.3f}  "
          f"shuf {hh['shuffled']:.2f}  med {hh['learned_percplx_median']:.3f}")
    print(f"  af3_holo : learned {ah['learned_randneg']:.3f}  frozen {ah['frozen_randneg']:.3f}  "
          f"shuf {ah['shuffled']:.2f}  med {ah['learned_percplx_median']:.3f}")
    print(f"  >>> learned holo->af3 gap {s['learned_gap_holo_to_af3']:+.3f} | "
          f"frozen gap {s['frozen_gap_holo_to_af3']:+.3f}")
    print(f"  >>> DELTA robustness (learned - frozen @ af3_holo) = {s['delta_robustness_learned_minus_frozen_af3']:+.3f}")
    print(f"  >>> do-no-harm (learned - frozen @ hh)            = {s['do_no_harm_hh_learned_minus_frozen']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", default=None, help="vicreg_*.pt checkpoint; omit for random-init control")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos_sc")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
