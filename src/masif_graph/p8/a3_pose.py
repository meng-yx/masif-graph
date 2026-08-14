"""Phase-8 A3 — rigid pose prediction from Stage-1 scores, and the F3 compute number.

Stage 2 of the funnel has to turn atom-atom complementarity scores into a pose. Before designing it,
measure what a *rigid* baseline already achieves from the existing Stage-1 embeddings — and how long
it takes, because fork F3 (docs/23 §6.5) asks whether pose prediction is affordable inside a 40k
screen at all.

Four conformer states (docs/23 D8-14). AF3 chains are superposed into the holo frame (verified: median
holo-vs-AF3 surface-atom deviation 0.6-5 A, not hundreds), so each state's own coordinates ARE its
native pose and the ground truth is well defined in every cell:
  HH holo-holo (ceiling)   AH af3-holo   HA holo-af3   AA af3-af3 (the deployment condition)

Native contacts are carried between states by the SAME intersection join the retrieval gate uses
(`Rec.inter`: holo_i, holo_j, af3_i, af3_j), so the contact set is the identical set of physical
atom pairs in all four cells.

Deliberately NOT interface-gated. Phase-4 M2's `align_one` restricted correspondences to atoms with
high MaSIF interface propensity; docs/10 §23 later traced part of frozen MaSIF's apparent edge to
exactly that kind of gated-oracle patch. Here every surface atom is a candidate, which is the honest
deployment condition.

Controls: a **random-correspondence arm** (same RANSAC, same counts, shuffled pairing) that must
fail; correspondence precision against native contacts; the starting iRMSD after the random pose.

KNOWN BIAS — read before interpreting iRMSD. Point-to-point Kabsch minimises |T(c2[j]) - c1[i]|^2,
so it drives CONTACTING ATOM CENTRES together. Real contacting atom centres are ~3.8 A apart
(measured median 3.79 A); only the *surfaces* nearly touch. MaSIF can fit this way because it fits
surface VERTICES; on atom centres the fitted pose interpenetrates — closest approach 1.06 A against
2.55 A native — costing a systematic **~1.9 A of iRMSD floor** before any correspondence error. So
the 4 A success threshold really allows only ~2.1 A of correspondence-driven error.

A global de-clash correction (slide the partner out along chain 1's interface normal until contact
distance is restored) was tried and made things WORSE: oracle iRMSD 1.9 -> 10.3 A, success
1.000 -> 0.125. The bias is per-contact along each pair's own local normal, not one global
translation, so on a curved interface a single shift destroys the fit. The correct fix is to fit at
the VERTEX level (contacts were defined by vertex proximity at 1.0 A, so the bias there is ~4x
smaller); vertex coordinates live in the reference precomputation, not in our npz. Worth doing when
Stage 1 is good enough for a 1.9 A floor to matter — at the measured 21-23 A it changes nothing.

Success is pre-registered (docs/24 §5): **fnat >= 0.3 AND iRMSD <= 4 A**.

Usage:
  python -m masif_graph.p8.a3_pose --data <npz_eval> --ids <ids> --ckpt <ckpt> \
      --tag ppionly_s0 --out logs/phase8A/a3/pose_ppionly_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from masif_graph.align.global_align import apply_T, kabsch_icp, random_pose, ransac_kabsch
from masif_graph.p4.eval_af3 import Rec, build_encoder
from masif_graph.p4.objective import normalize

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

CELLS = {"HH": ("holo", "holo"), "AH": ("af3", "holo"), "HA": ("holo", "af3"), "AA": ("af3", "af3")}
FNAT_OK, IRMSD_OK = 0.3, 4.0          # pre-registered
CONTACT_CUT = 5.0


def _cols(s1, s2):
    return (0 if s1 == "holo" else 2), (1 if s2 == "holo" else 3)


def _q(a, keys=("mean", "median", "p25", "p75")):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if a.size == 0:                      # keep the shape so callers can format unconditionally
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    o = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
         "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}
    return {k: o[k] for k in ("n",) + keys if k in o}


def pose_one(rec, cell, z, T, args, rng_seed, corr_mode="learned"):
    """One (complex, cell) pose attempt. Returns a dict of metrics + the timed prediction cost.

    corr_mode: 'learned' (the measurement) | 'random' (negative control, must fail) |
    'oracle' (POSITIVE control: the true native contact pairs — if this fails, the pose machinery
    itself is broken and no conclusion may be drawn from the 'learned' arm)."""
    s1, s2 = CELLS[cell]
    g1 = rec.hg1 if s1 == "holo" else rec.ag1
    g2 = rec.hg2 if s2 == "holo" else rec.ag2
    c1 = g1["coord"].cpu().numpy().astype(float)
    c2 = g2["coord"].cpu().numpy().astype(float)
    k1, k2 = _cols(s1, s2)
    nat = np.unique(rec.inter[:, [k1, k2]], axis=0)
    if len(nat) < 10 or len(c1) < 10 or len(c2) < 10:
        return None
    p2_iface = np.unique(nat[:, 1])
    natset = set(map(tuple, nat.tolist()))

    T_rand = random_pose(rng_seed)
    c2_start = apply_T(T_rand, c2)
    irmsd = lambda now: float(np.sqrt(np.mean(np.sum(          # noqa: E731
        (now[p2_iface] - c2[p2_iface]) ** 2, axis=1))))

    t0 = time.perf_counter()
    # --- correspondences from the learned bilinear score, over ALL surface atoms (no iface gate)
    z1 = z[(rec.cid, s1, "p1")]
    z2 = z[(rec.cid, s2, "p2")]
    S = (z1 @ T) @ z2.t()                                       # (n1, n2), higher = better
    n_take = min(args.top_corr, S.numel())
    # Hub diagnostics: how many DISTINCT partners does the score matrix ever prefer? If 500 atoms
    # all argmax onto the same 50, the matrix cannot define a pose no matter how it is fitted, and
    # that is a property of the training objective (chain-level median-of-max) rather than a bug.
    b2 = S.argmax(1).cpu().numpy()
    b1 = S.argmax(0).cpu().numpy()
    n_mutual = int((b1[b2] == np.arange(len(b2))).sum())
    hub = {"n_unique_argmax_partner": int(len(np.unique(b2))),
           "hub_concentration": float(len(np.unique(b2)) / max(len(b2), 1)),
           "n_mutual_best": n_mutual}
    if corr_mode == "mutual":
        # Mutual best: for each atom its partner's argmax, keep pairs that agree. Spreads
        # correspondences over the whole surface instead of letting one hot spot supply all of
        # them, which is closer to what MaSIF-search does and is the fairer test of the embedding.
        ii = np.nonzero(b1[b2] == np.arange(len(b2)))[0]
        jj = b2[ii]
        if len(ii) > n_take:
            keep = np.argsort(-S[ii, jj].cpu().numpy())[:n_take]
            ii, jj = ii[keep], jj[keep]
    else:
        flat = torch.topk(S.flatten(), n_take).indices.cpu().numpy()
        ii, jj = np.unravel_index(flat, tuple(S.shape))
    if corr_mode == "random":                                   # control: same count, no signal
        r = np.random.default_rng(rng_seed + 7)
        ii = r.integers(0, S.shape[0], size=len(ii))
        jj = r.integers(0, S.shape[1], size=len(jj))
    elif corr_mode in ("iface_q", "iface_both"):
        # DIAGNOSTIC with an ORACLE GATE ON WHICH ATOMS ARE INTERFACE (never on which pairs).
        # Stage-A/B atom-level InfoNCE queries ONLY true contacting atoms (objective.info_nce_complex:
        # the anchor is z1[pos[:,0]]), so the model was never trained to score an interface atom above
        # a non-interface one. Global top-k mixes that untrained axis with the trained matching axis;
        # gating separates them. Labelled as oracle-gated per docs/10 s23 — NOT a deployment number.
        q = np.unique(nat[:, 0])
        Sq = S[torch.as_tensor(q, dtype=torch.long)]
        if corr_mode == "iface_both":
            c = np.unique(nat[:, 1])
            Sq = Sq[:, torch.as_tensor(c, dtype=torch.long)]
        k = min(n_take, Sq.numel())
        fl = torch.topk(Sq.flatten(), k).indices.cpu().numpy()
        a, b = np.unravel_index(fl, tuple(Sq.shape))
        ii = q[a]
        jj = (np.unique(nat[:, 1])[b] if corr_mode == "iface_both" else b)
    elif corr_mode == "oracle":                                 # POSITIVE control: true contacts
        sel = nat if len(nat) <= n_take else nat[
            np.random.default_rng(rng_seed).choice(len(nat), n_take, replace=False)]
        ii, jj = sel[:, 0], sel[:, 1]
    corr = np.column_stack([jj, ii]).astype(np.int64)           # (src=p2 idx, tgt=p1 idx)
    if len(corr) < 4:
        # Not a failure to record as "no data" — too few correspondences IS the measurement.
        return {"cid": rec.cid, "cell": cell, "corr_mode": corr_mode, "n_native": int(len(nat)),
                "n_corr": int(len(corr)), "n_inliers": 0, "insufficient_correspondences": True,
                "corr_precision": float("nan"), "chance_precision": float("nan"),
                "precision_over_chance": float("nan"), "corr_native_dist_median": float("nan"),
                "frac_corr_within_5A": float("nan"), "irmsd_start": irmsd(c2_start),
                "irmsd": float("nan"), "irmsd_icp": float("nan"), "fnat": 0.0, "success": False,
                "predict_seconds": time.perf_counter() - t0, **hub}
    T_ransac, n_in = ransac_kabsch(c2_start, c1, corr, thr=args.ransac_thr,
                                   iters=args.ransac_iters, seed=rng_seed)
    predict_s = time.perf_counter() - t0

    c2_ransac = apply_T(T_ransac, c2_start)
    d_after = np.linalg.norm(c1[nat[:, 0]] - c2_ransac[nat[:, 1]], axis=1)
    fnat = float((d_after < CONTACT_CUT).mean())
    ir = irmsd(c2_ransac)
    # ICP kept as a diagnostic only (Phase-4 M2 found it overpacks atom centres and degrades)
    ir_icp = irmsd(apply_T(kabsch_icp(c2_start, c1, T_ransac, max_dist=args.icp_dist), c2_start))
    # Correspondence quality, measured geometrically in the NATIVE pose. Set membership in `nat`
    # understates it: `nat` is the holo/AF3 intersection, a subset of all contacts. The distance
    # between the two partners in their native positions is the honest measure. Chance is stated
    # alongside so "precision 0.002" can be read as a multiple of chance, not as a bare number.
    d_corr = np.linalg.norm(c1[corr[:, 1]] - c2[corr[:, 0]], axis=1)
    chance = float(len(nat)) / float(len(c1) * len(c2))
    prec = float(np.mean([tuple(x[::-1]) in natset for x in corr]))
    return {"cid": rec.cid, "cell": cell, "corr_mode": corr_mode, **hub,
            "n_native": int(len(nat)), "insufficient_correspondences": False,
            "n_corr": int(len(corr)), "n_inliers": int(n_in),
            "corr_precision": prec, "chance_precision": chance,
            "precision_over_chance": float(prec / chance) if chance > 0 else float("nan"),
            "corr_native_dist_median": float(np.median(d_corr)),
            "frac_corr_within_5A": float((d_corr < CONTACT_CUT).mean()),
            "irmsd_start": irmsd(c2_start), "irmsd": ir, "irmsd_icp": ir_icp, "fnat": fnat,
            "success": bool(fnat >= FNAT_OK and ir <= IRMSD_OK),
            "predict_seconds": predict_s}


@torch.no_grad()
def run(args):
    ids = [ln.strip() for ln in open(args.ids) if ln.strip() and not ln.startswith("#")]
    recs = []
    for cid in ids:
        r = Rec(args.data, cid, args.pos_key, args.device)
        if r.ok and r.has_af3 and len(r.inter) >= args.min_pos:
            recs.append(r)
        if args.limit and len(recs) >= args.limit:
            break
    print(f"usable complexes: {len(recs)}", flush=True)
    if not recs:
        raise SystemExit("no usable complexes")

    enc, comp, src = build_encoder(recs, args.ckpt, args.device)
    enc.eval()
    Tm = comp.T

    # Encode once. Centering is mandatory for the learned embedding (docs/10 §21/§24) and the mean
    # is taken over the SAME pool the retrieval gate uses, so scores are on the familiar scale.
    t0 = time.perf_counter()
    raw = {}
    for r in recs:
        raw[(r.cid, "holo", "p1")] = enc(r.hg1)
        raw[(r.cid, "holo", "p2")] = enc(r.hg2)
        raw[(r.cid, "af3", "p1")] = enc(r.ag1)
        raw[(r.cid, "af3", "p2")] = enc(r.ag2)
    mu = torch.cat(list(raw.values()), 0).mean(0, keepdim=True)
    z = {k: normalize(v - mu) for k, v in raw.items()}
    encode_s = time.perf_counter() - t0
    print(f"encoded {len(raw)} chain-states in {encode_s:.1f}s "
          f"({encode_s/max(len(raw),1):.3f}s per chain-state)", flush=True)

    rows, ctrl, orac, mutu, ifq, ifb = [], [], [], [], [], []
    for i, r in enumerate(recs):
        for cell in CELLS:
            o = pose_one(r, cell, z, Tm, args, rng_seed=args.seed)
            if o:
                rows.append(o)
        o = pose_one(r, "HH", z, Tm, args, rng_seed=args.seed, corr_mode="mutual")
        if o:
            mutu.append(o)
        for m, acc in (("iface_q", ifq), ("iface_both", ifb)):
            o = pose_one(r, "HH", z, Tm, args, rng_seed=args.seed, corr_mode=m)
            if o:
                acc.append(o)
        if args.control:
            o = pose_one(r, "HH", z, Tm, args, rng_seed=args.seed, corr_mode="random")
            if o:
                ctrl.append(o)
            o = pose_one(r, "HH", z, Tm, args, rng_seed=args.seed, corr_mode="oracle")
            if o:
                orac.append(o)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} complexes", flush=True)

    out = {"src": os.path.basename(src), "n_complexes": len(recs), "cells": {},
           "criterion": {"fnat_at_least": FNAT_OK, "irmsd_at_most": IRMSD_OK,
                         "pre_registered": "docs/24-phase8A-plan.md sec.5"},
           "config": {"top_corr": args.top_corr, "ransac_iters": args.ransac_iters,
                      "ransac_thr": args.ransac_thr, "interface_gated": False},
           "encode_seconds_per_chain_state": encode_s / max(len(raw), 1),
           "per_pose": rows}
    for cell in CELLS:
        e = [x for x in rows if x["cell"] == cell]
        if not e:
            continue
        out["cells"][cell] = {
            "n": len(e),
            "success_rate": float(np.mean([x["success"] for x in e])),
            "fnat": _q([x["fnat"] for x in e]),
            "irmsd": _q([x["irmsd"] for x in e]),
            "irmsd_icp": _q([x["irmsd_icp"] for x in e]),
            "irmsd_start": _q([x["irmsd_start"] for x in e]),
            "corr_precision": _q([x["corr_precision"] for x in e]),
            "precision_over_chance": _q([x["precision_over_chance"] for x in e]),
            "frac_corr_within_5A": _q([x["frac_corr_within_5A"] for x in e]),
            "corr_native_dist_median": _q([x["corr_native_dist_median"] for x in e]),
            "predict_seconds": _q([x["predict_seconds"] for x in e]),
            "hub_concentration": _q([x["hub_concentration"] for x in e]),
            "n_unique_argmax_partner": _q([x["n_unique_argmax_partner"] for x in e]),
            "n_mutual_best": _q([x["n_mutual_best"] for x in e]),
        }
    for name, arm in (("control_random_correspondences_HH", ctrl),
                      ("control_oracle_correspondences_HH", orac),
                      ("arm_mutual_best_HH", mutu),
                      ("diag_iface_gated_query_HH", ifq),
                      ("diag_iface_gated_both_HH", ifb)):
        if arm:
            out[name] = {
                "n": len(arm), "success_rate": float(np.mean([x["success"] for x in arm])),
                "fnat": _q([x["fnat"] for x in arm]), "irmsd": _q([x["irmsd"] for x in arm]),
                "n_corr": _q([x["n_corr"] for x in arm]),
                "corr_precision": _q([x["corr_precision"] for x in arm]),
                "precision_over_chance": _q([x["precision_over_chance"] for x in arm]),
                "frac_corr_within_5A": _q([x["frac_corr_within_5A"] for x in arm])}

    # fork F3: what a 40k-partner screen would cost with these embeddings precomputed
    med = float(np.median([x["predict_seconds"] for x in rows])) if rows else float("nan")
    out["f3_extrapolation"] = {
        "median_predict_seconds_per_pair": med,
        "core_hours_for_40k_pairs": med * 40000 / 3600,
        "note": "one query against a 40k database, embeddings precomputed; RANSAC cost is linear "
                f"in --ransac-iters (={args.ransac_iters}) and in --top-corr (={args.top_corr})",
    }

    print("=" * 78)
    print(f"A3 rigid pose baseline  src={out['src']}  n={len(recs)}  "
          f"(success = fnat>={FNAT_OK} AND iRMSD<={IRMSD_OK} A)")
    print(f"{'cell':6} {'n':>4} {'succ':>6} {'fnat_med':>9} {'iRMSD_med':>10} {'corrPrec':>9} "
          f"{'xchance':>8} {'<5A':>6} {'s/pair':>8}")
    for cell in ("HH", "AH", "HA", "AA"):
        c = out["cells"].get(cell)
        if c:
            print(f"{cell:6} {c['n']:4d} {c['success_rate']:6.3f} {c['fnat']['median']:9.3f} "
                  f"{c['irmsd']['median']:10.2f} {c['corr_precision']['median']:9.4f} "
                  f"{c['precision_over_chance']['median']:8.1f} "
                  f"{c['frac_corr_within_5A']['median']:6.3f} "
                  f"{c['predict_seconds']['median']:8.2f}")
    for lbl, name in (("random (must fail)", "control_random_correspondences_HH"),
                      ("ORACLE (must succeed)", "control_oracle_correspondences_HH"),
                      ("mutual-best arm", "arm_mutual_best_HH"),
                      ("iface-gated QUERY", "diag_iface_gated_query_HH"),
                      ("iface-gated BOTH", "diag_iface_gated_both_HH")):
        k = out.get(name)
        if k:
            print(f"-- {lbl:22s}: success {k['success_rate']:.3f}  "
                  f"fnat_med {k['fnat']['median']:.3f}  iRMSD_med {k['irmsd']['median']:.1f}  "
                  f"prec {k['corr_precision']['median']:.4f} "
                  f"({k['precision_over_chance']['median']:.1f}x chance)  "
                  f"ncorr {k['n_corr']['median']:.0f}")
    hh = out["cells"].get("HH")
    if hh:
        print(f"-- hub diagnostic (HH): only {hh['n_unique_argmax_partner']['median']:.0f} distinct "
              f"partners are ever the argmax (concentration {hh['hub_concentration']['median']:.3f}); "
              f"{hh['n_mutual_best']['median']:.0f} mutual-best pairs")
    f3 = out["f3_extrapolation"]
    print(f"-- F3: {f3['median_predict_seconds_per_pair']:.2f} s/pair -> "
          f"{f3['core_hours_for_40k_pairs']:.0f} core-h per 40k-pair screen")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pos-key", choices=["pos", "pos_sc"], default="pos")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-corr", type=int, default=1000)
    ap.add_argument("--ransac-iters", type=int, default=5000)
    ap.add_argument("--ransac-thr", type=float, default=6.0)
    ap.add_argument("--icp-dist", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true", default=True)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
