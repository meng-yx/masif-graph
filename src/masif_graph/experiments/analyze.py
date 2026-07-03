"""Consolidate M1 results into the gate evidence: per-complex PAIRED vertex-vs-atom deltas.

Comparing pooled or median AUCs across granularities is weaker than a paired test on the SAME
complexes (each complex contributes a vertex AUC and an atom AUC computed from the same
contacts). This reports the paired difference distribution — the cleanest gate statistic.
"""
from __future__ import annotations

import json
import numpy as np


def paired_deltas(results, positive_def="sc", min_pos=10, key="auc"):
    """For each granularity vs vertex, paired per-complex AUC deltas over complexes with n_pos>=min_pos.

    key selects which per-complex AUC to compare: "auc" (neg_mix) or "auc_randneg" (random-neg,
    identical construction across granularities -> no radius confound).
    """
    block = results["by_positive_def"][positive_def]
    if "vertex" not in block:
        return {}
    vmap = {c["id"]: c for c in block["vertex"]["per_complex"]}
    out = {}
    for g in ("atom_mean", "atom_max"):
        if g not in block:
            continue
        gmap = {c["id"]: c for c in block[g]["per_complex"]}
        deltas, pairs = [], []
        for cid, vc in vmap.items():
            gc = gmap.get(cid)
            if vc.get(key) is None or gc is None or gc.get(key) is None:
                continue
            if vc.get("n_pos", 0) < min_pos or gc.get("n_pos", 0) < min_pos:
                continue
            d = vc[key] - gc[key]  # positive => atom worse than vertex
            deltas.append(d)
            pairs.append({"id": cid, "vertex_auc": vc[key], "atom_auc": gc[key],
                          "delta": d, "n_pos_vertex": vc["n_pos"], "n_pos_atom": gc["n_pos"]})
        deltas = np.array(deltas)
        if len(deltas):
            out[g] = {
                "n_complexes": len(deltas),
                "mean_delta_vertex_minus_atom": float(deltas.mean()),
                "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "median_delta": float(np.median(deltas)),
                "atom_worse_count": int((deltas > 0).sum()),
                "atom_better_or_equal_count": int((deltas <= 0).sum()),
                "max_atom_deficit": float(deltas.max()),
                "best_atom_gain": float(-deltas.min()),
                "per_complex": sorted(pairs, key=lambda x: -x["delta"]),
            }
    return out


def summarize(results):
    lines = []
    cfg = results["config"]
    lines.append(f"N complexes used: {cfg['n_used']}  (requested {cfg['n_requested']})")
    for pdef in ("sc", "unfiltered", "atom_direct"):
        if pdef not in results["by_positive_def"]:
            continue
        block = results["by_positive_def"][pdef]
        lines.append(f"\n### positive_def = {pdef}")
        for g, b in block.items():
            lines.append(
                f"  {g:10s}  pooled_negmix={b['pooled_auc_negmix']:.4f}  "
                f"pooled_randneg={b['pooled_auc_randomneg']:.4f}  shuffled={b['shuffled_label_auc']:.4f}  "
                f"perC_median={_f(b['per_complex_auc_median'])}  perC_mean={_f(b['per_complex_auc_mean'])}  "
                f"nSpread={b['n_complexes_in_spread']}  n_pos={b['n_pos_total']}"
            )
        for key, tag in (("auc", "neg_mix"), ("auc_randneg", "randneg")):
            pd = paired_deltas(results, pdef, key=key)
            for g, d in pd.items():
                lines.append(
                    f"  PAIRED[{tag}] vertex−{g}: mean Δ={d['mean_delta_vertex_minus_atom']:+.4f} "
                    f"± {d['std_delta']:.4f} (median {d['median_delta']:+.4f}), "
                    f"atom worse in {d['atom_worse_count']}/{d['n_complexes']} complexes"
                )
    return "\n".join(lines)


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="logs/m1/m1_results.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = json.load(open(args.results))
    txt = summarize(res)
    print(txt)
    paired = {
        pd: {key: paired_deltas(res, pd, key=key) for key in ("auc", "auc_randneg")}
        for pd in ("sc", "unfiltered", "atom_direct") if pd in res["by_positive_def"]
    }
    if args.out:
        json.dump(paired, open(args.out, "w"), indent=2)
        print(f"\nwrote paired deltas -> {args.out}")
