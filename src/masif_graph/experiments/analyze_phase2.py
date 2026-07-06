"""Combine Phase-2 ablation JSONs into comparison tables + per-complex-robust stats.

Usage: python -m masif_graph.experiments.analyze_phase2 <label>=<dir> [<label>=<dir> ...]
Each dir holds a phase2_results.json. Prints, per run:
  - raw mean-pool holo/repack anchor (the do-no-harm reference),
  - per-cell holo/repack AUC (randneg + negmix) and differential degradation (mean±sd over seeds),
  - per-complex MEDIAN degradation (robust to a single dominating complex),
  - do-no-harm check (full holo vs raw mean-pool), attribution (spatial vs chemistry).
"""
import sys, json
import numpy as np


def seed_mean(cell, state, key):
    return float(np.mean([s[state][key] for s in cell["seeds"]]))


def seed_sd(cell, state, key):
    return float(np.std([s[state][key] for s in cell["seeds"]]))


def per_complex_median_deg(cell, key="auc_negmix"):
    """Median over complexes of (holo - repack) per-complex AUC, averaged over seeds."""
    meds = []
    for s in cell["seeds"]:
        h = {p["cid"]: p[key] for p in s["holo"]["per_complex"]}
        r = {p["cid"]: p[key] for p in s["repack"]["per_complex"]}
        d = [h[c] - r[c] for c in h if c in r]
        if d:
            meds.append(float(np.median(d)))
    return (float(np.mean(meds)), float(np.std(meds))) if meds else (np.nan, np.nan)


def per_complex_deg_by_cell(r, cell_name, key="auc_negmix"):
    """Mean-over-seeds per-complex degradation (holo-repack) -> {cid: deg}."""
    c = r["cells"][cell_name]
    acc = {}
    for s in c["seeds"]:
        h = {p["cid"]: p[key] for p in s["holo"]["per_complex"]}
        rp = {p["cid"]: p[key] for p in s["repack"]["per_complex"]}
        for cid in h:
            if cid in rp:
                acc.setdefault(cid, []).append(h[cid] - rp[cid])
    return {cid: float(np.mean(v)) for cid, v in acc.items()}


def paired(r, base="surface_only", key="auc_negmix"):
    """For each graph cell, per-complex paired comparison vs surface_only.
    Reports #complexes where the graph degrades LESS (helps) / MORE, mean paired diff, sign-test p."""
    from math import comb
    b = per_complex_deg_by_cell(r, base, key)
    print(f"  paired per-complex (negmix DEG, graph vs {base}; '+help' = graph degrades less):")
    for name in r["cells"]:
        if name == base:
            continue
        g = per_complex_deg_by_cell(r, name, key)
        cids = [c for c in b if c in g]
        diffs = [b[c] - g[c] for c in cids]  # positive => graph degrades less (helps)
        help_n = sum(d > 1e-6 for d in diffs); hurt_n = sum(d < -1e-6 for d in diffs); n = help_n + hurt_n
        # two-sided sign test p vs 0.5
        k = max(help_n, hurt_n)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)) if n else 1.0
        print(f"    {name:13s} help={help_n:2d} hurt={hurt_n:2d} (of {len(cids)})  "
              f"mean_paired_diff={np.mean(diffs):+.4f}  sign_p={p:.3f}")


def analyze(label, path):
    r = json.load(open(f"{path}/phase2_results.json"))
    print(f"\n{'='*92}\n### {label}  ({path})   n_train={r['n_train']} n_test={r['n_test']}  "
          f"rot_inv={r['rotation_invariance_max_abs_diff']:.1e}")
    rb = r["raw_baseline"]
    print(f"RAW mean-pool anchor:  holo rn {rb['holo']['pooled_randneg']:.3f} nm {rb['holo']['pooled_negmix']:.3f}"
          f"  |  repack rn {rb['repack']['pooled_randneg']:.3f} nm {rb['repack']['pooled_negmix']:.3f}"
          f"  |  shuf holo {rb['holo']['shuffled_randneg']:.2f} repack {rb['repack']['shuffled_randneg']:.2f}")
    print(f"RAW natural collapse:  Δrn {rb['holo']['pooled_randneg']-rb['repack']['pooled_randneg']:+.3f}"
          f"  Δnm {rb['holo']['pooled_negmix']-rb['repack']['pooled_negmix']:+.3f}")
    hdr = f"{'cell':13s} {'holo_rn':>8s} {'repk_rn':>8s} {'DEGrn':>12s} {'holo_nm':>8s} {'repk_nm':>8s} {'DEGnm(pool)':>13s} {'DEGnm(med)':>13s} {'shuf':>5s}"
    print(hdr)
    rows = {}
    for name, c in r["cells"].items():
        hrn = seed_mean(c, "holo", "pooled_randneg"); rrn = seed_mean(c, "repack", "pooled_randneg")
        hnm = seed_mean(c, "holo", "pooled_negmix"); rnm = seed_mean(c, "repack", "pooled_negmix")
        drn_m, drn_s = np.mean([s["degradation_randneg"] for s in c["seeds"]]), np.std([s["degradation_randneg"] for s in c["seeds"]])
        dnm_m, dnm_s = np.mean([s["degradation_negmix"] for s in c["seeds"]]), np.std([s["degradation_negmix"] for s in c["seeds"]])
        med_m, med_s = per_complex_median_deg(c)
        shuf = seed_mean(c, "repack", "shuffled_randneg")
        rows[name] = dict(hrn=hrn, rrn=rrn, hnm=hnm, rnm=rnm, dnm=dnm_m, medm=med_m)
        print(f"{name:13s} {hrn:8.3f} {rrn:8.3f} {drn_m:+7.3f}±{drn_s:.3f} {hnm:8.3f} {rnm:8.3f} "
              f"{dnm_m:+8.3f}±{dnm_s:.3f} {med_m:+8.3f}±{med_s:.3f} {shuf:5.2f}")
    # verdict helpers
    raw_h_nm = rb['holo']['pooled_negmix']; raw_r_nm = rb['repack']['pooled_negmix']
    so, fu, sp = rows.get("surface_only"), rows.get("full"), rows.get("spatial")
    print("--- checks ---")
    print(f"do-no-harm (full holo_nm {fu['hnm']:.3f} vs RAW mean-pool holo_nm {raw_h_nm:.3f}): "
          f"{'PASS' if fu['hnm']>=raw_h_nm-1e-9 else 'FAIL (trained head < raw pooling on holo)'}")
    print(f"robustness pooled (full DEGnm {fu['dnm']:+.3f} vs surface_only {so['dnm']:+.3f}): "
          f"{'full degrades LESS' if fu['dnm']<so['dnm'] else 'full degrades MORE/equal'} "
          f"(Δ={so['dnm']-fu['dnm']:+.3f})")
    print(f"robustness pooled (spatial DEGnm {sp['dnm']:+.3f} vs surface_only {so['dnm']:+.3f}): "
          f"Δ={so['dnm']-sp['dnm']:+.3f}")
    print(f"absolute apo (best repk_nm): surface_only {so['rnm']:.3f}  spatial {sp['rnm']:.3f}  full {fu['rnm']:.3f}"
          f"  | raw {raw_r_nm:.3f}")
    print(f"attribution: spatial carries robustness? spatial DEGnm {sp['dnm']:+.3f} <= full {fu['dnm']:+.3f}? "
          f"{'yes (chemistry does not stack)' if sp['dnm']<=fu['dnm'] else 'no'}")
    paired(r)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        try:
            analyze(label, path)
        except FileNotFoundError:
            print(f"\n### {label}: no phase2_results.json yet at {path}")
