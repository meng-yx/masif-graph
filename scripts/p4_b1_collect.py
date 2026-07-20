"""Collect the B.1 Stage-B matrix. Reports init / final-epoch / best (dnh-gated) for every run — the
full curve, not just best-epoch (docs/10 §15 lesson: best is a selection-biased spike of a noisy eval).
Aggregates full-graph vs no-atom-graph to answer: does atom connectivity earn robustness?
"""
import glob
import json
import os

import numpy as np

D = "logs/phase4/m2_b1"
runs = {}
for f in sorted(glob.glob(f"{D}/b1_*.json")):
    if os.path.basename(f) == "b1_summary.json":
        continue
    tag = os.path.basename(f)[3:-5]  # strip 'b1_' and '.json'
    try:
        r = json.load(open(f))
        hist = r["history"]
    except (json.JSONDecodeError, KeyError):
        continue  # partial file mid-write (race with a still-finishing run) — skip this pass
    if not hist:
        continue
    init = r["init"]
    fin = hist[-1] if hist else init
    # best final-selectable = max dnh-gated Δrobust over eval points (report WITH its epoch + the final)
    gated = [(h["delta_robustness_af3"] if h["do_no_harm_hh"] > -0.05 else -9, h) for h in hist]
    best_h = max(gated, key=lambda x: x[0])[1] if hist else init
    runs[tag] = {"init": init, "final": fin, "best": best_h,
                 "af3_final": fin["learned_af3_holo"], "af3_init": init["learned_af3_holo"],
                 "hh_final": fin["learned_hh"], "dnh_final": fin["do_no_harm_hh"],
                 "gap_final": fin["learned_hh"] - fin["learned_af3_holo"],
                 "delta_final": fin["delta_robustness_af3"], "delta_best": best_h["delta_robustness_af3"],
                 "frozen_af3": fin["frozen_af3_holo"], "frozen_hh": fin["frozen_hh"],
                 "n_evals": len(hist)}

print(f"{'run':22s} {'af3_init':>8} {'af3_fin':>8} {'hh_fin':>7} {'gap_fin':>8} {'Δ_fin':>7} {'Δ_best':>7} {'dnh':>7}")
for tag in sorted(runs):
    r = runs[tag]
    print(f"{tag:22s} {r['af3_init']:8.3f} {r['af3_final']:8.3f} {r['hh_final']:7.3f} "
          f"{r['gap_final']:+8.3f} {r['delta_final']:+7.3f} {r['delta_best']:+7.3f} {r['dnh_final']:+7.3f}")

fr_af3 = np.mean([r["frozen_af3"] for r in runs.values()]) if runs else float("nan")
fr_hh = np.mean([r["frozen_hh"] for r in runs.values()]) if runs else float("nan")
print(f"\nfrozen ceiling (identical pairs): hh={fr_hh:.3f} af3={fr_af3:.3f} "
      f"(frozen holo→af3 gap {fr_hh-fr_af3:+.3f})")


def agg(pred):
    sel = [r for t, r in runs.items() if pred(t)]
    if not sel:
        return None
    return (np.mean([r["af3_final"] for r in sel]), np.std([r["af3_final"] for r in sel]),
            np.mean([r["gap_final"] for r in sel]), np.mean([r["hh_final"] for r in sel]),
            np.mean([r["dnh_final"] for r in sel]))


print("\n=== full-graph vs no-atom-graph (final-epoch, mean over confs+seeds) — the chem-graph test ===")
for name, pred in [("full-graph", lambda t: t.startswith("full")),
                   ("no-atom-graph", lambda t: t.startswith("nograph"))]:
    a = agg(pred)
    if a:
        print(f"  {name:14s} af3={a[0]:.3f}±{a[1]:.3f}  gap(hh-af3)={a[2]:+.3f}  hh={a[3]:.3f}  dnh={a[4]:+.3f}")

print("\n=== 1-conf vs 2-conf (final-epoch, mean over graph+seeds) ===")
for name, pred in [("1conf", lambda t: "_1conf_" in t), ("2conf", lambda t: "_2conf_" in t)]:
    a = agg(pred)
    if a:
        print(f"  {name:14s} af3={a[0]:.3f}±{a[1]:.3f}  gap={a[2]:+.3f}  hh={a[3]:.3f}  dnh={a[4]:+.3f}")

json.dump(runs, open(f"{D}/b1_summary.json", "w"), indent=2, default=float)
print(f"\nwrote {D}/b1_summary.json ; runs collected: {len(runs)}/8")
