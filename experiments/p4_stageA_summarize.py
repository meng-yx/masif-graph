"""Summarize Stage-A feasibility runs (3 seeds) + estimate full-set GPU cost.

Reads stageA_result_seed*.json (train.py output) and reports, per seed and mean±std:
  - held-out SC-filtered learned AUC (best-epoch + final-epoch) vs the frozen 0.947 ceiling,
  - held-out dense learned AUC (final) vs the frozen dense ceiling,
  - per-complex median, shuffled control,
  - H100 median_step_sec.
Then extrapolates the full-set (default 4,943-complex) Stage-A GPU cost from the measured step time.

  python experiments/p4_stageA_summarize.py --dir /work/upthomae/Meng/phase4 --full-n 4943
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

CHF_PER_H100_HR = 0.52


def _final(h, key):  # last eval entry's metric
    return h[-1][key]


def summarize(dirpath, full_n, seeds_epochs=(50, 100, 150), n_seed_train=90):
    files = sorted(glob.glob(os.path.join(dirpath, "stageA_result_seed*.json")))
    if not files:
        return {"error": f"no stageA_result_seed*.json in {dirpath}"}
    rows = []
    for f in files:
        d = json.load(open(f))
        h = d["history"]
        seed = d["cfg"]["seed"]
        # best SC by learned_randneg over epochs
        sc_series = [(e["sc"]["learned_randneg"], e["epoch"], e) for e in h]
        best_sc, best_ep, best_entry = max(sc_series, key=lambda x: x[0])
        rows.append({
            "seed": seed,
            "sc_best": best_sc, "sc_best_epoch": best_ep,
            "sc_final": _final(h, "sc")["learned_randneg"],
            "sc_frozen": _final(h, "sc")["frozen_randneg"],
            "sc_percplx_med_final": _final(h, "sc")["learned_percplx_median"],
            "dense_final": _final(h, "dense")["learned_randneg"],
            "dense_frozen": _final(h, "dense")["frozen_randneg"],
            "shuffled_final": _final(h, "dense")["shuffled"],
            "train_loss_final": _final(h, "train_loss") if isinstance(_final(h, "train_loss"), float) else h[-1]["train_loss"],
            "median_step_sec": d["median_step_sec"],
            "n_params": d["n_params"], "n_steps": d["n_steps"],
            "epochs": d["cfg"]["epochs"],
        })

    def ms(key):
        v = np.array([r[key] for r in rows], float)
        return float(v.mean()), float(v.std())

    step = float(np.mean([r["median_step_sec"] for r in rows]))
    epochs_run = rows[0]["epochs"]
    # full-set cost: step_time * full_n steps/epoch * E epochs, per seed, in CHF
    cost = {}
    for E in seeds_epochs:
        hrs = step * full_n * E / 3600.0
        cost[f"E={E}"] = {"gpu_hours_per_seed": round(hrs, 2),
                          "chf_per_seed": round(hrs * CHF_PER_H100_HR, 2),
                          "chf_3seed": round(3 * hrs * CHF_PER_H100_HR, 2)}
    return {
        "n_seeds": len(rows), "rows": rows,
        "mean_std": {k: ms(k) for k in ["sc_best", "sc_final", "sc_frozen", "dense_final",
                                        "dense_frozen", "shuffled_final", "median_step_sec"]},
        "h100_median_step_sec": step, "epochs_run": epochs_run,
        "subset_train_n": n_seed_train, "full_n": full_n,
        "full_set_cost_chf": cost,
        "cost_caveats": (
            "step_time measured at 90-complex subset scale (all held in GPU mem). Full 4,943-set "
            "cannot fit in 94 GB → needs disk/CPU streaming, which may raise per-step time. "
            "Epochs-to-converge unknown at full scale (subset still climbing at 130); range shown."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/upthomae/Meng/phase4")
    ap.add_argument("--full-n", type=int, default=4943)
    ap.add_argument("--out", default="logs/phase4/stageA_summary.json")
    args = ap.parse_args()
    s = summarize(args.dir, args.full_n)
    if "error" in s:
        print(s["error"]); return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(s, open(args.out, "w"), indent=2)
    print(f"=== Stage-A summary ({s['n_seeds']} seeds, {s['subset_train_n']} train) ===")
    for r in s["rows"]:
        print(f" seed {r['seed']}: SC best={r['sc_best']:.3f}@ep{r['sc_best_epoch']} final={r['sc_final']:.3f} "
              f"| dense final={r['dense_final']:.3f} (frozen {r['dense_frozen']:.3f}) "
              f"| shuf={r['shuffled_final']:.3f} | step={r['median_step_sec']*1000:.0f}ms loss={r['train_loss_final']:.2f}")
    m = s["mean_std"]
    print(f"\n MEAN±STD  SC best={m['sc_best'][0]:.3f}±{m['sc_best'][1]:.3f}  "
          f"SC final={m['sc_final'][0]:.3f}±{m['sc_final'][1]:.3f}  (SC frozen ceiling {m['sc_frozen'][0]:.3f})")
    print(f"           dense final={m['dense_final'][0]:.3f}±{m['dense_final'][1]:.3f}  "
          f"(dense frozen {m['dense_frozen'][0]:.3f})  shuffled={m['shuffled_final'][0]:.3f}")
    print(f"\n H100 median step {s['h100_median_step_sec']*1000:.0f} ms; full-set ({s['full_n']}) cost:")
    for E, c in s["full_set_cost_chf"].items():
        print(f"   {E}: {c['gpu_hours_per_seed']} GPU-h/seed → CHF {c['chf_per_seed']}/seed, CHF {c['chf_3seed']} for 3 seeds")
    print(f"\n caveats: {s['cost_caveats']}")
    print(f"\n wrote {args.out}")


if __name__ == "__main__":
    main()
