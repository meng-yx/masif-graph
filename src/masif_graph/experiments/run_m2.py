"""Run the Milestone-2 global-alignment prototype over a set of complexes and summarize.

Usage: python -m masif_graph.experiments.run_m2 --ids logs/m1/complexes_used.txt --out logs/m2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from masif_graph.align.global_align import align_one


def run(ids, out_dir, seed=0, op="mean", n_seeds=1):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cid in ids:
        for s in range(n_seeds):
            try:
                r = align_one(cid, op=op, seed=seed + s)
                d = r.__dict__.copy()
                d["seed"] = seed + s
                rows.append(d)
                print(f"{cid} seed{seed+s}: corres={r.n_corres} prec={r.corres_precision:.2f} "
                      f"iRMSD {r.irmsd_start:.0f}->{r.irmsd_ransac:.2f} contact_rec={r.contact_recovery_icp:.2f} ok={r.success}")
            except Exception as e:  # noqa: BLE001
                rows.append({"complex_id": cid, "seed": seed + s, "success": False, "error": str(e)})
                print(f"{cid} seed{seed+s}: ERROR {e}")

    ok = [r for r in rows if r.get("success") and np.isfinite(r.get("irmsd_ransac", np.nan))]
    summary = {
        "n_attempted": len(rows),
        "n_success": len(ok),
        "irmsd_ransac_values": [round(r["irmsd_ransac"], 3) for r in ok],
        "contact_recovery_values": [round(r["contact_recovery_icp"], 3) for r in ok],
    }
    if ok:
        ir = np.array([r["irmsd_ransac"] for r in ok])
        cr = np.array([r["contact_recovery_icp"] for r in ok])
        summary.update({
            "irmsd_median": float(np.median(ir)), "irmsd_mean": float(ir.mean()),
            "n_under_5A": int((ir < 5).sum()), "n_under_10A": int((ir < 10).sum()),
            "contact_recovery_median": float(np.median(cr)), "contact_recovery_mean": float(cr.mean()),
        })
    out = {"summary": summary, "rows": rows}
    json.dump(out, open(os.path.join(out_dir, "m2_results.json"), "w"), indent=2)
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="file with one complex id per line, or a JSON list")
    ap.add_argument("--out", default="logs/m2")
    ap.add_argument("--n_complexes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.ids.endswith(".json"):
        ids = json.load(open(args.ids))
    else:
        ids = [l.strip() for l in open(args.ids) if l.strip()]
    ids = ids[: args.n_complexes]
    run(ids, args.out, seed=args.seed)
