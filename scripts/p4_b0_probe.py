"""B.0 zero-training probe driver: load the 30 AF3-eval complexes ONCE, then run the AF3->holo
eval for each holo-only Stage-A checkpoint + a random-init control. Writes one combined JSON.

Answers: does the from-scratch encoder, trained holo-only, already degrade LESS than frozen MaSIF
when the query is an AF3 model — before ANY invariance training?
"""
import json
import os
import sys

import numpy as np
import torch

from masif_graph.p4.eval_af3 import Rec, build_encoder, encode_all, evaluate

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

DATA = "logs/phase4/m2_npz"
IDS = "logs/phase4/m2_eval_ids.txt"
W = "/work/upthomae/Meng/phase4"
POS = sys.argv[1] if len(sys.argv) > 1 else "pos_sc"
SEEDS = 3
OUT = f"logs/phase4/m2_b0/b0_combined_{POS}.json"
CKPTS = [("random_init", None)] + [
    (c, f"{W}/vicreg_{c}.pt") for c in
    ("sc_best_seed0", "sc_best_seed1", "dense_best_seed0", "dense_best_seed1")]

ids = [l.strip() for l in open(IDS) if l.strip() and not l.startswith("#")]
recs = []
for cid in ids:
    r = Rec(DATA, cid, POS)
    if r.ok and len(r.inter) >= 8:
        recs.append(r)
print(f"usable records: {len(recs)} (pos_key={POS})", flush=True)
rr = np.array([r.retention for r in recs if r.has_af3])
print(f"retention mean={rr.mean():.2f} median={np.median(rr):.2f} min={rr.min():.2f} "
      f"n<0.5={(rr < 0.5).sum()}", flush=True)

results = {}
for name, path in CKPTS:
    enc, comp, src = build_encoder(recs, path, "cpu")
    enc.eval()
    emb = encode_all(enc, recs, "cpu")
    acc = {"hh": [], "af3_holo": []}
    for s in range(SEEDS):
        for rg in ("hh", "af3_holo"):
            acc[rg].append(evaluate(comp, recs, emb, "cpu", rg, seed=1000 + s))

    def mn(vals, k):
        v = np.array([x[k] for x in vals if x[k] is not None], float)
        return float(v.mean()) if len(v) else float("nan")

    hh_l = mn(acc["hh"], "learned_randneg"); hh_f = mn(acc["hh"], "frozen_randneg")
    ah_l = mn(acc["af3_holo"], "learned_randneg"); ah_f = mn(acc["af3_holo"], "frozen_randneg")
    hh_med = mn(acc["hh"], "learned_percplx_median"); ah_med = mn(acc["af3_holo"], "learned_percplx_median")
    shuf = mn(acc["af3_holo"], "shuffled")
    row = {"learned_hh": hh_l, "frozen_hh": hh_f, "learned_af3_holo": ah_l, "frozen_af3_holo": ah_f,
           "learned_hh_median": hh_med, "learned_af3_holo_median": ah_med, "shuffled": shuf,
           "delta_robustness_af3": ah_l - ah_f, "do_no_harm_hh": hh_l - hh_f,
           "learned_gap": hh_l - ah_l, "frozen_gap": hh_f - ah_f}
    results[name] = row
    print(f"[{name:16s}] hh L={hh_l:.3f} F={hh_f:.3f} | af3 L={ah_l:.3f} F={ah_f:.3f} "
          f"| Δrobust={row['delta_robustness_af3']:+.3f} dnh={row['do_no_harm_hh']:+.3f} "
          f"| Lgap={row['learned_gap']:+.3f} Fgap={row['frozen_gap']:+.3f} shuf={shuf:.2f}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"pos_key": POS, "n_complexes": len(recs), "seeds": SEEDS,
           "retention_mean": float(rr.mean()), "results": results}, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
open("logs/phase4/m2_b0/B0_DONE", "w").write("done\n")
