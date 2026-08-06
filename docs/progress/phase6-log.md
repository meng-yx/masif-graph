# Phase 6 — running log

## Workstream A — encoder I/O walkthrough — DONE
`notebooks/encoder_io_walkthrough.ipynb` (committed ab9b665). Contract: encoder consumes a per-chain
HeteroSurfaceGraph, emits a per-surface-atom 32-D embedding field (NOT a per-atom score); binding score is
downstream pairwise `median_i max_j zᵢᵀ T zⱼ`. Example 1G60: 1944 atoms → 1141 surface atoms → (1141,32);
true partner retrieval rank 1/89.

## Workstream B — data-scaling ablation (is 4,872 PPI complexes enough?)
**Design (rigorous, avoids the fixed-init confound):** retrain the FULL pipeline (VICReg Stage-A →
retrieval fine-tune, both on the SAME subset) per training-set size; evaluate each on the Phase-5 gate
metric (dense-`pos` AA retrieval + holo→AA robustness) on the **leak-free 287-clean** eval set
(`eval_sc304_clean_vs_enc.txt`), which is clean vs ALL 4811 train → clean vs every subset (no new leakage).
- **Sizes:** 600, 1500, 3000 (nested prefixes: larger ⊇ smaller, lowers curve variance) + **4811≈full
  reused** from Phase-4/5 (`ret_full_ctr_best.pt`, same recipe: sc-VICReg init + centered 60-ep retrieval).
- **Seeds:** 0 and 1 (different shuffles → different subsets → bounds subset-draw noise).
- **Recipe fixed** = the Phase-4 anti-collapse recipe (vicreg-var 2.0/cov 0.04, freeze-τ@0.1, T-wd 1e-3,
  lr 5e-4, cosine, d64/dout32/L4). Same for all points → only data size varies.
- Pipeline sbatch: `/work/upthomae/Meng/phase6/p6_scaling_pipeline.sbatch <size> <seed>`; subsets under
  `/work/upthomae/Meng/phase6/subsets/`. Cost ~CHF 0.5–1.6/pipeline (est cap 3.10 for the 6h walltime).
- **Verdict rule:** curve still rising at 4811 → more PPI data helps (fund Workstream B(b): DIPS ~42k etc.);
  flat → saturated (Workstream C invests in the ligand axis, not scale).

**Guardrails (ml-research-guardrails):** consistent eval set + harness across sizes; ≥2 seeds w/ spread;
frozen-MaSIF ceiling unchanged (data-independent); shuffled control already validated in Phase 5; report
per-complex spread not just the pooled top-5; don't over-read a noisy 4-point curve.

### RESUME STATE
- Smoke: Kuma pipeline 600/s0 running (validates both stages + times it) before the sweep. Monitor bkqnh5jmu.
- After smoke OK → submit remaining 5 pipelines (600/s1, 1500/s{0,1}, 3000/s{0,1}) → gate-eval each → plot.

### B3 sweep RUNNING (2026-08-06)
Smoke 600/s0 PASSED: VICReg ~11min + retrieval ~37min = ~48min; z_std 0.176 (no collapse); train_top1 0.46.
On the leaky m2 val, 600-model af3 top5 ~0.31 vs full ~0.57 -> early hint the curve is RISING.
- Pipelines: 600/s0 DONE; **600/s1=4002511, 1500/s0=4002512, 1500/s1=4002513, 3000/s0=4002514, 3000/s1=4002515** (Kuma). Longest ~4h.
- Encoders saved `/work/upthomae/Meng/phase6/ret_{size}_s{seed}_best.pt`; DONE_{tag} markers.
- **Monitor b2ufs5lqe** -> when all 6 DONE, submits `scripts/p6_scale_eval.sbatch` (dense-pos gate on
  287-clean for each) -> `logs/phase6/gate_scale_{tag}_pos.json`.
- **B4 next:** collect dense-AA top5 + holo→AA robustness per (size,seed) + the full point
  (`gate_fullclean_pos.json`, x=4811) -> plot metric vs training-set size (2 seeds, spread). Verdict.

### B4 EVAL running
All 6 pipelines DONE (encoders present). Gate eval **job 65975990** (Jed, 6 dense-pos gates on 287-clean,
sequential ~1.5-2h) -> `logs/phase6/gate_scale_{tag}_pos.json`. Monitor **b4hpqip9s** -> runs
`scripts/p6_scale_plot.py` when 6 jsons present -> `notebooks/figs/fig_scaling.png` + verdict
(AA top5 & HH top5 vs size, 2 seeds; full point x=4811 from gate_fullclean). Verdict rule: slope 3000->4811
> +0.02 = climbing (fund larger corpus / DIPS); else saturated (invest in ligands, Workstream C).
