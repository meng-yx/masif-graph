# Phase 8 Stage A — build log (append-only)

Plan: `docs/24-phase8A-plan.md`. Contract: `docs/23-phase8-design.md`.
Ends at the **PAUSE** for the D8-12 decision (apo-prediction method + holo:apo ratio) — Stage A
produces a *recommendation*, never the choice.

Spend is tracked per step. Jed CPU ≈ CHF 0.005/core-h; Kuma H100 ≈ CHF 0.52/GPU-h.

---

## 2026-08-12 — step 0: tooling verified, scaffold, A1 launched

**Tooling closed out** (was the open item from the plan review):
* **TMalign INSTALLED** — compiled from source (`zhanggroup.org/TM-align/TMalign.cpp`,
  v20220412) into `/work/upthomae/Meng/conda_envs/masif-graph/bin/TMalign`. Compiling beat a conda
  solve: no dependency churn in a working env. **Verified functionally**, not just "it exists":
  self-alignment TM = 1.00000, an unrelated pair 0.155 / 0.483. `tmtools` is no longer needed.
* **Chai found** — `/home/ymeng/miniconda3/envs/chai`, `chai_lab` 0.6.1 on torch 2.6.0+cu124.
  My earlier "not importable" was a bad search path (I looked only under `/work/.../conda_envs/`).
* **Shared-MSA path verified** — our AF3 `<chain>_data.json` carries the alignments inline as a3m
  (`1bq4_A`: `unpairedMsa` 16,705 seqs / 5.9 MB, `pairedMsa` 50,000 / 19.8 MB, 4 templates), and
  `chai_lab...aligned_pqt.a3m_to_aligned_dataframe` converts a3m → chai `.aligned.pqt`.
  `run_inference(msa_directory=)` consumes it. So A0 compares **inference on one identical MSA**
  instead of accidentally benchmarking five different MSA searches. `docs/24` §2.2 updated.

**Scaffold**: `src/masif_graph/p8/`, `logs/phase8A/{a0..a4}`, this log.

### A1 — design deviation from `docs/24` §3.1, recorded before running

The plan proposed rebuilding eval graphs with **backbone-only atom nodes**. That is unsound as
written: the encoder emits one row per *surface atom* (`z = readout(ha[surf_node_idx])`) and ~60% of
surface atoms **are** sidechain atoms (measured: 0.598 on `1A99_C_D` p1). Deleting them renumbers
`surf_node_idx`, invalidates `Rec.inter`, and leaves the ablated arm retrieving over a smaller,
different patch — so any drop would be confounded with patch size.

Replaced with a **row-preserving ablation ladder** (`p8.ablate`) that cuts edges and destroys
features but never deletes a node, so every arm scores the identical rows:

| ablation | what it destroys |
|---|---|
| `none` | nothing — must reproduce the published number (reproduction check) |
| `sc_feat` | sidechain atom **chemistry** (feature rows permuted among sidechain atoms) |
| `sc_edge` | sidechain **connectivity** (aa edges incident to a sidechain atom cut) |
| `sc_all` | both, plus va edges into sidechain atoms — sidechain atoms fully isolated |
| `bb_feat` | backbone atom features (comparison channel) |
| `vert_feat` | the surface chemistry channel |
| `all_feat` | **positive control** — every node feature in the graph. MUST collapse. |

`all_feat` is the guardrail: if retrieval survives destroying every node feature, the harness is not
measuring what it claims and no sidechain-blindness conclusion may be drawn from the other rows.
Pre-registered before seeing any result.

Verified each ablation actually bites (`1A99_C_D` p1, 2680 atoms / 1193 surface rows):
`sc_feat` 1265 atom rows changed; `sc_edge` aa 5496→2726; `sc_all` aa 5496→2726 **and** va
42256→18810; `vert_feat` 5433 vertex rows; `all_feat` 2511 + 5433. Row count preserved in all
(z stays (1193, 32)); the source graph is never mutated.

Also measured, alongside retrieval: **per-atom embedding displacement** split by whether the surface
row is a sidechain or backbone atom, plus Spearman(displacement, `flex_depth`) under `sc_all` —
the closest thing the current encoder has to an implicit per-atom σ (D8-9 / D8-19).

**Two seeds** (D8-11): `phase6C/ret_ppionly_best.pt` and `phase7/ret_ppionly_s1_best.pt`, whose
published HH top-5 are 0.651 / 0.638 (Phase-7 §2 reports 0.644 ± 0.007 — matches).

**Launched**: `sbatch --array=0-1 scripts/p8a_a1.sbatch` → **job 66063328**, est. CHF 1.41 (that is
the `--time=8h` worst case; the Phase-7 gate ran axis 1 in minutes, so actual will be far less).
Expected signal: `logs/phase8A/a1/{disp_*.json, ret_*_<ablation>.json}`.

**Code change**: `p5.retrieval_bench.run()` gained an optional `transform=` hook so the ablated arms
run through the *identical* benchmark rather than a reimplementation. Non-breaking (default `None`).

Spend so far: ~CHF 0 (login-node work) + job 66063328 pending.
