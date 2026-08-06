# Phase 5 — HANDOFF (self-contained brief for an unattended / resumed agent)

## Autonomy contract
You are building **all of Phase 5** autonomously. No human in the loop. Decide, document, and keep
going; do not stop to ask. A negative result, honestly verified, is a valid finish. Only pause to
post a spend checkpoint if a single action would blow the budget. **Persistence is the rule:** never
stop because you finished a sub-step — pick up the next one. Continue until the gate is met or every
option inside the boundaries is exhausted.

## Goal / definition of done
Decide the **Phase-5 gate** (`docs/13-phase5-design.md` §3): on a **sequence-cluster-clean** held-out
test set (+ self-generated AF3 apo + clean decoys), does the learned invariant encoder retrieve the
true binder **≥ frozen MaSIF on holo→holo** AND with **smaller holo→AF3 degradation** than frozen
across the AF3 cells (AH/HA/**AA**=headline)? Deliverable = `docs/15-phase5-results.md` with the
4-cell matrix, robustness Δ, per-complex spread, controls, graph-ablation, and a verdict + Phase-6
recommendation. Touch sentinel `logs/phase5/PHASE5_DONE` only when self-verified.

## Boundaries / locked decisions
- **Budget: CHF 100 compute** for the whole phase — spend it, log cumulative cost, right-size, no GPU
  launch that would exceed it. (Interactive session; the CHF-100 is the working budget for this task.)
- apo = **AI-predicted (AF3)**, NOT experimental unbound. Validation on the **PDB list**
  (`masif-neosurf-af2/masif/data/masif_ppi_search/lists/{training,testing}.txt` = 4943/959), NOT the
  TED domainome. Ligands/neosurfaces + TED-domainome inference = **Phase 6**. Stage-C FP/precision
  funnel = later phase. DB5.x / experimental apo = never.
- Learned encoder default = `/work/upthomae/Meng/phase4/ret_full_ctr_best.pt`, evaluated **with
  DC-offset centering** (`--center`); retrain (conformer-aug) only if AF3 cells underperform (D-P5.1).

## Compute model = CONDUCTOR
Jed CPU work + preprocessing here (login node has internet; heavy CPU via `sbatch --qos=serial`).
GPU (AF3 inference, any training) on **Kuma** via ssh (cannot sbatch Jed→Kuma):
`ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'sbatch -A upthomae -p h100 -q normal --gres=gpu:1 ...'`.
Stage shared data via `/work/upthomae/Meng`. `sbatch --test-only` for cost before every child job.

## Read-first pointers
- `docs/progress/phase5-log.md` — **RESUME STATE block + running log (your memory)**. Read first.
- `docs/13-phase5-design.md` — the plan, gate, D-decisions, milestones, risks.
- `docs/14-phase5-user-comment.md` — steering; reply inline, don't stop.
- `docs/10-phase4-results.md` §22–24 — DC-offset centering (mandatory) + the retrieval harness lineage.

## Reusable assets (see log §0 for full detail + schema)
- `.sif` surface gen: `/scratch/ymeng/Neosurf_Neosurf/masif-neosurf_v0.1.sif` via
  `scripts/phase4_preproc.sbatch` + `scripts/holo_prep_batch.sh`; then `p4.precompute` → npz.
- AF3: `alphafold3.sif` (`/work/upthomae/Meng/AlphaPulldown/container/`), weights `AF3_weights/af3.bin.zst`,
  `scripts/af3_*`, envs `protenix`/`chai`.
- Holo npz store `/work/upthomae/Meng/phase4/stageA_full_npz` (train 4812/4943; test 60/959 → gen the rest).
- p4 code `src/masif_graph/p4/*`; env python `/work/upthomae/Meng/conda_envs/masif-graph/bin/python`.

## Guardrails (mandatory; skill dir is empty so they live here)
Shuffled-label ≈0.5 control every eval; frozen MaSIF on **identical** pos/neg pairs = ceiling;
per-complex spread + median rank, never a single top-k point; sequence-cluster holdout (no homolog
leak); assert `z_std` sanity + `--center` on the learned encoder; state "pipeline ran" separately
from "result valid"; try to break your own good news (more seeds / shuffled / held-out) before believing it.

## Milestones & sentinel
M0 (split+harness+HH sanity+AF3 cost) → M1 (AF3 at scale) → M2 (the gate) → M3 (graph ablation+writeup).
Log every child job-id + cumulative CHF in the progress log's RESUME STATE. Sentinel on done:
`touch logs/phase5/PHASE5_DONE`.
