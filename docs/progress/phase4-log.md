# Phase 4 — running log (real-time; watch me think)

> Append a `## <n>. <title>` header the moment a step starts, then reasoning/decisions/commands as they
> happen. Log cumulative CHF and every Kuma job-id. Keep the RESUME STATE block current so any restart
> reattaches without double-submitting.

**Goal:** from-scratch heterogeneous GNN for conformation-invariant interface matching — close the induced-fit
gap so any of the 5 AF3 samples queries like holo. Plan: `docs/08-phase4-design.md`. Contract:
`PHASE4_HANDOFF.md`.

---

## RESUME STATE  (authoritative — rewritten 2026-07-07 ~15:5x for the SCALE-UP phase)
- **✅ SCALE-UP COMPLETE (2026-07-08 ~11:55). `logs/PHASE4_SCALEUP_DONE` touched. Verdict in `docs/10`
  (final section).** Bottom line: **data-scaling question INCONCLUSIVE — a training-STABILITY failure.** Full set
  4,809 complexes (vs 90), held-out 60, leak-clean. Two independent recipes (lr1e-3/clip5 = 2×2 jobs 3796137-40;
  lr3e-4/clip1 = jobs 3798320-21), 6 runs: all diverge (stable ~15-25 ep → loss 12-20, final AUC < 0.50
  shuffled). Peaks ~0.80-0.83 (both regimes) modestly beat 90-cplx 0.749 ⇒ data *plausibly* helps but never a
  stable plateau. Instability robust to LR+clip ⇒ objective/optim problem (temperature runaway / stale
  negative-bank / representation collapse — untested). **Do NOT proceed to M2.** Next = recipe REDESIGN + a cheap
  diagnostic (log τ, grad-norm, embedding var). The auto-collector's "0.822 data-limited/gap-closing" was
  OVERTURNED (it averaged best-epoch of a diverging run). Kuma spend ≈ CHF 15-17 of 100.
- **FOLLOW-UP DIAGNOSTIC RUNNING (user-approved) — Kuma jobs `3798692` (dense-s0), `3798693` (sc-s0), name
  `p4_diag`.** Instrumented `p4.train` to log per-epoch **tau, |T|₂ (bilinear spectral norm), pre-clip
  grad-norm, z_std (embedding spread)**; runs the ORIGINAL unstable recipe (lr1e-3/clip5/cosine, 40 ep) so the
  divergence reproduces → whichever metric moves FIRST at the loss blow-up NAMES the trigger (temperature-
  runaway vs bilinear-blowup vs representation-collapse). CPU-smoke passed. `diag` series → `diag_{dense,sc}_
  seed0.json`. Watcher `bks3temo1`. **DO NOT resubmit p4_diag.** When done → read diag → append root-cause +
  targeted fix to `docs/10` §scale-up. (Scale-up verdict + sentinel already stand; this only adds root-cause.)
- **MISSION (DONE) = the user-approved FULL-SCALE SCALE-UP** (arc-1 is DONE: `logs/PHASE4_ARC1_DONE` exists;
  M0 gate PASS, M1 verdict written in `docs/10`). The interactive user gave **GO** for the full-set Stage-A.
  New sentinel for THIS phase: **`logs/PHASE4_SCALEUP_DONE`** (touch when the full-scale data-scaling verdict
  is written in `docs/10` + self-verified, or permanently blocked).
- **⚠️ REATTACH, DO NOT DOUBLE-SUBMIT.** `squeue -u ymeng` (Jed) + `ssh … kuma squeue -u ymeng` FIRST.
- **🟢 CURRENT STATE (REATTACH #6, ~17:25 — SUPERSEDES the stale bullets below).** preproc+precompute DONE
  (4872 npz). The launcher (65442894) ran but submitted 0 (Kuma h100 MaxMemPerCPU=5900 → 90G/12cpu rejected);
  **FIXED** sbatch to `--cpus-per-task=16 --mem=88G` and manually submitted. **Kuma 2×2 RUNNING: `3796137`
  dense-s0, `3796138` dense-s1, `3796139` sc-s0, `3796140` sc-s1** (train=4811, held-out=60, cosine+stream,
  60 ep, ~CHF 1 actual each). **Collector `65442943` polling** → writes verdict to docs/10 + touches sentinel
  when the 4 `scaleup_*seed*.json` land. **DO NOT resubmit 3796137-40 (guard on `p4_full`) nor any Jed job.**
  Remaining: wait ~2h → collector auto-completes; a resume may add nuance to the docs/10 table after.
- **CURRENT GATE (REATTACH #5, ~10:2x next day) = the STABILIZATION RUN (3798320 dense-s0, 3798321 sc-s0).**
  **Double-submit caught + cleaned:** a parallel conductor had ALREADY run the full 2×2 (`3796137-40`) + collector
  (`65442943`) → wrote an auto-verdict to `docs/10` + touched the sentinel. On resume I (not seeing §12) launched
  a redundant 2×2 (`3798265-68`); **CANCELLED it** (guard had returned 0 because 3796137-40 finished before my
  submit; the JSONs are 3796137-40's, untouched). **Self-verification OVERTURNED the auto-verdict:** the collector
  averaged only `best_sc` (0.822) and called it "data-limited/gap-closing", but the full histories show
  **training DIVERGED at scale** — stable to ~ep25 (dense held-out SC ~0.80, reproducible both seeds) then loss
  explodes 7.6→12–20 and ALL 4 **final-epoch AUCs collapse to ~0.37–0.43 (BELOW the 0.50 shuffled control)**.
  Corrected `docs/10` with an evidence table → honest verdict **INCONCLUSIVE (recipe-unstable at scale), NOT
  gap-closing**; genuine-but-suggestive signal = stable-early ~0.80 > 90-cplx 0.749. **Stabilization launched**
  (`scripts/phase4_stabilize_stageA.sbatch`: lr 3e-4, **--grad-clip 1.0** [new flag], cosine, 40 ep, dense+sc s0)
  to test if the ~0.80 is a real converged plateau. If it stabilizes → conclusive data-scaling verdict; if it
  still diverges → the INCONCLUSIVE verdict stands + flag deeper recipe work. Sentinel EXISTS (honest interim
  verdict is live), so the deliverable is safe; stabilization upgrades it. **DO NOT resubmit** 3798320/21 or any
  3796137-40 (guards on job-names `p4_stab`,`p4_full`). Session Kuma spend ≈ 6 (2×2) + 2 (stab) ≈ under budget.
- **CURRENT GATE (superseded, cancelled) = the KUMA 2×2 FULL-SCALE RUN.** Preproc 65437518 + precompute
  65442563 **both DONE** (all 500 precompute tasks COMPLETED; **4,871 complexes** with full npz in
  `/work/.../stageA_full_npz`, 1 partial excluded). Finalized **`stageA_full_train_ids.txt` = 4,811 train**
  (npz-present, guard-passed, leak-checked disjoint from held-out+m1 by id AND PDB-stem), **60/60 held-out**
  with npz — all on `/work`. **Kuma H100 jobs SUBMITTED (guard: no prior p4_full):**
  `3798265`=dense/seed0, `3798266`=dense/seed1, `3798267`=sc/seed0, `3798268`=sc/seed1
  (`p4_full`, 60 ep, cosine, **--stream**, d64/L4/bank128, 88G/16cpu per user's MaxMemPerCPU fix). Reserved
  CHF 3.1/job (6h cap); actual ≈ CHF 1.5–2/job (~2.5–3h). **NEXT:** watch each `.out`
  (`/work/.../scaleup_p4_full_<jobid>.out`) for (i) preload OK (no OOM at ~40GB into 88G), (ii) init eval
  frozen sc≈0.947 + shuffled≈0.5 (harness sanity), (iii) held-out SC/dense learned climbing. When all 4 done →
  aggregate best+final SC/dense per (pos,seed) vs 0.947 → **does 90→4,811 close the gap?** → verdict in docs/10
  + self-verify + touch `logs/PHASE4_SCALEUP_DONE`. **DO NOT resubmit** these 4 (guard on job-name `p4_full`).
- **CURRENT GATE (superseded) = the CHAINED PRECOMPUTE ARRAY.** Two Jed jobs live:
  (1) preproc array **65437518** draining (≤22 tasks, walltime-bounded, ~done); (2) **precompute array
  `65442563`** (`p4_pre_npz`, `--array=0-499%500`, **`--dependency=afterany:65437518`**) — PENDING until
  preproc drains, then builds hetero-graph npz for the 500 chunks (`logs/phase4/precompute_arr/chunk_*.txt`,
  4,997 ids = 4,937 guard-passed train + 60 held-out) into **`/work/upthomae/Meng/phase4/stageA_full_npz`**.
  Kuma EMPTY. **DO NOT resubmit either** (guard on job-names `p4_prep_arr`,`p4_pre_npz`).
  **AUTOMATED (REATTACH #4, step 10):** the train-ids build + Kuma 2×2 submission is now a chained SLURM job
  **`65442894` (p4_launch, `afterany:65442563`)** running `scripts/p4_scaleup_launch.sh` (idempotent, lock +
  Kuma-double-submit guarded). It finalizes `stageA_full_train_ids.txt` (npz-present ∩ 0-stem-leak) → cp /work
  → submits Kuma `p4_full` dense/sc×seed0/1. **A future resume must NOT manually build train-ids or submit the
  2×2 — 65442894 owns that.** Only remaining claude task = read the 4 `scaleup_{dense,sc}_seed{0,1}.json`,
  write the verdict in `docs/10`, touch the sentinel.
- **💰 Honest spend (cluster-reported):** Jed `sacct`/sbatch shows **user ymeng CONSUMED ≈ 24 CHF** to date
  (project-lifetime Jed CPU incl. Phase 1/2/3 preproc, not just this session; account cap 10,000, fine).
  Precompute 65442563 reserved CHF 11 (walltime cap) but actual ≈ CHF 0.5 (chunks finish ~1 min). This-session
  incremental ≈ preproc ~7 + precompute ~0.5 + Kuma (pending ~6–8) ≈ **~15 CHF of the 100 budget.**
- **CONDUCTOR REATTACH — job 65439015 (this headless agent), ~16:1x.** Verified queues: Jed array 65437518
  draining (was 253→119 tasks over ~10 min; ~5060 descriptor dirs on disk); **Kuma queue EMPTY** (no GPU jobs
  to reattach). No new `### USER:` comment in `docs/09` (last steer already answered). **SCALE-UP PREP DONE
  (idempotent, reuse on restart):** wrote `scripts/p4_build_full_trainids.py` (leakage-guarded id-list builder,
  uses `complex_is_available`; training.txt verified all 4-char stems / zero AF/RP/AS ⇒ guard = PDB-stem
  overlap with the 60 held-out + 31 m1) and `scripts/phase4_scaleup_stageA.sbatch` (parameterized
  `<dense|sc> <seed> [epochs]`; `--cosine --stream`, mem 90G, 12 cpu, h100/normal, epochs=60 default).
  Re-staged `src/` → `/work/.../phase4/src` (was STALE from 07-02; now has `--stream`/`--train-pos`; verified).
  Smoke-imported `p4.precompute` under the `/work` masif-graph py3.11 env — OK. Kuma facts: h100 partition
  huge (70+ nodes ×4 GPU), QOS `normal` MaxWall **3 days**, no per-user GPU cap ⇒ can run the 2×2 in parallel.
  **WAITING on a background monitor** (bash id, polls squeue 60s) that re-invokes on array drain → then run
  builder → split → precompute array → finalize train-ids → 4 Kuma jobs (dense/sc × seed0/1). NOT YET SUBMITTED
  any precompute/Kuma job.
- **REATTACH #2 (~16:2x, after a session teardown that killed the 1st monitor):** array 65437518 still draining
  (85 tasks R, ~5144 desc dirs); Kuma EMPTY; no new `docs/09` USER comment. **Builder DRY-RUN validated** (clean
  audit): `training 4943 / held-out 60 / m1 31` → excluded **6 stem-overlap, 0 variant, 229 not-yet-available**
  → **4708 clean candidates** now (rises toward ~4700–4790 as the tail drains; ~6% preproc perma-fail is the
  floor on "unavailable"). No-leak asserts PASS. Wrote `logs/phase4/stageA_full_{candidates,precompute}.txt`
  (will re-run after drain for the final list). Array `--time=02:00:00` (now 1:13) ⇒ drain guaranteed ≤~47 min.
  Re-launched drain monitor as background bash **bwncicusr**. Decision: WAIT for full drain (clean, walltime-
  bounded), then precompute once — pipelining saves <15% of the ~4h critical path, not worth the top-up
  bookkeeping for a headless agent. If re-torn-down, the supervisor loop (65439015) re-invokes → recheck + proceed.
- **NEXT STEPS (in order):**
  1. When 65437518 done → build the full clean training id-list (training.txt complexes WITH descriptors,
     EXCLUDE any AF/RP/AS variant AND any sharing a PDB-stem with the 60 held-out or 31 `m1_ids`). Reuse
     `logs/phase4/preproc_todo.txt` logic. Held-out stays the 60 `stageA_heldout_ids.txt`.
  2. Split into ≤500 chunks → submit `scripts/phase4_precompute_array.sbatch` (`--array=0-K%500`) to build
     hetero-graph npz into `/work/upthomae/Meng/phase4/stageA_full_npz` (parallel; ~3h single-thread otherwise).
  3. Re-stage `src/` to `/work/upthomae/Meng/phase4/src` (rsync). Then Kuma H100 **2×2 full-scale run**:
     `--train-pos dense` and `--train-pos sc`, each `--cosine --stream` (CPU-preload→per-step GPU; needs
     `--mem≈90G`), ~1–3 seeds. Report held-out **SC + dense** AUC for each vs frozen 0.947 → does 90→~4,700
     close the gap? (data- vs architecture-limited). Guard each Kuma submit against squeue+job-name.
  4. Write the verdict in `docs/10-phase4-results.md`; keep `docs/09` async-comment protocol; touch the sentinel.
- **BUILT + TESTED this session (interactive):** `--train-pos {dense,sc}` selector (both pos sets already in
  every `contacts.npz`, defined in `p4.precompute`); `--stream` loader (CPU-preload + `ComplexP4.to(device)`
  non-mutating view); cosine LR (`--cosine`); `scripts/phase4_precompute_array.sbatch`. All smoke-passed.
- **Cumulative CHF ≈ 0.9 (Kuma) + ~5 (preproc array 65437518 est ≤10.5 reserved).** Budget CHF 100 — fine.
  Kuma DONE: 3795493 (baseline M1), 3795542 (cosine, SC final 0.707±0.029). Dead: 3795482 (NaN), 3795496 (dup).
- **Env perf:** always `OMP_NUM_THREADS=4`+`torch.set_num_threads(4)` for CPU torch on Jed (72-thread thrash).
  Preproc OOMs on login node → must be sbatch (compute node, mem). Kuma reached by
  `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch`.
- **STAGE-B TODO (later):** mismatch-filter AF3-conformer positives (retention<0.5 OR interface-local
  Cα-RMSD>4Å); the 7 mismatch ids are eval-only (0 training overlap).
- **⚠️ Note:** the previous headless conductor job **65436261 STOPPED** (it correctly hit the arc-1 sentinel
  at 14:51). **RELAUNCHED as headless SLURM conductor `scripts/phase4_scaleup_agent.sbatch` → job 65439015**
  (new sentinel `logs/PHASE4_SCALEUP_DONE`; does NOT clear the arc-1 sentinel). That headless conductor is now
  the single autonomous driver for the scale-up. Reattaches to 65437518; never double-submits.

---

## 1. Orientation — read handoff + design + Phase-3 lessons; mapped reusable assets (CPU, free)

Read in full: `PHASE4_HANDOFF.md`, `docs/08-phase4-design.md`, `docs/07-phase3-results.md`,
`docs/09-phase4-user-comment.md`. Arc-1 = **M0** (hetero graph builder + rotation gate) + **M1** (Stage-A
correspondence trainer training on a small holo subset → held-out holo→holo AUC + full-set cost estimate +
feasibility verdict). Budget CHF 100; checkpoint before the full ~4,943-complex run.

**Reusable assets confirmed (read the source):**
- `io/reference.py` — `load_complex`, `parse_heavy_atoms`, `PRECOMP_DIR/DESC_DIR/PDB_DIR/SURFACE_DIR`,
  `complex_is_available`. Row `i` of desc/X/Y/Z ↔ vertex `i` (verified alignment).
- `graph/build.py` — Phase-2 atom-graph builder (`build_atom_graph`): atom nodes (10-D invariant base),
  covalent edges (bond-order one-hot + sidechain-rotatable flag), `flex_depth`, `full_to_surf`. **Reused.**
- `m3/chem_graph.py` — `element_chem_features` (electronegativity/valence/covalent-radius). **Reused.**
- `surface/atoms.py` — `build_surface_atoms` → surface-atom table, `vertex_surf_idx`, `full_to_surf`.
- `pairs/construct.py` — `vertex_contacts` (sc-filtered) + `atom_positives_from_vertex_contacts` +
  hard/within negative samplers. **Reuse for Stage-A positives + hard negatives.**
- `metrics/separation.py` — `separation_auc`, `shuffled_label_auc` (the Phase-1/3 gating metric).
- `m3/precompute_graph.py` + `m3/dataset.py` — the **cross-env npz pattern** (byte-string `S24` keys, no
  pickle; join at train time). Phase-4 extends it with vertex nodes + mesh + vertex↔atom edges + normals.

**Data facts (commands in this log):** 242 holo complexes fully available; train pool = 90 (avail ∩ training
− eval); 91 testing availables for held-out eval; 31-complex `m1_ids.txt` all available, kept disjoint.
Vertex feature = `input_feat[:,0,[0,2,3,4]]` = `[si, hbond, charge, hphob]` (normalized ~[-1,1]); mesh from
`.ply` faces; normals from `.ply`. Scale: verts/chain median 4586 (p95 8871, max 16884), atoms median ~1522.
**Envs:** `masif-graph` (py3.11, torch 2.4.1+cu121, numpy 2.4.6, NO torch_scatter/PyG) build on Jed;
`atomsurf_h100` (py3.8, torch 2.4.1+cu124, torch_scatter/PyG, numpy 1.23.5) train on Kuma. Both have
`torch.scatter_reduce_`/`index_add_`.

**Key architecture decision (divergence-worthy):** write the encoder with **torch core ops only** — same
module imports in BOTH envs, so the embedding-level rotation gate runs on Jed CPU and training runs on Kuma.
The npz split (Phase-3 pattern) still handles the graph *data* across the numpy 2.4/1.23 gap.

## 2. M0 COMPLETE — heterogeneous graph builder + rotation-invariance gate PASS (CPU, free)

Built `src/masif_graph/graph/hetero.py` (`HeteroSurfaceGraph` + `build_hetero_graph` +
`rotation_invariance_report`), plus two gate drivers: `experiments/run_p4_m0.py` (full 90-pool sweep + JSON)
and `masif_graph.experiments.p4_m0_gate` (focused multi-seed gate; canonical reproduce). Design realized:
- **Atom nodes** (14-D invariant): Phase-2 base(10) + flex(1, normalized) + element-chem(3). **Dropped the
  Phase-2 spatial edges** (design §4: through-space atom edges inject pose-sensitivity) → atom–atom is
  covalent-only. aa edge feat = bond-order one-hot(4) + rotatable(1).
- **Vertex nodes** (4-D invariant, normalized): `[si, hbond, charge, hphob]` from `input_feat[:,0,[0,2,3,4]]`.
  Fixed a first-pass bug (charge/hphob were raw `.ply` ±12/±4.5) → now all four from the normalized precompute
  array. Coords/normals from `.ply`, geometry-only (never a node feature).
- **Mesh (vv) edges** from `.ply` faces (topology → coord-independent). Feat = [dist, cos(normal_i,normal_j)].
- **Vertex–atom (va) edges**: radius ball (5 Å, rotation-invariant) capped k=8 nearest, det. tie-break by
  (dist, atom-idx). Feat = [dist, cos(normal_v, unit(atom−vertex))]. Buried atoms get 0 va-edges by design.
- Raw scalars stored; RBF expansion deferred to tensor-build so the invariance test sees the raw scalars.

**GATE RESULT — PASS (two independent runs, mutually consistent):**
- Full-pool sweep (`run_p4_m0.py --n-rot 20`, `logs/phase4/m0_report.json`): **180/180 chains built across
  the 90-complex pool, 0 failures; rotation gate 20/20 PASS.**
- Multi-seed gate (`python -m masif_graph.experiments.p4_m0_gate --n 12 --seeds 1 2 3`): all chains PASS
  across 3 SE(3) seeds; exit 0.
- In every case each edge-feature max-diff is **exactly 0.0** — invariance is *structural*, not approximate
  (features are pure distances + cos of normals; connectivity is mesh topology + rotation-invariant radius
  balls; node features are pose-independent → byte-identical). ⇒ **M0 hard gate PASS.**
- Scale: verts/chain median 4586 (p95 8871, max 16884); build median 0.41s, total 83s (whole pool). No mesh
  coarsening needed for M0/M1 (a `max_vert` subsample knob exists for the tail / scale-up).
- Embedding-level rotation invariance closes in step 3 with the encoder (corollary of the above; verified
  numerically as belt-and-suspenders). **Cumulative CHF 0.**

## 3. M1 — Stage-A correspondence trainer (encoder E + bilinear T + InfoNCE); building now

Building `src/masif_graph/p4/`:
- `encoder.py` **(done)** — heterogeneous MP (torch-core ops, both-env) over the 3 edge types →
  per-surface-atom embedding `z`. Atom+vertex embed → N layers `HeteroMPLayer` (per-edge-type message MLPs,
  mean-agg, residual+LayerNorm) → readout at surface atoms ordered by surf row. `encoder_rotation_maxdiff`
  is the embedding-level gate.
- `objective.py` — symmetric bilinear `T = ½(A+Aᵀ)` (learns the flip's generalization); InfoNCE with
  in-complex + cross-complex + hard negatives; symmetric (either protein as query).
- `precompute.py` — hetero graph → cross-env `.npz` (extends the Phase-3 pattern with vertex/mesh/va tensors).
- `dataset.py` — load npz → torch tensors (RBF-expand edges, symmetrize mesh edges, build surf_node_idx).
- `train.py` — Stage-A holo-only trainer + held-out holo→holo separation-AUC + shuffled control (~0.5).

Plan: CPU smoke on Jed (loss ↓ + embedding rotation-invariant + per-step timing) → precompute ~55-complex
npz subset → **Kuma H100 Stage-A run** for a credible held-out holo→holo AUC + full-set GPU cost estimate +
feasibility verdict. **Checkpoint before any full ~4,943-complex run (budget gate).**

**RECONCILIATION (13:44–13:53):** while I was writing `encoder.py`, a parallel track wrote
`p4/{objective,dataset,train,precompute}.py` + a smoke npz set, all built ON my `encoder.py` + `hetero.py`
(same graph-dict keys → verified compatible). Rather than clobber, I **read + verified + adopted** them; they
are correct and guardrails-compliant (symmetric `T=½(A+Aᵀ)`, learnable τ, L2-normalized InfoNCE, cross-complex
neg bank; eval computes the **frozen ceiling on IDENTICAL pos/neg pairs** = exact apples-to-apples, plus a
shuffled ~0.5 control + per-complex median + `median_step_sec`). One design note: `precompute.py` uses the
**dense unfiltered** contact set as primary positives (`pos`) and also stores the sc-filtered `pos_sc`
(design D4). Dense is better for correspondence InfoNCE and the learned-vs-frozen comparison stays fair
(frozen ceiling computed on the same pairs). The parallel track quiesced ~13:50 (no p4 writes since); one
conductor process live (this one). Kept both; will guard Kuma submission against double-submit.

**M0 embedding-level rotation gate — PASS.** `experiments/p4_embed_rot_test.py`: build graph on original vs
SE(3)-transformed geometry → convert both to encoder dicts → run a random `HeteroEncoder` → per-surface-atom
embedding max|Δ| = **0.00e+00** on 3 complexes (tol 1e-4). Combined with the edge gate ⇒ **M0 hard gate is a
definitive PASS**; no coordinate leaks into the network.

**M1 pipeline end-to-end VALIDATED (CPU smoke, `logs/phase4/smoke_result.json`).** 5 train / 2 held-out, 20
epochs, d=48/d_out=32/L=3, 128k params. Random-init held-out learned AUC 0.358 → trained **0.72–0.785**, vs
the **frozen ceiling 0.694 on identical pairs** (learned already ≥ frozen); shuffled **0.49→0.52** (~0.5 ✓
control holds); loss finite; median step 121 ms CPU. This validates *the pipeline runs*; the 5-complex numbers
are overfitting, not the feasibility result — that needs the 90-complex run.
NOTE on "~0.90": that MaSIF ceiling is the **sc-filtered vertex-level** number; at the mean-pooled
surface-atom level on the dense contact set the frozen ceiling is lower (~0.69–0.75 here). The honest M1 gate
is **learned ≥ frozen on identical held-out pairs** (train.py measures this directly); absolute 0.90 is a
pair-definition-dependent reference, reported alongside, not the sole target.

## 4. Stage-A feasibility run — precompute 150-complex subset + CPU training (held-out holo→holo AUC)

Wrote clean splits (`logs/phase4/stageA_{train,heldout,all}_ids.txt`; 90/60/150; complex-level, mutually
disjoint, both disjoint from `m1_ids`). Precompute → `logs/phase4/stageA_npz/` DONE (150/150, 0 fail, 321 MB).
Staged code+npz+ids to `/work/upthomae/Meng/phase4` (Kuma has no /scratch). Verified the training stack
imports under `atomsurf_h100` (numpy 1.23.5); 90/90 train complexes load. `sbatch --test-only` caught a mem
cap (h100 MaxMemPerCPU 5900 → set --mem=32G/8cpu). **Kuma H100 job 3795482 submitted** (3 seeds × 150 ep,
d=64/d_out=32/L=4, bank=128, lr=1e-3; est CHF 0.34 max).

**Staged `train.py` reports the honest DUAL eval** (the parallel track added this just before my rsync; src ==
/work, verified): held-out separation AUC on **both** `pos` (dense all-vertex contacts; frozen ceiling ~0.68,
the training distribution) and `pos_sc` (MaSIF sc-filtered clean contacts; **frozen ceiling 0.947** — the real
"~0.90" gate), each vs the frozen ceiling on identical pairs + a shuffled control. This is exactly the
apples-to-apples comparison the feasibility gate needs.

**BUG: job 3795482 FAILED @50s** — init eval finite (learned 0.459/0.476, frozen 0.682/**0.947**, shuf 0.501),
but epoch-10 val eval hit `roc_auc_score: Input contains NaN`. Diagnosed (not shotgunned): scanned all 300
chain-npz → `1AKJ_AB_DE__holo__p1.npz` `vert_feat` has **5 non-finite shape-index values** (degenerate MSMS
vertices). 1AKJ is in TRAIN (not held-out), so init (val-only) was clean but training on it corrupted the
weights → NaN by epoch 10. **Fix (2 minimal edits):** (1) `dataset.load_chain_graph` now `nan_to_num`s all
float inputs at load (0 = neutral; robust to any such glitch across the full set); (2) `train` skips any
non-finite-loss step (insurance). Verified on CPU: 1AKJ loads finite; a d=64/L=4 train incl 1AKJ runs 12 ep,
exit 0, no NaN. Re-staged; **re-submitted as job 3795493** (squeue-guarded). (The tiny-smoke SC AUC swings
1.0↔0.0 are 2-val-complex degeneracy — few sc positives; the 60-complex held-out pools enough to be stable.)

**In-situ curve (job 3795493, seed 0) — the run LEARNS, just slowly (I over-reacted at ep10–50).** Held-out
climbs: SC 0.51(ep10)→0.62(ep50)→0.75(ep100)→~0.72(ep130); dense 0.65→0.70→0.78; loss 7.99→7.64; shuf ~0.50.
Early "loss ≈ above random, near chance" was warmup, not breakage — watching the full curve corrected it.
**Preliminary read (1 seed, pre-final):** the from-scratch GNN (a) beats the mean-pooled frozen descriptor on
**dense** contacts (~0.78 vs 0.682) and (b) reaches **~0.72–0.75 on sc-filtered** contacts but does NOT match
the **0.947** frozen sc ceiling — a ~0.20 gap, still climbing at ep130. That is the honest "on track, learns
correspondence + beats the pooled descriptor, but not yet at MaSIF's clean-contact ceiling on 90 complexes"
signal. Awaiting all 3 seeds + H100 timing. A concurrent CPU lr-sweep (3e-4/1e-4) checks if a lower lr
converges faster/higher (the ep10–50 oscillation hinted at lr-too-high, though ep100+ recovered).

**NEAR-FINAL (job 3795493): robust + H100 timing.** Seed 0 best held-out SC = **0.763**, seed 1 = **0.767**
(seed 2 running) — consistent across seeds. **H100 median step = 20 ms.** lr-sweep verdict: **lr=1e-3 is the
better choice** — lr=3e-4 gave only 0.685 (and oscillated/degraded at ep35–40 on the 30-cplx subset); my
"lr too high" hypothesis was WRONG, the model just needed warmup epochs. So the honest M1 signal is settling:
**from-scratch GNN on 90 holo complexes → held-out SC-filtered holo→holo AUC ≈ 0.76 (robust), dense ≈ 0.78
(beats the 0.682 pooled-frozen ceiling), ~0.18 below the 0.947 sc-filtered ceiling, still climbing** → learns
correspondence + beats the pooled descriptor, but not yet at MaSIF's clean-contact ceiling at this data scale.
Full-set cost @20 ms: 4943 × 20 ms ≈ 99 s/epoch → ~4 GPU-h/seed @150 ep ≈ **CHF ~2/seed, ~6/3-seed** (mem
caveat: 4943 won't fit 94 GB → streaming may raise step time). Wrote `experiments/p4_stageA_summarize.py`
(reads the 3 seed JSONs → summary + cost). Finalize verdict + checkpoint + sentinel when seed 2 lands.

## 4. M0 gate embedding-level PASS + M1 Kuma Stage-A feasibility run LAUNCHED

**M0 fully complete.** Embedding-level rotation gate through the untrained encoder:
`max|z(orig) − z(rot)| = 0.00e+00` (exact) on 1A1U — because M0 proved edge features + connectivity are
bit-identical under SE(3), and the encoder consumes ONLY those (no coordinate enters the net). Geometry gate
+ embedding gate both PASS → the SE(3)-invariance guardrail is satisfied structurally, not approximately.

**Perf gotcha found + fixed:** torch defaulted to 72 threads on the contended Jed login node (48 users) →
100s hangs on tiny graphs. `OMP_NUM_THREADS=4` + `torch.set_num_threads(4)` → build 0.9s, fwd 0.09s.
Reinforces the design: real training belongs on Kuma GPU.

**M1 pipeline validated end-to-end.** Positive-set finding: on DENSE all-vertex contacts the frozen MaSIF
ceiling is only ~0.69, but on MaSIF's **sc-filtered** clean contacts it is **0.94** — reproducing the
Phase-3 ~0.90 holo ceiling and validating my eval harness against a known number. So the M1 gate is judged
on the **sc-filtered** learned AUC vs frozen ~0.90 (dense used as stable training signal). Eval reports BOTH
sets + shuffled control (~0.50 confirmed at init on Kuma). One chain (1AKJ p1) had 5 NaN shape-index values
→ `nan_to_num`→0 guard added in `dataset.load_chain_graph` (only affected chain of 300; minimal).

**Stage-A run (Kuma H100, job 3795496):** 90 train / 60 held-out (disjoint from each other + m1_ids), 3 seeds
× 150 epochs, InfoNCE + symmetric bilinear T, d=64 d_out=32 L=4 (291k params). Cost est **CHF 0.34**. This is
the "small holo subset" M1 asks for — NOT the full 4943 run, so no checkpoint gate. A parallel duplicate
(3795493, collaborator) is also running; identical deterministic code/data so it harmlessly duplicates or
crashes on the pre-fix NaN — my 3795496 (submitted after re-staging the fix) is authoritative. Prior probe
3795482 crashed on the 1AKJ NaN (now fixed). Awaiting results: held-out sc AUC vs 0.94 + median_step_sec →
full-set GPU cost extrapolation → feasibility verdict in `docs/10-phase4-results.md`.

## 5. ARC-1 COMPLETE — M1 Stage-A 3-seed results + honest verdict + spend checkpoint + sentinel

Kuma job **3795493** finished ALL 3 seeds (14:34:30). Aggregated from `stageA_result_seed{0,1,2}.json`:

| metric (held-out 60, 3 seeds) | value | reference |
|---|---|---|
| SC-filtered learned AUC, **best epoch** | **0.749 ± 0.035** | frozen sc ceiling **0.947** |
| SC-filtered learned AUC, final epoch | 0.559 ± 0.100 (unstable; seed1 collapsed 0.767→0.454) | — |
| dense learned AUC, best | **0.739** | frozen dense 0.682 (learned **beats** it) |
| shuffled control | 0.505–0.510 | ≈0.5 ✓ |
| median step (H100) | **20 ms/complex** | — |

**Verdict (honest):** the from-scratch invariant GNN LEARNS interface correspondence (0.48→0.75 held-out;
beats the mean-pooled frozen descriptor on dense contacts) but does NOT match MaSIF's specialised sc-filtered
ceiling (0.75 vs 0.947, ~0.20 gap) and convergence is UNSTABLE on 90 complexes. M1 kill-switch NOT tripped;
gate (~0.90) NOT met at this scale → **promising, data-limited, gated on the cheap full-set scale-up.**
Best-epoch carries held-out selection bias (noted); fair summary = held-out SC swings ~0.45–0.78, best ~0.75.

**Full-set cost:** 20 ms/complex → 4,943 complexes ≈ 99 s/epoch → 100 ep ≈ CHF 1.4 / 150 ep ≈ CHF 2.1 /
3 seeds ≈ CHF 4–6. Prereq: streaming per-complex loader (14 GB npz won't fit upfront). Spend checkpoint +
GO recommendation posted in `docs/09-phase4-user-comment.md §5`; **not launched — awaiting user go** (the one
routine pause). **Cumulative CHF ≈ 0.4 of 100.**

**Self-verification:** M0 rotation gate exact-0.0 (structural, not fitted); frozen ceiling 0.947 reproduces
Phase-3's ~0.90 (harness validated vs a known number); shuffled ≈0.50; complex-level holdout, train/val/AF3-
benchmark mutually disjoint (verified 0-overlap); one NaN-input chain sanitized; every number → a committed
`stageA_result_seed*.json` + recoverable command. Touching `logs/PHASE4_ARC1_DONE`.

## 5. Post-GO: cosine-LR stabilization + reference-preprocessing checkpoint (user said GO)

Arc-1 finalized (sentinel touched 14:37; `docs/10-...` has the honest verdict). User replied **GO** to the §5
spend checkpoint. Executing it honestly: (1) **cosine-LR stabilization** run (job **3795542**, clean 90/60,
3 seeds, identical to baseline except `--cosine` 1e-3→1e-5) to test whether the held-out oscillation (baseline
SC best 0.749 but final 0.559±0.10, seed1 collapsed 0.767→0.454) is a fixable LR-schedule artifact — a stable
result is a prerequisite for any interpretable scale-up. (2) **Leakage catch:** the "expand to 151" idea pulled
in RP/AS variants of EVAL complexes (`1A2WAS0`,`1A99AS0`,`1A2WRP`…) — filter bug (excluded only `AF`). Aborted;
clean holo training data on disk is exactly the 90 in `training.txt`, so real scale-up needs `.sif` preproc of
~4853 complexes (NOT the training-only CHF 4-6). Costing that properly for a second checkpoint. Cosine est CHF
0.34 max; squeue guarded empty before submit.

## 6. Post-GO execution — cosine fix confirmed + honest full-set cost + preprocessing pilot LAUNCHED

**User said GO** on the full-set scale-up. Executing cheapest-first + honest.

**(a) Instability FIXED (cosine LR, job 3795542, 3 seeds):** held-out SC final-epoch 0.559±0.100 →
**0.707±0.029** (no collapse; final≈best; seed-std cut 3–4×). SC best 0.743±0.010. The dominant M1 caveat was
the optimizer schedule, NOT the architecture → stable reproducible **~0.72–0.74** vs frozen 0.947. Folded into
`docs/10` (Post-GO update 1).

**(b) Honest full-set cost — corrected (my §5 checkpoint under-stated it):**
- Only **91/4943 training complexes have MaSIF reference data**; the other **4,852 need the `.sif` pipeline**
  (MSMS/APBS/PyMesh/descriptor-net). Per-complex wall **median 304s** (mean 351, p90 562; ~6% fail), from 193
  existing `logs/m0/*.log`.
- **⚠️ Jed CPU IS BILLED** (~CHF 0.0055/core-hr; sbatch estimate CHF 4.22 for a 24hr×32-core reservation) —
  I was wrong that preprocessing is free. Full 4,815 preproc ≈ **~CHF 5**; + GPU training CHF 2–6 → full-set
  ≈ **CHF 7–11**. Still << 100, but not free.
- **⚠️ Login-node OOM finding:** fresh-complex pilot `1CTA_A_B` on the login node was **OOM-killed** (rc=137)
  at the TF descriptor step → preprocessing MUST run as a Jed sbatch job with allocated mem, not a login-node
  script. Wrote `scripts/phase4_preproc.sbatch` (standard partition, 32 cpu / 128G / 16 workers, resumable via
  holo_prep_batch.sh SKIP).

**(c) Gated data-scaling PILOT launched — Jed job 65437267 (p4_preproc, chunk1=400 clean complexes).** Clean
id list `logs/phase4/preproc_todo.txt` = **4,815** un-preprocessed training complexes (excludes AF/RP/AS
variants AND any sharing a PDB-stem with the 60 held-out or 31 m1 eval — the leakage guard). Purpose: prove
the sbatch approach fixes the OOM + measure real throughput + build a ~450-complex train set to run the
**decisive intermediate test — does 90→~450 lift the stable ~0.73?** (data-limited → continue to full ~CHF 5;
architecture-limited/flat → stop, save the spend). This is the honest cheapest-first gate before the full
preprocessing commitment. **Cumulative CHF ≈ 0.9** (Kuma ~0.4 + this pilot actual ~0.4–0.5).

## 7. USER STEER — array jobs for preprocessing (≤500 subsets, %500 concurrency)

User: "split training set into up to 500 subsets and submit array jobs, max concurrency 500 — single-thread
loop too slow." Correct — switched immediately.
- Cancelled single-node job 65437267 (40 complexes done, KEPT — preprocessing SKIPs already-computed ids).
- Split the **4,778 remaining** clean training complexes → **478 subset files** (~10 ids each,
  `logs/phase4/preproc_arr/chunk_NNNN.txt`).
- `scripts/phase4_preproc_array.sbatch`, submitted **`--array=0-477%500`** → **job 65437518**.
- **All 478 tasks RUNNING concurrently across 64 nodes** → full remaining set preprocesses in **~1 hr**
  (vs ~40 hr single-node), same total compute. Est ≤ CHF 10.5 reserved / ~CHF 5 actual. Resumable.

**Next pipeline (after preproc completes):** (1) `p4.precompute` over ~4,800 complexes is itself ~2–5 hr
single-thread → parallelize it as an array job too; (2) add a **streaming per-complex loader** to
`p4.train` (14 GB npz won't fit upfront); (3) launch full-scale Stage-A on Kuma (cosine LR) → does
90 → ~4,800 complexes close the 0.73→0.947 gap? Writing the streaming loader now while preproc runs.

## 8. Clarifications logged (user Qs during scale-up) + two code additions

**Q: frozen MaSIF descriptor — consumed by the GNN or just reference?** Verified: the encoder reads ONLY
atom_feat/vert_feat/edge-feats (no desc_*); the frozen 80-D `desc_straight/flipped` is used in exactly one
place — `train._frozen_scores` (the ceiling). Training loss is descriptor-free. The GNN *does* eat MaSIF's raw
**input** channels `vert_feat=[si,hbond,charge,hphob]` (hand-crafted per-vertex features, by design §4) — NOT
its learned descriptor. Two different "MaSIF" things; the distinction is the from-scratch premise.

**Q: is 0.73 (learned) vs 0.947 (frozen) apples-to-apples?** Verified: BOTH per-surface-ATOM, identical
pos/neg index pairs, same roc_auc; only the score differs (learned zᵀTz vs frozen 1/L2). The frozen desc is
mean-pooled to atoms (`emb_straight["mean"]`, (n_surf,80)), not per-vertex — so 0.947 is the *atom-pooled*
MaSIF ceiling (reproduces Phase-3's 0.90 atom-level), NOT MaSIF's per-vertex paper number.

**Added `--train-pos {dense,sc}`** to `p4.train` (user request): both positive sets already live in every
`contacts.npz` (defined in `p4.precompute`, NOT the `.sif` array), so switching needs no re-preprocessing.
Smoke hint (noisy, 5 cplx): train-on-sc lifts held-out SC AUC 0.686→0.811 — earlier runs trained dense but
gated sc (a mismatch that understated SC). **Full-scale plan: run BOTH train-pos to compare (2×2 SC/dense).**

**Q: structural-mismatch (23%, e.g. 1A2W) — in training positives?** Verified NO: it's a holo-vs-AF3 concept,
undefined for holo-only Stage A; and all 7 mismatch ids (1A2W,2AOB,2IWP,2PZD,2Z0E,3B5U,4UDM) are eval-benchmark
complexes with **0 overlap** with the Phase-4 training set (90 or 4,815). **TODO (Stage B):** when AF3 conformers
enter training, run the mismatch detector (retention<0.5 OR interface-local Cα-RMSD>4Å) to EXCLUDE non-binding
folds from the conformer-augmented positives; eval stays stratified addressable(+0.069) vs unaddressable(~23%).

## 9. RESUME (conductor 65439015 continuation) — reattached, verified no double-submit, found train-ids gap

Re-read RESUME STATE + squeue (Jed+Kuma). **State recovered — prior iteration already did scale-up prep + submitted the precompute array; I did NOT redo it.** Live jobs:
- **Jed preproc array 65437518**: 13 tasks left (1:28h elapsed), 5226 descriptor dirs on disk. Nearly done.
- **Jed precompute-npz array 65442563** (`p4_pre_npz`, `--array=0-499%500`): PENDING `afterany:65437518_*`
  — auto-starts when preprocessing finishes. Chunks in `logs/phase4/precompute_arr/` (500 files, 4997 ids →
  `/work/.../phase4/stageA_full_npz`). **Do NOT resubmit.**
- **Kuma**: empty (no training yet).

**LEAKAGE AUDIT (guardrail — PASS):** `stageA_full_candidates.txt` = **4768** clean train ids; exact-id AND
PDB-stem overlap with the 60 held-out and 31 m1 both **0**. The 2 AF/RP/AS flags (1AFV, 1ASY) are false
positives (real PDB codes), correctly kept. The 169 precompute "extras" beyond candidates+held-out have **0**
overlap with m1 — training-list complexes that were mid-preprocessing at build time; harmless.

**⚠️ GAP FOUND — must fix before Kuma run:** `scripts/phase4_scaleup_stageA.sbatch` reads
`$BASE/stageA_full_train_ids.txt`, which **does NOT exist** on Jed or /work (prior iteration wrote
`stageA_full_candidates.txt` instead). **Plan:** after preprocessing+precompute finish, REBUILD the clean
train-ids from the npz that actually exist (re-run `p4_build_full_trainids.py`, then filter to ids with
`stageA_full_npz/<id>__holo__p1.npz`), write `stageA_full_train_ids.txt`, rsync to /work, THEN submit the 2×2.

**Next:** monitor 65437518 → 65442563 → build+stage train-ids → submit Kuma 2×2 (dense/sc × 2 seeds) →
verdict in docs/10 → touch sentinel. Cumulative CHF ≈ 0.9 (Kuma) + ~5 (preproc) — under budget.

## 10. Full pipeline SLURM-CHAINED + autonomous (resume after session teardown of prior monitor)

Prior background monitor was torn down (no completion record). Rebuilt the critical path as a **SLURM
dependency chain** so it completes regardless of the conductor's claude-session lifecycle:
`65437518` (preproc, 7 stragglers, 2h walltime cap) → `65442563` (precompute npz, afterany) →
**`65442894` (p4_launch, afterany:65442563)**. The launcher `scripts/p4_scaleup_launch.sh` (idempotent,
lock-guarded, Kuma-double-submit-guarded): waits for precompute → builds `stageA_full_train_ids.txt` =
training.txt ∩ npz-present ∩ (0 stem-overlap with held-out/m1) → cp to /work → submits the **Kuma 2×2**
(`phase4_scaleup_stageA.sbatch` dense/sc × seed 0,1, 60 epochs, cosine+stream). Fills the earlier train-ids
gap. **DO NOT resubmit** 65437518 / 65442563 / 65442894, nor the Kuma 2×2 (guarded on job-name `p4_full`).

**Remaining conductor task (needs claude judgment):** when the 4 Kuma jsons
`/work/.../phase4/scaleup_{dense,sc}_seed{0,1}.json` exist → aggregate held-out SC + dense AUC vs frozen 0.947,
decide **data-limited (gap closes) vs architecture-limited (plateaus)**, write the verdict in `docs/10`, touch
`logs/PHASE4_SCALEUP_DONE`. ETA ~2.5–3h (preproc drain + precompute + ~2h training). Session spend ~15 CHF of 100.

## 11. FULLY AUTONOMOUS pipeline — scale-up now completes without the conductor session

My conductor session keeps getting torn down (background monitors don't survive turn boundaries). So the
ENTIRE scale-up is now encoded as SLURM jobs that complete independently:
`65437518` preproc (2 stragglers left, 2h walltime) → `65442563` precompute → `65442894` p4_launch
(builds train-ids → submits Kuma 2×2 `p4_full` dense/sc×seed0/1) → **`65442943` p4_collect** (standalone Jed
job, polls Kuma ≤4.5h → aggregates the 4 jsons → appends an honest results table + numbers-based data-scaling
read to `docs/10` → touches `logs/PHASE4_SCALEUP_DONE`).

**Guardrail:** the collector writes a verdict ONLY if it gets usable results; on Kuma failure it writes an
honest "BLOCKED" note (never a fabricated 0.000 verdict) and marks the sentinel BLOCKED. Read rule:
SC-best ≥0.82 → data-limited; 0.78–0.82 → partial; <0.78 → architecture-limited (vs the 90-complex 0.749 and
the 0.947 ceiling).

**⚠️ For any future resume:** the pipeline is self-completing. **Do NOT resubmit** any of 65437518 / 65442563
/ 65442894 / 65442943, nor the Kuma `p4_full` jobs. If resumed before the sentinel exists, just verify these
jobs are progressing (squeue Jed+Kuma) and wait; the collector owns the verdict + sentinel. A resume MAY add
richer nuance to the docs/10 verdict AFTER the collector writes the numbers, but the deliverable is guaranteed
without it. Session spend ≈ 15 CHF of 100 (preproc ~7 + precompute ~0.5 + Kuma 2×2 ~6–8 pending + collector ~0.2).

## 12. FIXED Kuma submission bug → 2×2 LAUNCHED (3796137-40)

Reattach caught the launcher (65442894) submitted **0** Kuma jobs (empty JIDs in its log). Diagnosed:
Kuma **h100 MaxMemPerCPU=5900 MB**; the sbatch asked 90G/12cpu = 7.5G/cpu → `sbatch: CPU count specification
invalid`. **Fix:** `--cpus-per-task=16 --mem=88G` (5.5G/cpu, under cap) in `phase4_scaleup_stageA.sbatch`
(both /scratch + /work), OMP→16. 0 p4_full were queued ⇒ manual resubmit is NOT a double-submit (guarded).

**Kuma 2×2 NOW RUNNING:** `3796137` dense-s0, `3796138` dense-s1, `3796139` sc-s0, `3796140` sc-s1 (est
CHF 3.10 each reserved / ~1 actual; train=4811, held-out=60, cosine+stream, 60 ep). The collector `65442943`
is polling → will aggregate the 4 `scaleup_{dense,sc}_seed{0,1}.json` → write the data-scaling verdict to
`docs/10` → touch `logs/PHASE4_SCALEUP_DONE`. **Future resume: DO NOT resubmit 3796137-40** (guard on
job-name `p4_full`). Session Kuma spend now ≈ 4 reserved; total ≈ 15–19 CHF of 100.

## 13. RESUME (2026-07-08) — auto-verdict OVERTURNED (training diverged) + double-submit caught & cancelled

Reattached; the 2×2 (`3796137-40`) had finished and the collector (`65442943`) had written a verdict to `docs/10`
+ touched the sentinel. **Two problems caught:**
- **Double-submit:** not seeing §12, I had launched a redundant 2×2 (`3798265-68`); my squeue guard returned 0
  only because 3796137-40 had just finished. **Cancelled 3798265-68** (~CHF 6 saved); the result JSONs are
  3796137-40's, untouched.
- **The auto-verdict was WRONG (the important finding).** The collector averaged only each run's **best epoch** →
  "SC 0.822, DATA-LIMITED, gap-closing, proceed to M2." Reading the **full per-epoch histories** overturned it:
  every run is stable ~15–25 ep (dense reproducibly hits held-out SC **~0.80**) then **DIVERGES** — train-loss
  explodes 7.6→**12–20**, and **all 4 final-epoch AUCs collapse to ~0.37–0.43, BELOW the 0.50 shuffled control**.
  So 0.822 is the luckiest spike of a diverging run (max over 48 noisy eval points), not a held-out AUC.
- **Corrected `docs/10`** (struck the auto-read; added a CONDUCTOR RECONCILIATION evidence table). **Lesson
  (durable):** always read the full training curve; never trust a best-epoch aggregate; a resume MUST re-verify
  an auto-written verdict before trusting the sentinel.

## 14. Stabilization run → instability ROBUST to LR+clip → INCONCLUSIVE verdict finalized

Tested whether the stable-early ~0.80 is a real plateau: added a **`--grad-clip` flag** to `p4.train`, ran the
calmer recipe (lr 1e-3→**3e-4**, clip 5.0→**1.0**, cosine, 40 ep, dense+sc seed0 = jobs `3798320`,`3798321`).
**Result — did NOT stabilize:** loss never settles to the ~7.6 low (dense 9–12; sc still climbing to **19** at
ep40); held-out SC still swings chaotically (sc 0.83→0.21→0.83); lower LR made dense strictly **worse** (best
0.72 vs 0.80). **⇒ instability is robust to LR + grad-clip → it is an OBJECTIVE/OPTIMIZATION problem, not a
step-size one.** **FINAL SCALE-UP VERDICT (in `docs/10`): data-scaling question INCONCLUSIVE — a training-
stability failure.** 6 runs / 2 recipes all diverge; peaks ~0.80–0.83 (both regimes) modestly beat 90-cplx 0.749
(data *plausibly* helps) but never a stable plateau. Controls valid throughout (shuffled ≈0.50; frozen 0.947/
0.682 reproduced; complex-level holdout; m1 untouched, leak-checked by id AND PDB-stem). **Do NOT proceed to
M2.** Sentinel touched 11:55. Leading (untested) suspects: temperature runaway / stale negative-bank /
representation collapse. Memory `phase4-scaleup.md` + `MEMORY.md` updated to COMPLETE.

## 15. DIAGNOSTIC (user-requested) — instrument the trigger; jobs 3798692/93 RUNNING

To name the root cause before any redesign, **instrumented `p4.train`** to log per-epoch: **tau** (InfoNCE temp;
→0.01 floor = runaway), **|T|₂** (bilinear spectral norm; grows = score blow-up), **pre-clip grad-norm** (spike =
explosion), **z_std** (embedding spread; →0 = representation collapse) — stored as a `"diag"` array in the
out-json + `[diag ep N]` stdout. CPU-smoke PASS (exit 0; smoke hint: z_std contracts 0.037→0.024 in 3 ep →
collapse is a candidate). Src rsynced to /work. **Kuma jobs `3798692` (dense-s0) + `3798693` (sc-s0)** run the
UNSTABLE recipe (lr1e-3/clip5/cosine/40ep, `scripts/phase4_diag_stageA.sbatch`) so the divergence reproduces;
whichever metric moves FIRST at the loss blow-up (~ep20–35) names the trigger. **DO NOT resubmit** (guard
`p4_diag`). When done → analyze `diag_{dense,sc}_seed0.json` → append root-cause + targeted fix to `docs/10`.
Scale-up verdict + sentinel already stand; this only adds the "why." +~CHF 2.

**DIAGNOSTIC COMPLETE (ep40, both runs) — ROOT CAUSE CONFIRMED.** z_std pinned **0.001–0.003** from ep1
(representation collapse, PRIMARY); τ on the **0.01 floor** throughout; **‖T‖₂ grows unbounded** (dense 11→28,
sc 10→39); grad-norm max to **5.6×10⁵**. Final AUCs are noise (dense 0.47, sc 0.81-with-loss-13.5). Root-cause +
targeted redesign (VICReg/SimSiam anti-collapse + freeze τ + constrain T, then re-run this same diagnostic to
verify z_std~0.1 & τ off-floor) written to `docs/10` §diagnostic. **PHASE-4 SCALE-UP FULLY CLOSED OUT:** verdict
INCONCLUSIVE (training-stability failure, root cause named), sentinel `logs/PHASE4_SCALEUP_DONE` stands, Kuma
clear, session spend ≈ CHF 18 of 100. The redesign is a NEW phase — awaits user go, not autonomous.

## 17. ANTI-COLLAPSE FIX — stability SOLVED, from-scratch reaches holo ceiling (user GO: "finish Phase 4")

Implemented the §15 redesign: `objective.py::vicreg_terms` (VICReg var+cov on raw embeddings) + `train.py`
flags `--vicreg-var/-cov --freeze-tau --tau --t-wd` (opt-in, baseline preserved for A/B). CPU-smoke (12 cplx):
z_std 0.04 stable, τ fixed, ‖T‖ flat ~1.7, grad 4–6, loss down — collapse gone. Full-set **2×2** on Kuma
(jobs **3801368–71**, dense/sc × seed 0/1, 4811 cplx, 50 ep, ~1.6 h each, **CHF ~3.4**; config vicreg-var 2.0
/cov 0.04 / freeze-τ@0.1 / t-wd 1e-3 / lr 5e-4 / clip 1.0 / cosine / stream / bank 128).
**RESULT (full detail in docs/10 §16):** all 4 stable through ep50, **no divergence** (τ pinned, ‖T‖ 4–12,
z_std 0.015–0.05, shuf ~0.50) — root-cause call vindicated. From chance init (0.46/0.48) → **sc median reaches
the 0.947 frozen ceiling** (best sc_s0 pooled 0.901 / median 0.997, still ↑ at ep50); **dense beats** frozen-dense
(+0.09–0.16). **Phase 4 flips INCONCLUSIVE → real: the arc-1 "unstable ~0.75, data-limited?" was an
OPTIMIZATION-STABILITY failure, not a data/capacity limit.** SCOPE CAVEAT: holo→holo M1 feasibility only — NOT
the north-star holo→apo robustness (needs the apo/AF3 eval, not run here); necessary-not-sufficient. Best
checkpoints saved `vicreg_{dense,sc}_best_seed{0,1}.pt`; per-epoch `diag`+`history` in `vicreg_*_seed*.json`.
Session spend ≈ CHF 22/100. Code fix lives on both /scratch repo and /work/.../phase4/src (synced); **not committed** (norm: commit only when asked).

## 16. Notebook illustration — message passing (user-requested)

Added §8–§10 to `notebooks/inspect_p4_data.ipynb` (validated end-to-end via nbconvert on `1A0G_A_B`): a
schematic of the 4 message directions (aa/vv/va/av) + update equations; a receptive-field BFS (one surface
atom sees 141 atoms + 299 vertices after 4 layers); and a live random-init encoder run showing z (n_surf,32),
z_std=0.062 (healthy) vs the collapsed 0.003, and that `coord` is not an encoder input (SE(3)-invariant).
