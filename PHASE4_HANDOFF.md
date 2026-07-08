# PHASE 4 HANDOFF — Autonomous conductor: a from-scratch GNN for conformation-invariant interface matching

You are a **high-autonomy SLURM conductor agent**. You own a standing goal, not a narrow task. You
**decide, build, test, and iterate on your own**; you do **not** ask the user to approve each step. The
user steers asynchronously via a comment file (§3). Operate under `ml-research-guardrails` and
`slurm-claude-agent` at all times.

**The full technical plan already exists — read `docs/08-phase4-design.md` in FULL first.** This handoff is
the autonomy/compute/operations contract; the design doc is the science. Do not duplicate; follow it.

---

## 0. The standing goal (north star)

Phases 1–3 kept the pretrained MaSIF 80-D descriptor **frozen** and layered structure on top. That line is
**exhausted** (see §2). Phase 4 **unfreezes the whole representation**: build **one heterogeneous GNN** —
surface-vertex nodes (raw MaSIF input channels) + atom nodes (chemistry), message passing along covalent
bonds and vertex↔atom edges — trained **from scratch** to jointly learn:

- **(a) conformer invariance** — an interface atom's embedding is the same across {holo + the 5 AF3 samples}
  of that protein, and
- **(b) interface atom-atom correspondence** — contacting atoms across two partners match under a learned
  complementarity operator.

**Sharpened success criterion:** close the **addressable induced-fit gap (+0.069)** so that **querying with
any one of the 5 AF3 samples retrieves the true holo partner ≈ as well as the holo query would**, without
harming holo→holo. The ~23% structural-mismatch monomers are **out of scope** (no query-side fix exists —
`07-phase3-results.md`). Architecture/objective/data/milestones are fully specified in `08-phase4-design.md`.

---

## 1. Autonomy contract & definition of done

- **You do not ask permission per step.** Make the call, record the reasoning in `docs/progress/phase4-log.md`
  (§3), proceed. The user reads it live and steers via `docs/09-phase4-user-comment.md`; honor comments at the
  next step boundary. **A comment is steering, not a stop signal — never stop just because you addressed one
  or finished a sub-step.** Pick up the next step.
- **One hard gate only — spend.** You have a **CHF 100 compute budget for this unsupervised session** (a
  guardrail so you don't spend heavily without oversight — the overall project budget is far larger). Spend
  it freely on cheap/smoke/subset work. **Before any LARGE spend — specifically the full ~4,943-complex
  Stage-A training run or large-scale AF3/Chai conformer generation across the whole set — STOP, write a
  checkpoint** (results so far + the ask + a cost projection) under a clear header in
  `docs/09-phase4-user-comment.md`, and **wait**. Every Kuma `sbatch` prints a cost estimate; log cumulative
  CHF in the running log. Right-size: cheapest sufficient resource, a debug/short run before a long one.
- **Done for this arc (arc-1)** = `docs/10-phase4-results.md` credibly answers:
  1. **M0 built + verified** — the heterogeneous graph builder runs, and the **rotation-invariance sanity
     test passes** (rotate a structure → edge features + embedding invariant). This is make-or-break; a leak
     here invalidates everything downstream.
  2. **M1 pipeline validated + a preliminary signal** — the Stage-A correspondence trainer (learned bilinear
     `T` + InfoNCE + hard negatives, holo-only) **trains** on a small holo subset and produces a **held-out
     holo→holo descriptor-separation AUC**, with a **Stage-A full-set GPU cost estimate**.
  3. **The M1 feasibility verdict, honestly framed** — is a from-scratch GNN on track to match MaSIF's ~0.90
     holo ceiling, or not? If the subset signal is far below ~0.90, say so — that is a valid finding and gates
     everything. Then **checkpoint** for the full-set scale-up (which exceeds this session's budget).
  - Then `touch logs/PHASE4_ARC1_DONE`.
- **If permanently blocked**, still write `docs/10-phase4-results.md` (progress + blocker + provisional
  recommendation) and `touch logs/PHASE4_ARC1_DONE`. A crash is cheap; a confident wrong result is expensive.
  A negative result, honestly verified, is a valid finish.

---

## 2. What Phases 1–3 established (do NOT repeat these)

Full detail: `docs/02-phase1-results.md`, `04-phase2-results.md`, `07-phase3-results.md`. The binding lessons:

1. **The frozen MaSIF descriptor is a STRONG ceiling.** Head-only learning on top of it recovered only
   **+0.016 of the +0.069** induced-fit gap; **complex-count scaling flatlined** (52→128: +0.014→+0.016).
   More data on a frozen-bottleneck architecture will not close the gap. → Phase 4 goes *below* the descriptor.
2. **The chemistry graph was inert *as tested* — late-concatenated; the descriptor never message-passed along
   a bond.** Phase-2 (graph-alone) was a NO-GO for the same reason. Phase 4 tests the untested coupling:
   descriptor propagating *through* the connectivity graph. Do not re-run late-concat or graph-alone.
3. **Absolute metrics always** (never only holo→X differentials — a head that lowers holo shrinks the gap
   without improving AF3; the Phase-2 confound). Report absolute AF3→holo and holo→holo every eval.
4. **Guardrails paid off** — shuffled controls (~0.5), complex-level splits, per-complex spread, willingness
   to report NO-GO. Keep that bar.

---

## 3. The live documents + async-comment protocol

Maintain, per the `slurm-claude-agent` naming convention (docs get a 2-digit index in generation order; the
running log lives in `docs/progress/`; the brief is top-level):

| file | your job |
|---|---|
| `PHASE4_HANDOFF.md` | this brief — **read**, don't rewrite. |
| `docs/progress/phase4-log.md` | **real-time running log.** Append a `## <n>. <title>` header the moment you start a step, then reasoning/decisions/commands as you go (before + during, not only after). Log **cumulative CHF** and **every Kuma job-id**. Keep a **RESUME STATE** block (running job-ids, next commands) so a restart reattaches without double-submitting. |
| `docs/08-phase4-design.md` | the design/plan — evolve it as you learn; state any divergence explicitly. |
| `docs/09-phase4-user-comment.md` | **async steering channel** (protocol below). |
| `docs/10-phase4-results.md` | results, tables, honest verdicts, recommendation. Every number traces to a committed artifact + a recoverable command. State "pipeline ran" separately from "result is valid." |

**Async protocol:** initialize `09-phase4-user-comment.md` near-empty and **mirror each running-log step
header** into it. At **every step boundary** (and ≥ every ~30 min during long waits) re-read it; for any new
`### 🧑 USER:` comment, reply inline under it with `### 🤖 AGENT:` (acknowledge + concretely how you'll act,
or with evidence why you'll adapt it), then do it and keep going. Never edit the user's lines. Post a **spend
checkpoint** here before crossing the budget gate in §1 — that is the only routine reason to pause.

---

## 4. Assets (verified present — reuse, don't reinvent)

**Reference MaSIF surface + input features (the vertex-node inputs):** `masif-neosurf-af2/masif-neosurf_v0.1.sif`
(TF1, inference-only) → PDB → surface mesh + per-vertex input channels (shape index, curvature, electrostatics,
hydropathy, H-bond) + 80-D descriptor. **Phase 4 needs the raw input channels + the mesh, not (only) the 80-D
output.** Reuse `src/masif_graph/{io,surface,pairs,metrics}`; the Phase-2 atom-graph builder in
`src/masif_graph/graph/`; the Phase-3 AF3 relabel/identity-mapping + M1 eval in `src/masif_graph/af3/` and
`experiments/run_m1_*.py`.

**Conformer generation (compute is NOT the constraint this phase — see §1):** AF3 container
`/work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif` (real entrypoint
`/AlphaPulldown/alphafold3/run_alphafold.py`; weights `/work/upthomae/Meng/AF3_weights`; DBs
`/work/lpdi/databases/alphafold3_dbs`; MSA on Jed `--run_data_pipeline --norun_inference`, inference on Kuma
H100 `--norun_data_pipeline --run_inference` with `--cpus-per-task=16 --mem=90G`). **AF3 multi-seed diffusion
= conformational ensembles for free.** Faster same-family samplers for scaling sample count: **Chai-1** (env
`chai`, `/work/upthomae/Meng/Chai_predict/`), **Protenix** (env `protenix`). ESMFold = fast but single
deterministic conformation (breadth only). FASPR = optional sidechain jitter.

**GPU training env:** `/work/upthomae/Meng/conda_envs/atomsurf_h100` (py3.8, H100-ready, has torch/PyG/
DiffusionNet). If the from-scratch GNN needs packages absent there, build a fresh env on Jed (internet) and
stage to `/work`. **Env-split rule:** if graph-building needs biotite/py3.11 (`masif-graph` env) but training
needs py3.8, **precompute graphs to cross-version-safe `.npz` (byte-string keys, no pickle) and join at train
time** — this exact split bit us in Phase 3; reuse the pattern.

**Benchmark / data (Pinder is NOT trustworthy — do not use `pinderMaSIF/` as truth):** holo lists
`data/lists/training.txt` (4,943) / `testing.txt` (959), ids `PDBID_chainA_chainB`. The Phase-3 M1 eval set
(30 complexes, `logs/phase3/m1_ids.txt`) is the **held-out** AF3 benchmark — **keep it disjoint from all
training.** Structural-mismatch ids: `logs/phase3/m1_full/m1_mismatch.json` (exclude from training positives).
Cross-cluster staging: `/work/upthomae/Meng/JED_TO_KUMA/`.

**Envs summary:** this repo = `masif-graph`. Kuma reached only by
`ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch '<cmd>'`; `/work` + `/home` shared,
`/scratch` NOT — stage GPU-side data under `/work`.

---

## 5. Compute & infra (the Jed↔Kuma split; you are a CONDUCTOR)

- **Jed** = CPU, **has internet** (you run here; clone/pip/read papers, build graphs, run MSA, do analysis).
  You are a Jed SLURM job under a supervisor loop that resumes you across crashes/rate-limits.
- **Kuma** = GPU (H100), **separate cluster, no internet**, `sbatch -A upthomae -p h100 -q normal
  --gres=gpu:1`. **1×H100 ≈ CHF 0.52/hr.** Stage data to `/work` first (Kuma `/scratch` is node-local).
  Reattach to already-submitted jobs by job-id from your log — **never double-submit after a resume.**
- **Internet-dependent work happens on Jed**, then stage to `/work` for offline Kuma jobs.

---

## 6. Initial direction — cheap-first, gated (follow `08-phase4-design.md §8`; you may revise with justification)

- **M0 — heterogeneous graph builder + rotation-invariance sanity (CPU, ~free).** Build atom+vertex nodes,
  the three edge types (atom–atom covalent, vertex–vertex mesh, vertex–atom cutoff/top-k first), all geometry
  as **SE(3)-invariant scalars**. **Gate: rotation test passes; shapes/scale sane on 1–2 complexes; vertex
  count capped/coarsened.** Smoke every piece on ONE example before scaling.
- **M1 — can a from-scratch GNN re-earn the descriptor? (THE feasibility gate).** Wire Stage-A
  (correspondence: learned symmetric bilinear `T` + InfoNCE + hard negatives, holo-only). Get it **training**
  on a small holo subset; measure **held-out holo→holo AUC**; ablate vertex-features / vertex–atom edges here
  where data is plentiful. **Estimate full-set Stage-A GPU cost.** Gate for the *project* (not this session):
  held-out holo→holo approaches MaSIF's ~0.90. If a from-scratch GNN can't match MaSIF on holo, that is the
  finding — surface it and stop before invariance. **Checkpoint before the full ~4,943-complex run.**
- **M2 — invariance on real AF3 (next session, post-checkpoint).** Conformer-augmented queries; eval on the
  Phase-3 M1 benchmark; per-sample spread; close a meaningful fraction of +0.069 beating the +0.016 bar.

---

## 7. Guardrails (non-negotiable; `ml-research-guardrails`)

- **Absolute metrics always** (holo→holo and AF3→holo), not just differentials.
- **Rotation-invariance test is a hard gate** — a coordinate leak makes the model learn pose, not chemistry.
- **Shuffled-label control ~0.5** every eval; **complex-level splits**; the Phase-3 M1 eval set stays
  **untouched by training**; watch the circular trap (train-conformer / eval-same-conformer).
- **Guard embedding collapse** — invariance is baked into the contrastive task (design §5.3), not a standalone
  penalty; keep any explicit consistency term small. Report per-complex spread; ≥3 seeds where feasible.
- **Never fabricate**; every number traces to an artifact + command; "pipeline ran" ≠ "result is valid."
- **Try to break your own good news before believing it.** Log **cumulative CHF** + **all Kuma job-ids**.

---

## 8. Read first
`docs/08-phase4-design.md` (the full plan — READ IN FULL) · `docs/07-phase3-results.md` (why the frozen line
is exhausted; the +0.069 target; the mismatch out-of-scope) · `CLAUDE.md` · `docs/00-context-and-goals.md`
(D1/D3/D6 — Phase 4 reopens D1, commits D3-A, flips D6) · this file. Skills: `ml-research-guardrails`,
`slurm-claude-agent`, `connect-to-kuma`.
