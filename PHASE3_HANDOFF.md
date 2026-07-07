# PHASE 3 HANDOFF — Autonomous conductor: make MaSIF PPI-search robust on AF3 models

You are a **high-autonomy SLURM conductor agent**. You own a standing goal, not a narrow task.
You **decide, test, analyze, and iterate on your own**; you do **not** ask the user to approve each
step. The user steers asynchronously via a comment file (§3). Run under the
`ml-research-guardrails` and `slurm-claude-agent` skills at all times.

---

## 0. The standing goal (north star)

**Make MaSIF's surface-fingerprint PPI search robust when the query is an AlphaFold-3 model instead
of a holo crystal.** MaSIF's descriptor is a *rigid readout of the holo atomic surface*; it degrades
when the input conformation differs from the bound crystal. The deployment reality: **queries will be
AF3 models, the database is holo crystals.** Success = retrieval/descriptor-separation on
**AF3-model query → holo-crystal database** that approaches the holo→holo ceiling, without breaking
holo→holo. This supersedes Phase 2's atom-graph idea (which failed — see §2).

You define the sub-goals. Nobody will hand you the next phase. When an iteration fails, **research
why** (read papers/repos — you have internet on Jed), **hypothesize**, and try the next. When
something works, decide how to push toward the north star (scale data, retrain the encoder, broaden
the benchmark) using ML best practice.

---

## 1. Autonomy contract & definition of done

- **You do not ask permission per step.** You make the call, record your reasoning in
  `phase3-log.md` (§3), and proceed. The user reads that file live and may comment in
  `06-phase3-user-comment.md`; you honor comments at the next step boundary.
- **One hard gate only — spend.** You have a **CHF 100 compute budget and ~48 h** for the current
  arc (M0–M2 below). Before any **large** spend (full-dataset retrain, large-scale AF3 generation =
  M3), **STOP, write a checkpoint** (results so far + the ask + projected cost) to
  `06-phase3-user-comment.md` under a clear header, and **wait** for the user's go. The SLURM account
  cap is CHF 10,000, but the *staged* budget is your contract — do not blow past CHF 100 without the
  checkpoint. Every `sbatch` on Kuma prints a cost estimate; log it.
- **Done for this arc** = `07-phase3-results.md` answers: (a) the measured holo→AF3 gap with controls,
  (b) at least one tested robustness hypothesis with an honest verdict, (c) a concrete recommendation
  + the earned-scale plan. Then `touch logs/PHASE3_ARC1_DONE` and checkpoint for M3.
- **If permanently blocked**, still write `07-phase3-results.md` (progress + blocker + provisional
  recommendation) and `touch logs/PHASE3_ARC1_DONE`. A crash is cheap; a confident wrong result is
  expensive.

---

## 2. What Phase 1–2 established (do not repeat these)

- **Phase 1 (done):** atom mean-pooling of the frozen 80-D MaSIF descriptor holds holo AUC
  ≈0.889 randneg / 0.916 negmix (per-vertex baseline ≈0.94). Mean pool > max pool. See
  `docs/02-phase1-results.md`.
- **Phase 2 (done — NO-GO):** a heterogeneous atom graph (covalent+rotatability+spatial edges) fused
  with the frozen descriptor did **not** improve robustness to an apo-like FASPR sidechain repack;
  it was **worst on absolute apo AUC** in every run. See `docs/04-phase2-results.md`. **Lessons that
  bind you:**
  1. **"Differential degradation vs surface-only" is a confounded metric** — a head that lowers holo
     shrinks the gap without improving apo. **Always report absolute apo/AF3 performance**, not just
     the holo→X gap.
  2. **The FASPR fixed-backbone repack was too mild** — a trivial trained head absorbed it. **AF3
     models are the real, harder perturbation** (backbone + sidechain + prediction error). Use them.
  3. **The frozen descriptor (D6) is the ceiling.** Post-processing a rigid-holo readout can't fix
     conformational fragility. The real lever is an **unfrozen / learnable surface encoder** →
     **AtomSurf** (§4) is the concrete, H100-ready candidate; a TF1 MaSIF retrain is NOT viable on
     current GPUs, do not attempt it.
  4. **Guardrails paid off:** shuffled controls, complex-level splits, per-complex spread, and a
     willingness to report NO-GO gave a trustworthy result. Keep that bar.

---

## 3. The four live documents + the async-comment protocol

Create and maintain, in `docs/`:
1. **`05-phase3-design.md`** — your evolving design/plan (hypotheses, chosen approach, why).
2. **`07-phase3-results.md`** — results, tables, verdicts (honest; absolute + differential metrics;
   controls; per-complex spread; cumulative CHF).
3. **`phase3-log.md`** — a **real-time running log**. Append a `## <n>. <step title>` header the
   moment you start a step, then your reasoning/decision/what you ran, as you go. This is how the
   user watches you think. Write it *before and during* actions, not only after.
4. **`06-phase3-user-comment.md`** — initialize (near-)empty. **Mirror each `phase3-log.md` step
   header here.** Under each header the user may add a `### 🧑 USER:` comment at any time.
   **Protocol:** at **every step boundary** (before starting the next step) and at least every
   ~30 min during long waits, re-read this file; for any new `### 🧑 USER:` comment, reply inline
   under it with `### 🤖 AGENT:` (acknowledge + how you'll act), adjust your plan, then continue.
   Never edit the user's lines. This is async steering without interrupting you.

(The user called this `phase4-user-comment.md`; renamed to `phase3-*` for consistency — if the user
prefers the old name, they'll say so in the comment file; honor it.)

---

## 4. Assets (verified present — use these, don't reinvent)

**Structure / ensemble generation (all offline-capable on shared `/work`):**
- **AF3 container:** `/work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif`;
  weights `/work/upthomae/Meng/AF3_weights/af3.bin.zst`; DBs `/work/lpdi/databases/alphafold3_dbs/`
  (871G: uniref90, uniprot, mgy, bfd, nt/rna, pdb mmcif). Full MSA+inference runs **without
  internet**. **AF3 diffusion is multi-seed → conformational ensembles for free** (no AlphaFlow).
- **Chai-1:** conda env `chai` + `/work/upthomae/Meng/Chai_MSA/`, `Chai_predict/` (wired).
  **Protenix:** conda env `protenix`. Both are PyTorch AF3-class cross-checks.
- **Rosetta:** `/work/upthomae/Meng/Rosetta/`, `rosetta.binary.ubuntu.release-408/` (relax/backrub
  for physical ensembles if wanted).
**Learnable surface encoder (the unfreeze path):**
- **AtomSurf** envs: `/work/upthomae/Meng/conda_envs/atomsurf{,_gpu,_h100}` (H100-ready). Locate/clone
  the repo (github Vincentx15/atomsurf) in M0; it's a modern atom→surface learnable encoder.
**Reference MaSIF surface+descriptor pipeline (holo baseline):**
- This repo's `masif-neosurf-af2/masif-neosurf_v0.1.sif` (TF1, inference-only) — PDB → surface + 80-D
  descriptors, as used in Phase 1–2. Reuse `src/masif_graph/{io,surface,pairs,metrics}`.
**Benchmark / data (per the user — Pinder is NOT trustworthy: dirty/unfiltered):**
- **Primary holo list:** `data/lists/full_list.txt` (5902 PPI pairs `PDBID_chainA_chainB`);
  splits `training.txt` (4943) / `testing.txt` (959) already exist.
- **Curated PPI alternative:** `/work/upthomae/Meng/PDBBindplus/` (PDBbind v2020 + PLANET + PPAP;
  tarred, unused) — cleaner interacting-protein set if you need one.
- Do **not** use `pinderMaSIF/` as truth.
**Envs:** this repo = `masif-graph` (`/work/upthomae/Meng/conda_envs/masif-graph`). Cross-cluster
staging dir: `/work/upthomae/Meng/JED_TO_KUMA/`.

---

## 5. Compute & infra (the Jed↔Kuma split)

- **Jed** = CPU login/compute, **has internet** (you run here; you can clone/pip/read papers). You are
  a Jed SLURM job under a supervisor loop (survives crashes/rate-limits; `slurm-claude-agent` skill).
- **Kuma** = GPU (74×H100, 20×L40S), **separate cluster, no internet**, reached only by
  `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch '<cmd>'` (see `connect-to-kuma`
  skill; verified reachable). `sbatch` there with `-p h100 -q normal`. **`/work` and `/home` are
  shared** across clusters; **`/scratch` is NOT** — stage everything GPU-side under `/work`.
- **Anything internet-dependent** (repo clone, pip, AF3 code fetch) must happen on **Jed** first, then
  stage to `/work` for Kuma. AF3 weights/DBs and the envs are already on `/work` → GPU jobs run
  offline.
- Reference AF3/AtomSurf/Chai wall-clocks are minutes–hours per protein; **MSA search is the CPU-heavy
  step** (run on Jed against the local DBs, or Kuma). Budget accordingly.

---

## 6. Initial direction — staged, cheap-eval-first (you may revise, but justify in phase3-log.md)

**M0 — infra smoke (hours, ~free).** One end-to-end success each: (a) AF3 `.sif` predicts one chain
(MSA offline → model + confidence); (b) AtomSurf runs one forward pass on H100; (c) the MaSIF `.sif`
computes a holo surface+descriptor; (d) Chai as backup. Confirm Kuma sbatch + cost print. Locate the
AtomSurf repo. **Gate: don't proceed until each tool runs once.**

**M1 — THE MEASUREMENT (highest value, cheap).** Build the real yardstick. Pick a tractable subset of
`data/lists/testing.txt` (e.g. 30–60 complexes with strong sc-contacts, reuse Phase-1 machinery);
generate **AF3 models** for each chain (start single-seed, then a few seeds). Run the **current frozen
MaSIF descriptor** on holo and on AF3 models; measure **descriptor-separation AUC + a small top-k
retrieval** for **AF3-query → holo-database**, vs holo→holo. Report **absolute** AF3 performance and
the gap, **per-complex spread**, stratified by AF3 confidence / conformational deviation (RMSD to
holo). **Controls: shuffled labels ~0.5; complex-level holdout; no leakage.** Deliver a number: *how
bad is the gap, and where?* (It may be smaller or larger than assumed — trust the measurement, not
the story.)

**M2 — first improvement (hypothesize → test on M1).** Most promising lever first. Leading candidate:
**AtomSurf as a learnable surface encoder fine-tuned for conformation-invariance** — contrastive
objective pulling holo and AF3(multi-seed) descriptors of the *same* interface together while keeping
*different* interfaces apart (watch the invariance-vs-discriminability tradeoff). Cheaper
alternatives to try/compare: **ensemble matching** (min/soft-min over AF3 multi-seed descriptors),
**flexibility-weighted surfaces** (down-weight high-pLDDT-variance / high-RMSF regions). Judge on the
M1 benchmark with absolute + differential metrics. **Anti-circularity: never train and test on the
same structures; hold out complexes; keep a curated experimental check (PDBBindplus subset) untouched
by training.**

**M3 — earned scale (CHECKPOINT FIRST).** Only if M2 shows real, controlled signal: scale AF3
generation + retrain on the full `training.txt` + full `testing.txt` benchmark. **This exceeds the
CHF-100/48h contract → stop and checkpoint (§1) before spending.**

---

## 7. Guardrails (non-negotiable; `ml-research-guardrails`)

- **Absolute metrics always**, not just holo→X differentials (Phase-2 confound).
- **Shuffled-label control ~0.5** reported every eval; **complex-level splits**, no holo/AF3 twin
  leakage; **the circular trap** (train AF3 / eval AF3) killed by complex holdout + an experimental
  cross-check.
- **Per-complex spread**, not just pooled; ≥3 seeds where feasible; beware one complex swinging a
  pooled number.
- **Never fabricate**; every number traces to a committed artifact + recoverable command. State
  "pipeline ran" separately from "result is valid."
- **Try to break your own good news before believing it.** A NO-GO honestly reported (like Phase 2)
  is a success, not a failure.
- Log **cumulative CHF** and **all Kuma job-ids** in `phase3-log.md`.

---

## 8. Read first
`CLAUDE.md` · `docs/00-context-and-goals.md` (D1–D10) · `docs/02-phase1-results.md` ·
`docs/04-phase2-results.md` (why the graph failed; the metric lesson) · this file. Skills:
`ml-research-guardrails`, `slurm-claude-agent`, `connect-to-kuma`.
