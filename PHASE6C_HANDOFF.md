# PHASE 6 — Workstream C — HANDOFF (autonomous headless agent)

You are a **headless Claude Code agent** running unattended on SLURM (Jed). No human will
answer questions until this finishes. **Decide, document, and keep going.** A bash supervisor
loop (`scripts/phase6C_agent.sbatch`) resumes you if you stop — so never exit on a wall; pick
up the next step. You own a **standing goal**, not a checklist.

Invoke the **`ml-research-guardrails`** skill continuously for any training / splitting /
eval / result-reporting. Ethos: *try to break your own good news before you believe it; a
crash is cheap, a confident wrong result is expensive.*

---

## 1. Definition of done (the deliverable)

Workstream C is a **ligand-capable unified retrieval encoder**. You are DONE when **C(c)** is
complete and self-verified:

- A unified encoder (input **26-D** atom features, `src/masif_graph/p6/atoms.py`) retrained on
  a **combined corpus** = PPI complexes (≥3,000; see B verdict) **+** protein–ligand complexes
  (PDBbind refined ~5.3k), with a clean split, and evaluated on **three** axes:
  1. **Do-no-harm PPI gate** (the Phase-5 retrieval gate) — the combined model must not
     meaningfully regress PPI retrieval vs the PPI-only Phase-5 encoder.
  2. **Mixed held-out** retrieval (protein–ligand held-out val, cluster-clean).
  3. **Neosurface benchmark** — MolGlueDB (114 ternary PDBs, `data/molgluedb_benchmark_pdbs.txt`)
     and/or the masif-neosurf 13-target set, inference-only.
- Results written to **`docs/19-phase6C-results.md`** with honest verdicts, per-item spread,
  shuffled-control (~chance), and both filtered+unfiltered numbers; every number traceable to a
  committed artifact + a recoverable command.
- **A negative result, honestly verified, is a valid finish** (e.g. "transfer does not occur"
  or "neosurface benchmark too small to conclude"). State "pipeline ran" separately from
  "result is valid".

**Stop signal:** when done + self-verified, `touch logs/PHASE6C_DONE`. If permanently blocked,
write the blocker + a provisional recommendation into the results doc, then `touch
logs/PHASE6C_DONE`. (A `logs/PHASE6C_GIVEUP` file also stops the supervisor.)

---

## 2. Compute model — you are a **CONDUCTOR**. Budget = **CHF 100 to SPEND**.

The retrain needs a **GPU (Kuma)**; the preprocessing is **heavy CPU (Jed)**. Your own SLURM
job is small and long-lived — its purpose is to **submit and monitor CHILD jobs** that hold the
real compute. The CHF 100 is a **budget to use**, not a ceiling on your own job.

- **Jed CPU children** (surface preprocessing, featurization, splitting): `sbatch` directly from
  inside your job (nested sbatch works). **Always `sbatch --test-only` first** — it prints the
  CHF cost estimate — then submit. Prefer array jobs for the ~5k-complex preprocessing.
- **Kuma GPU children** (VICReg + retrieval training): you **cannot** sbatch Jed→Kuma. Stage
  data to shared `/work/upthomae/Meng/...`, then over ssh rsync to Kuma `/scratch` and submit:
  ```
  ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch \
    'sbatch -A upthomae -p h100 -q normal --gres=gpu:1 --time=... /work/upthomae/Meng/<job>.sbatch'
  ```
  Poll with `ssh ... 'squeue -j <id>'`; **back off between polls** (don't busy-poll — it burns
  context/tokens). `/home` + `/work` are shared Jed↔Kuma; `/scratch` is NOT. See the
  `connect-to-kuma` skill.
- **Track cumulative spend** in `docs/progress/phase6C-log.md`. Right-size everything (a debug
  run before a long one). **Halt and report** (in the user-comment doc) if a run would exceed the
  cap. Phase-5's whole build was ~CHF 38, so CHF 100 is ample for one combined retrain + preprocessing.

**Reference cost anchors:** 8 CPU cores × 24h on Jed ≈ CHF 1. A Phase-4/5 VICReg+retrieval run
on one Kuma H100 was a few CHF. AF3/MSA are NOT needed here (no new apo generation in C).

---

## 3. Locked decisions — DO NOT relitigate

- **D-C1 dataset:** start with **PDBbind refined ~5.3k** (`data/pdbbind/` → 19,442 complexes
  symlinked; refined index = `data/pdbbind_index/index/INDEX_refined_data.2020`, 5,316 rows).
  General set / nucleic acids (D-C2) are deferred — do NOT pull them in.
- **PPI ≥ 3,000** (Workstream B verdict: PPI retrieval is **SATURATED** by ~3,000 complexes —
  do not scale PPI beyond that; more PPI data buys nothing). Use ~3,000 PPI complexes in the mix.
- **Unified 26-D atom featurizer** — `src/masif_graph/p6/atoms.py` (`DIM=26`), already built &
  validated (protein 1CQ3_A → (1687,26); ligand 5hls → (26,26)). Protein + ligand share ONE
  feature space so PPI complementarity transfers to protein–ligand. The 14→26 dim change forces
  a **full retrain** and **re-featurization of the PPI npz to 26-D**.
- **Ligand representation — Path B (LOCKED, committed HEAD 410238d):**
  **ligand heavy atoms are GRAPH NODES**, not a computed ligand surface. Rationale: the
  masif-neosurf `score_binder` `.sif` ligand-surface path is WIP/buggy (6 integration fixes
  applied, still hits the protonation `heavy<total` invariant in `extract_ligand`; chasing it is
  a fragile rabbit hole at 5k scale). So:
    - Build the **protein** surface with the **normal, reliable** `.sif` pipeline
      (`scripts/p6_protein_surface_one.sh`, validated on 5hls).
    - Add **ligand heavy atoms as graph nodes** with unified 26-D features
      (`atoms.ligand_features(mol)` via RDKit), ligand bonds as covalent edges, and
      **vertex↔atom edges** from protein-pocket surface-vertices → nearby ligand atoms.
    - Encoder emits embeddings for protein surface atoms **and** ligand atoms.
    - **Training pair:** protein-pocket-surface ↔ ligand-atoms (in-contact, e.g. ≤ ~5 Å).
    - This gives up ligand-derived surface *vertices* (shape) but keeps the atom-level
      **protein↔ligand complementarity** — which is the core goal. Robust + self-contained.
  - *Optional stretch (only if Path B works AND budget allows):* a from-scratch ligand-surface
    Tier-0 (shape) reusing `computeMSMS` (which DOES work — it produced 220 ligand-owned vertices;
    only `extract_ligand`'s chemistry is buggy) + RDKit Gasteiger charges. **Not required for
    C(c).** Do NOT block the deliverable on it.
- **Objective:** shared encoder + shared bilinear `T`; positive pairs = PPI
  (protein-surf ↔ protein-surf) **AND** protein–ligand (protein-pocket-surf ↔ ligand-atoms).
  Mixture train + mixture held-out val. Score = `median_i max_j zᵢᵀ T zⱼ` (Phase-4/5 form), with
  **`--center` (DC-offset centering) mandatory** at eval (Phase-4 finding: centering converts
  invariance→retrieval; without it the retrieval collapses).
- **Leakage control:** protein **sequence-cluster** (mmseqs2 ~30% id) **+ ligand scaffold**
  cluster. Held-out val must be clean on BOTH axes. (Phase-5 had a real train/eval leak — be
  paranoid; filter eval vs the ACTUAL train ids, not the intended split.)
- **Recipe (from Phase-4 scaleup, stable):** VICReg pretrain (var 2.0 / cov 0.04) → retrieval
  fine-tune; freeze-τ @ 0.1, T weight-decay 1e-3, lr 5e-4 cosine, d=64 / d_out=32 / L=4.
  This recipe reaches the frozen holo ceiling and is anti-collapse-hardened. Keep it unless you
  have measured reason to change (log any divergence).

---

## 4. Read-first pointers (get these into context on first run)

- `docs/16-phase6-design.md` — Phase 6 design; **§5 = Workstream C** training design + D-C1/D-C2.
- `docs/progress/phase6C-log.md` — the running log incl. the Path-B pivot & the 6-fix history.
  **Re-read its RESUME STATE on every resume.**
- `docs/progress/phase6-log.md` — Workstreams A (I/O contract) & B (data-scaling → SATURATED).
- `src/masif_graph/p6/atoms.py` — the unified 26-D featurizer (protein + ligand). Read it before
  writing the graph builder; reuse `ligand_features` / `protein_features`.
- Phase-5 assets (the do-no-harm gate + harness): `docs/15-phase5-results.md`,
  `src/masif_graph/p5/retrieval_bench.py`, `scripts/p5_gate.sbatch`. The PPI-only encoder to beat
  / not-regress: `/work/upthomae/Meng/phase4/ret_full_ctr_best.pt` (14-D input — you will train a
  fresh 26-D one; compare on the SAME clean PPI eval).
- Phase-4 training recipe: `scripts/phase4_vicreg_stageA.sbatch`,
  `scripts/p4_retrieval_proof_center.sbatch`, `src/masif_graph/` graph/train/score modules.
- `scripts/p6_protein_surface_one.sh` — VALIDATED protein-surface build for a PDBbind complex via
  `.sif` (synthetic id `pl${ID}`; runs 01/04/desc). This is your Path-B protein-surface primitive.
- `data/DATASETS_SUMMARY.md`, `data/pdbbind_index/`, `data/molgluedb_benchmark_pdbs.txt`.

---

## 5. Suggested step order (you decide; log divergence)

- **C(a).3** — Path-B ligand-aware graph builder: protein surface (normal `.sif`) + ligand heavy
  atoms as nodes (unified 26-D) + covalent + vertex↔atom edges → `.npz`/`.pt`. **Validate ONE
  PDBbind complex end-to-end** (read the actual output: node counts, edge types, feature ranges,
  a contact pair) before scaling. `5hls` protein surface already built — start there.
- **C(b).1** — Re-featurize the ~3,000 PPI complexes to **26-D** (unified precompute). Preprocess
  PDBbind refined ~5.3k (protein surfaces + ligand-node graphs). Fan out as **Jed array child
  jobs**; smoke a few, verify completeness before trusting the stage; re-run idempotently.
- **C(b).2** — Combined corpus + **cluster-clean split** (protein seq-cluster + ligand scaffold);
  carve a mixed held-out val. Verify no leakage against the ACTUAL train ids.
- **C(c).1** — Extend objective/dataloader for protein–ligand pairs (protein-pocket ↔ ligand-atom
  contrastive) alongside the PPI pairs. Reuse the Phase-4 train loop; keep VICReg+retrieval.
- **C(c).2** — Retrain the unified encoder (26-D, combined) on **Kuma GPU**. De-risk with a short
  run first; watch for collapse (z_std, τ-floor, ‖T‖ runaway — the Phase-4 anti-collapse recipe
  guards these).
- **C(c).3** — Eval: (1) do-no-harm PPI gate, (2) mixed held-out, (3) neosurface benchmark. Write
  `docs/19-phase6C-results.md`. Shuffled control ~chance. Then `touch logs/PHASE6C_DONE`.

---

## 6. Safety rails / gotchas (learned the hard way)

- **`/scratch` has a 30-day cleanup that deleted `.git` + untouched source once.** Commit +
  push often (`git push` to `git@github.com:meng-yx/masif-graph.git`). New code lives here; the
  /work backup at `/work/upthomae/Meng/phase4/src` saved the last wipe. **Commit after each
  working step.** (Do not commit `data/**` except `data/lists/`; `.npy/.npz/.pt/.ply` are ignored.)
- **Gates/big evals OOM-kill on the LOGIN node** — always run heavy eval on a **compute node**
  (sbatch), never inline on login.
- **No double-submit after a resume.** Before submitting any child, check the RESUME STATE block
  in `docs/progress/phase6C-log.md` for a recorded job-id and `squeue`/`ssh squeue` for it.
  Record every child job-id in the log the moment you submit.
- **Verify data completeness before trusting a stage** (races: `squeue` right after `sbatch` can
  read empty). Watchdog / `scancel` stragglers over a sane elapsed limit.
- **`.sif` `00-pdb_download` uses a dead FTP endpoint** — fetch PDBs from
  `https://files.rcsb.org/download/{PDB}.pdb` then protonate (see `scripts/m0_run_one.sh`).
- Reference tools that call `git rev-parse --show-toplevel` must run **inside** the
  `masif-neosurf-af2/` repo, not this parent repo.
- The `.sif` needs `--bind /scratch/ymeng/masif-graph` (and `/work`) to see project files.

---

## 7. Env setup (already in the sbatch, restated for inline shells)

```
export HOME=/home/ymeng
export PATH=/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH
conda activate masif-graph     # this repo's env (torch/PyG/e3nn/rdkit CPU)
cd /scratch/ymeng/masif-graph
```
Auth = Claude Max OAuth (`~/.claude/.credentials.json`), auto-refreshes while the node has
internet. Jed compute nodes have internet (verified in smoke). `singularity` at
`/usr/bin/singularity`.

---

## 8. Async steering — read every step boundary

`docs/18-phase6C-user-comment.md` is the user's steering channel. **Re-read it at every step
boundary** (and periodically during long waits). For any new `### 🧑 USER:` comment, reply
inline `### 🤖 AGENT:` — acknowledge, say concretely how you'll act (or, with evidence, why you
adapt it), then **do it and keep going**. Never edit the user's lines. A comment is steering,
NOT a stop signal. The ONLY routine reason to pause is a **budget-gate checkpoint**: if a
planned action would cross a budget gate, post results-so-far + the ask + a cost projection
there and continue other work while noting it.

**Keep the running log (`docs/progress/phase6C-log.md`) append-only and current** — append a
`## <n>. <title>` header the moment you START a step, log reasoning/commands/spend/job-ids as you
go, and keep the **RESUME STATE** block at the end accurate. It is your memory across restarts.
