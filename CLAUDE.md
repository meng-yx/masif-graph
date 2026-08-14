# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaSIF-graph: atom-graph-enhanced surface fingerprints for protein–protein / neosurface
search. A ground-up **rewrite** of MaSIF-PPI-search whose fundamental unit is the **surface
heavy atom**, not the mesh vertex. It is *not* a fork of the reference code.

## Central goal (north star) — close the holo→apo gap
MaSIF's learned surface descriptor is implicitly tuned to **bound-state (holo, crystal)
sidechain rotamers** and degrades on **apo / AF2 / unbound** conformations (false positives &
negatives) **despite a high holo-benchmark AUC**. The atom graph exists chiefly to encode
**connectivity and bond rotatability** (how sidechain atoms can move), so the representation is
**robust to sidechain conformation**. **Improving holo→apo generalization is the project's
success criterion.** Consequence to keep in mind: the current holo-only validation set cannot,
by itself, demonstrate this benefit — evaluation must include apo-like structures (Phase 2 uses
fixed-backbone sidechain repack as the controlled proxy; see `docs/03-phase2-design.md`).

### North star, sharpened (2026-08-11, with the user) — read this, not just the paragraph above
Train a **generalizable model that evaluates biomolecular interactions from apo / predicted
(AF2/AF3) structures**, without requiring the induced-fit holo conformation. A partner may be a
protein, a protein complex, a small molecule, a protein–small-molecule complex, later a nucleic
acid. The model should learn **per atom, conditioned on local environment, how much shape and
chemical mismatch is tolerable** — an implied latent conformational landscape reachable from the
apo structure. Deployment mode is **retrieval/screening**; the training signal is **evaluation**:
`P(A and B form a biologically meaningful assembly)`. **Not** Kd/Ki prediction. Molecular glue is
the LAST deployment target (too little data, and Phase 7 showed it is a third relation, not
PPI ∘ P–L). Full contract: `docs/23-phase8-design.md`.

**Status: Phases 1–7 complete; Phase 8 Stage A COMPLETE; Stage R COMPLETE — its gate FAILED.**
- Phase 5 **met the robustness gate** — the from-scratch invariant encoder beats frozen MaSIF on
  AF3-apo retrieval and is conformation-robust. Phase 7 extended that to the ligand axis.
- The **atom/chem-graph thesis of §2 in `docs/00` is NOT earned** — five independent nulls
  (Phases 2→7). Robustness came from invariant features + the contrastive recipe. Do not invest
  further in chemistry-graph elaboration.
- **Two representation upgrades failed the capacity gate** (Phase 6C unified 26-D atoms; Phase 7
  full ligand surfaces): train-set protein–ligand retrieval stayed ~0.11. The bottleneck is the
  **objective and the label**, which is what Phase 8 changes.
- Phase 7 also showed **capacity competition** (ligand surfaces cost PPI −0.169 and hurt PPI
  *training*), and that **one seed lies** (the Phase-6C −0.041 gap vanished at 2 seeds).
Design still runs ahead of code — read the docs before building.

## Read before writing code
- `docs/00-context-and-goals.md` — north star: hypothesis, key design decisions **D1–D10**,
  phasing, evaluation, risks. The D-decisions are the load-bearing forks.
- `docs/01-phase1-design.md` + `docs/02-phase1-results.md` — Phase 1 (done): the pooling probe
  and its CONDITIONAL GO (mean pooling; ~0.03–0.05 holo pooling cost).
- `docs/03-phase2-design.md` — **current work:** holo→apo robustness via a heterogeneous atom
  graph (connectivity + bond rotatability); the re-targeted gate and the fixed-backbone repack.
- `docs/21-phase7-results.md` + `docs/23-phase8-design.md` — the latest results and the **current
  design contract**. Start here for anything about what to build next.
- `README.md` — human-facing overview and repo layout.
- `docs/22-pymol-viz-guideline.md` — **standing requirement** for any PyMOL visualisation of the
  network input: one `.npz` per training pair, both partners in full, an object for every feature
  the GNN consumes, `{carrier}_{feature}_{left/right}` naming, training positives drawn. Follow it
  whenever the user asks to visualise the model input, and update its §4 table when the
  architecture changes.

When a task touches modelling choices, check the docs first; if you diverge from a
D-decision, say so explicitly. Phase 2 locks D6(freeze)/D3-A/D2/D4 provisionally (see
`03-phase2-design.md §2`); D1-B is the escalation if the graph can't close the gap.

## Training-corpus provenance (state it in every results summary)
**Phases 1–7 trained on HOLO structures only.** Apo has only ever been on the evaluation side:
PPI training 4,767 holo / **0 apo**; PDBbind P–L 5,239 holo / 298 apo (held-out eval only);
Phase-5 eval 301 holo / 284 apo. So the reported holo→apo robustness is **zero-shot**, not learned.
Phase 8 changes this (D8-12) — the holo:apo ratio and the prediction method are **undecided pending
the Stage-A0 benchmark and a user decision**. Never report a training result without saying which
states were in the training set.

## Phase-8 Stage A result and the Stage-R pivot (2026-08-12)
Stage A (`docs/25-phase8A-results.md`) cleared two risks and broke one:
- **A1**: the encoder is **not** sidechain-blind (isolating sidechain atoms costs 70% of the graph's
  information), so Phase-5 robustness is real. **A2**: apo pockets mostly do not close (ratio 1.001;
  20% degrade, sidechain-mediated). **A0**: chai-1 on a shared MSA ≈ AF3 (paired p=0.38); AF3's 5
  diffusion samples cost the same as 1; **a FASPR repack reproduces 91% of the AF3 perturbation**.
- **A3 broke the funnel**: rigid pose from Stage-1 scores succeeds **0/269** while the same fitter
  with true correspondences succeeds 100%. Given a true interface atom as query, the top-1 predicted
  partner is a median **19.4 Å** away. **A4**: the same embeddings add nothing over interface area
  (BSA-only AUROC 0.827 is the bar Stage 3 must beat).
- **Stage R FAILED its gate** (`docs/27-phase8R-results.md`): spatial 17–20 Å vs <5 Å, and chain
  retrieval collapsed 0.644 → 0.024. The decisive control: a **random atom among the partner's true
  interface atoms** scores 19.99 Å / 0.071-within-5 Å and the Phase-6C/7 encoder scores 19.4 Å /
  **0.071** — identical. The encoder finds the interface and is **random inside it**. Ablations are
  flat, it cannot overfit 20 complexes in 60 epochs, and the score matrix is near-uniform
  (effective 1,197 of 1,265). **Keep the Phase-6C/7 encoder; adopt no Stage-R checkpoint.**
  The live option is the pre-registered fallback: **collapse Stages 2 and 3 and score interfaces
  directly**. D8-12 is resolved (repack + chai-1 MSA-free; AFDB rejected).

## The Phase-8 gate (what the current work is deciding)
A **three-stage funnel** (`docs/23-phase8-design.md`, reconciling `docs/11`):
Stage 1 atom-level encoder (EXISTS) → Stage 2 pose prediction → Stage 3 pose-level
`P(biologically meaningful)`. Trained on an ordinal **0/1/2** label: biological assembly = 2,
crystal contact / crystallisation additive = **1** (a real, complementary, non-biological contact —
the exact false-positive mode we must kill), random = 0.

**Two cheap diagnostics gate the whole plan and come first:** (A1) is the encoder modelling
flexibility or merely **ignoring sidechains**? (A2) do **cryptic pockets** close in apo models,
bounding what any model can do from apo input? Either can invalidate the design.

Hard controls that are not optional: a **BSA-only baseline** Stage 3 must beat (interface size is
the obvious shortcut), a **scheduled structure-only ablation** of the conservation features, and
**≥2 seeds for every claim**.

## Commands
```bash
# new-code env (this repo)
conda env create -f environment.yml      # creates env `masif-graph`
conda activate masif-graph
pip install -e .                         # wires up the masif_graph package

ruff check src                           # lint (config in pyproject.toml, line-length 100)
pytest                                   # tests (once tests/ exists)
pytest tests/path::test_name             # single test
```
Runtime deps live in `environment.yml` (conda + pip), **not** in `pyproject.toml`
(`dependencies = []` there is intentional). PyTorch/PyG/e3nn are CPU wheels for now.

## Reference pipeline (`masif-neosurf-af2/`) — used as a tool, not extended
The legacy `masif-neosurf` repo is re-cloned at `masif-neosurf-af2/` (git-ignored) as a
**reference/template** and, for Phase-1 Milestone 0, as an **executable tool** to turn raw
PDBs → surfaces + per-vertex 80-D descriptors that we then pool onto atoms.
- The reference stack (**TensorFlow 1.13 / py3.7 + MSMS/APBS/PyMesh**) has **no conda env** —
  it runs **entirely** from a prebuilt container that ships in the repo:
  **`masif-neosurf-af2/masif-neosurf_v0.1.sif`** (Singularity/Apptainer image, ~1.7 GB). Run
  every reference command inside this `.sif`. (Ignore any mention of a `masif-neosurf-ref`
  conda env in the design docs / README — it does not exist.)
- Reference entry points: `masif-neosurf-af2/preprocess_pdb.sh` (wrapper, supports
  `--ligand/--sdf/--mol2`) → `preprocess_pdb.py`; search/benchmark under
  `masif_search.py`, `computational_benchmark/`, `masif_seed_search/`. Descriptor net and
  utilities live under `masif-neosurf-af2/masif/source/`.
- The **only artifact reused verbatim** is the PDB lists in `data/lists/` (already copied;
  4,943 train / 959 test — line ids are `PDBID_chainA_chainB`). Everything else
  (surfaces, atoms, graphs, descriptors) is regenerated from PDBs per D10.

## Compute (Jed vs Kuma — separate clusters)
- You run on **Jed** (CPU-only login node, has internet). GPUs live on the **separate Kuma**
  cluster — you must `ssh` in to use them (see the `connect-to-kuma` skill; you cannot
  `sbatch` to Kuma from Jed).
- Stage shared code/data on `/home` or `/work` (`/work/upthomae/Meng`). **`/scratch`
  (including this working dir) is NOT shared between clusters.**
- SLURM account `upthomae` is budget-capped. **Do not launch GPU training or large data
  transfers without explicit human go-ahead.** Phase 1 is CPU-feasible by design; every
  GPU-dependent step is gated. Reference descriptor-net inference runs CPU (slow but fine for
  the ~30–50-complex probe).
- **Budget framing — the CHF-100 ceiling is a *per-session* guardrail, not a project total.**
  It bounds a single autonomous/headless agent run that has *no* human in the loop, so a runaway
  loop can't drain the account unattended. It does **not** accumulate across sessions: each new
  session (especially an interactive one with a human present) starts fresh at CHF 100, and prior
  sessions' spend (e.g. the ~CHF 22 logged through Phase-4 arc-1) does **not** count against it.
  Do not track or subtract historical spend. The gating rule above (no GPU/large-transfer launch
  without human go-ahead) is what actually protects the budget; the CHF-100 number is the ceiling
  for one unattended stretch, not a lifetime cap.

## Skills (auto-available; see `.claude/skills/`)
- **`ml-research-guardrails`** — invoke continuously during any training / data-splitting /
  evaluation / result-reporting. Ethos: *try to break your own good news before you believe
  it; a crash is cheap, a confident wrong result is expensive.* Leakage-checks (identical
  complexes + pos/neg construction for the vertex vs atom comparison), shuffled-label
  controls, per-complex spread reporting, honest stop conditions.
- **`connect-to-kuma`** — reaching the Kuma GPU cluster from Jed for SLURM jobs.

## Working norms
- Keep an append-only progress log for long/unattended work (the guardrails skill explains).
- One conda env, **`masif-graph`** (this repo), for all new code. The reference stack is the
  `.sif` container only (no conda env) — used solely to generate probe inputs.
- This is a git repo; **don't commit unless asked.** `data/**` is git-ignored except
  `data/lists/`; model/surface artifacts (`*.ply *.npy *.pt` …) are ignored.
