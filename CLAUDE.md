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

**Status: Phase 1 complete (CONDITIONAL GO); Phase 2 in design.** Phase-1 code exists under
`src/masif_graph/{io,surface,pairs,metrics,align,experiments}` (+ `scripts/`, `logs/m1`,
`logs/m2`, `docs/figures/`); Phase-2 modules (`graph/ perturb/ train/ score/`) are planned, not
yet written. Design still runs ahead of code — read the docs before building.

## Read before writing code
- `docs/00-context-and-goals.md` — north star: hypothesis, key design decisions **D1–D10**,
  phasing, evaluation, risks. The D-decisions are the load-bearing forks.
- `docs/01-phase1-design.md` + `docs/02-phase1-results.md` — Phase 1 (done): the pooling probe
  and its CONDITIONAL GO (mean pooling; ~0.03–0.05 holo pooling cost).
- `docs/03-phase2-design.md` — **current work:** holo→apo robustness via a heterogeneous atom
  graph (connectivity + bond rotatability); the re-targeted gate and the fixed-backbone repack.
- `README.md` — human-facing overview and repo layout.

When a task touches modelling choices, check the docs first; if you diverge from a
D-decision, say so explicitly. Phase 2 locks D6(freeze)/D3-A/D2/D4 provisionally (see
`03-phase2-design.md §2`); D1-B is the escalation if the graph can't close the gap.

## The Phase-2 gate (what the current work is deciding)
Does a graph encoding **atom connectivity + bond rotatability**, fused with the frozen surface
descriptor, make the representation **robust to sidechain conformation** — i.e. degrade less
under an apo-like fixed-backbone sidechain repack than surface-only — **without harming holo
performance**? Holo AUC is a do-no-harm floor, *not* the objective. Do **not** build Phase 3+
(learned pose scorer, aligner hardening, ligands, true apo/AF2 training) until this gate is met.
The graph showing ~zero gain on **holo** is expected — the benefit lives in the apo-like regime.

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
