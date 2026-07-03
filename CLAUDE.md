# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaSIF-graph: atom-graph-enhanced surface fingerprints for protein–protein / neosurface
search. A ground-up **rewrite** of MaSIF-PPI-search whose fundamental unit is the **surface
heavy atom**, not the mesh vertex. It is *not* a fork of the reference code.

**Status: early design / Phase 1 — the code is essentially a stub.** The only Python that
exists is `src/masif_graph/__init__.py`; the `io/ surface/ pairs/ align/ metrics/
experiments/` modules, `scripts/`, and `tests/` described in the docs are **planned, not yet
written**. Expect to create them. Design is far ahead of code.

## Read before writing code
- `docs/00-context-and-goals.md` — north star: hypothesis, key design decisions **D1–D10**,
  phasing, evaluation, risks. The D-decisions are the load-bearing forks.
- `docs/01-phase1-design.md` — current work: per-atom reframing + the go/no-go **pooling
  feasibility probe** (Milestones 0–2), and the proposed `src/masif_graph/` module layout.
- `README.md` — human-facing overview and repo layout.

When a task touches modelling choices, check the docs first; if you diverge from a
D-decision, say so explicitly. Phase 1 provisionally locks D1/D4/D8/D10 (see
`01-phase1-design.md §1`); D2/D3/D5/D6/D7/D9 stay open.

## The Phase-1 gate (what the current work is actually deciding)
Does a **per-atom pooled** fingerprint separate true contacting atom pairs from decoys as
well as the **per-vertex** baseline? Greenlight if per-atom descriptor-separation ROC-AUC is
within ~0.02 of per-vertex (baseline ≈0.98). Do **not** build Phases 2–4 (graph embeddings,
retraining, aligner hardening, ligands) until this gate is met. A small Phase-1 drop is not
failure — the intended gain is Phase 2.

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
