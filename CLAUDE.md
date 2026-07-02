# CLAUDE.md — MaSIF-graph

Atom-graph-enhanced surface fingerprints for protein–protein / neosurface search. A
ground-up rewrite of MaSIF-PPI-search whose fundamental unit is the **surface heavy atom**,
not the mesh vertex. **Status: early design / Phase 1 — no production pipeline yet.**

## Read before writing code
- `docs/00-context-and-goals.md` — north star: hypothesis, design decisions D1–D10, phasing.
- `docs/01-phase1-design.md` — current work: the go/no-go **pooling feasibility probe**.
- `README.md` — human-facing overview and repo layout.

Design is ahead of code here — when a task touches modelling choices, check the docs
before improvising, and if you diverge from a D-decision, say so explicitly.

## Skills (auto-available in this repo; see `.claude/skills/`)
- **`ml-research-guardrails`** — invoke continuously during any training / data-splitting /
  evaluation / result-reporting. The ethos: *try to break your own good news before you
  believe it; a crash is cheap, a confident wrong result is expensive.* Leakage-checks,
  shuffled-label controls, and honest stop conditions live here.
- **`connect-to-kuma`** — how to reach the Kuma GPU cluster from Jed to run SLURM jobs.

## Compute
- You run on **Jed** (CPU-only login node, has internet). GPUs live on the **separate**
  **Kuma** cluster — you must `ssh` in to use them (see the `connect-to-kuma` skill).
- Stage code/data on shared `/home` or `/work` (`/work/upthomae/Meng`). `/scratch` is
  **not** shared between clusters.
- SLURM account `upthomae` (budget-capped). **Do not launch GPU training or large data
  transfers without explicit human go-ahead** — Phase 1 is CPU-feasible by design; GPU
  steps are gated.

## Environments
Two conda envs (they conflict): **`masif-graph`** (this repo — PyTorch/PyG/e3nn/RDKit,
`environment.yml`) and **`masif-neosurf-ref`** (the reference `../masif-neosurf-af2` stack,
used only to preprocess PDBs for the Phase-1 probe).

## Working norms
- Keep an append-only progress log for long/unattended work (the guardrails skill explains).
- The only artifact reused verbatim from the legacy code is the PDB lists in `data/lists/`.
- This is a git repo; don't commit unless asked.
