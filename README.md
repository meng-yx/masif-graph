# MaSIF-graph

Atom-graph-enhanced surface fingerprints for protein–protein (and protein–ligand neosurface) search.

A ground-up redesign of MaSIF-PPI-search in which the fundamental unit is the **surface heavy atom** rather than the surface mesh vertex. Each surface atom carries a fused embedding from (a) the classical MaSIF learned surface fingerprint, (b) a chemical/bond-graph embedding, and (c) a rotation-invariant local-geometry embedding. Binder discovery moves from per-patch local docking to **global** protein–protein pose optimization over complementary atom pairs.

> **Central goal — close the holo→apo gap.** MaSIF's learned surface descriptor is implicitly tuned to **bound-state (holo, crystal) sidechain rotamers**, and degrades on **apo / AF2 / unbound** conformations — producing false positives and negatives *despite* a high holo-benchmark AUC. The atom graph exists chiefly to encode **connectivity and bond rotatability** (how sidechain atoms can move), so the representation becomes **robust to sidechain conformation**. **Improving holo→apo generalization is the project's north-star success criterion** — note this means the current holo-only validation set cannot, on its own, demonstrate the benefit (see [`docs/03-phase2-design.md`](docs/03-phase2-design.md)).

> **Status:** Phase 1 complete (CONDITIONAL GO — see [`docs/02-phase1-results.md`](docs/02-phase1-results.md)); Phase 2 in design. No production pipeline yet. See the design docs before writing code.

## Documentation
- [`docs/00-context-and-goals.md`](docs/00-context-and-goals.md) — north-star: motivation, hypothesis, reference-system spec, **key design decisions (D1–D10)**, phasing, evaluation, risks.
- [`docs/01-phase1-design.md`](docs/01-phase1-design.md) — Phase 1: per-atom reframing + the go/no-go **pooling feasibility probe**.
- [`docs/02-phase1-results.md`](docs/02-phase1-results.md) — Phase 1 results & CONDITIONAL GO (mean pooling; ~0.03–0.05 holo pooling cost).
- [`docs/03-phase2-design.md`](docs/03-phase2-design.md) — Phase 2: **holo→apo robustness** via a heterogeneous atom graph (connectivity + bond rotatability); re-targeted gate.

## Relationship to the reference
This is a **rewrite**, not a fork. The legacy `masif-neosurf` code (re-cloned at `../masif-neosurf-af2`) is used as a **reference/template** and, during Phase 1, as an **executable tool** to generate probe inputs from raw PDBs. The **only artifact reused verbatim** is the PDB dataset lists in [`data/lists/`](data/lists/) (4,943 train / 959 test).

## Repository layout
```
data/lists/          reused PDB lists (the one reused artifact)
docs/                design docs (00 context, 01 phase-1, …)
src/masif_graph/     package (io, surface, pairs, align, metrics, experiments, …)
scripts/             CLI entry points; SLURM stubs (GPU-gated)
tests/
```

## Environments
- **`masif-graph`** (this repo, `environment.yml`) — the one conda env for all new code:
  Python 3.10 + PyTorch/PyG/e3nn + RDKit/Open3D/BioPython. CPU wheels for now.
- The reference TF1 stack has **no conda env** — it runs entirely from the Singularity image
  `masif-neosurf-af2/masif-neosurf_v0.1.sif`, used only to preprocess PDBs and compute
  reference descriptors for the Phase-1 probe.

```bash
conda env create -f environment.yml
conda activate masif-graph
pip install -e .
```

## Current focus
The Phase-1 **pooling feasibility probe**: does a per-atom pooled fingerprint separate true contacting atom pairs from decoys as well as the per-vertex baseline? This gates the rest of the project — see [`docs/01-phase1-design.md`](docs/01-phase1-design.md).

## License
TODO — recommend Apache-2.0 to match the MaSIF lineage (`cp ../masif-neosurf-af2/LICENSE .`).
