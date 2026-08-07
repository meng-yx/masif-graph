# Phase 6 Workstream C — autonomous build log (through C(c))

Autonomy: no per-step approval; budget **CHF 100**; de-risk on 1 example before scaling; honest stop if
dead-end. Datasets locked (D-C1 PDBbind refined ~5.3k `data/pdbbind/`; PPI>=3k; MolGlueDB 114 benchmark).

## Design recap (locked with user)
- Unified 26-D atom features (protein+ligand) — `src/masif_graph/p6/atoms.py` DONE.
- Ligand-modified surface (MSMS w/ ligand) so ligand surface-atoms get embeddings like protein.
- Objective: shared encoder + shared T; positive pairs = PPI (protein-surf<->protein-surf) AND
  protein-ligand (protein-pocket-surf <-> ligand-surf-atoms, contact). Transfer via unified atoms.
- Mixture train + mixture held-out val; do-no-harm on Phase-5 PPI gate; neosurface benchmark.
- Leakage: protein seq-cluster + ligand scaffold.
- **Dim change 14->26 => full retrain (VICReg+retrieval) required; existing PPI npz must be RE-featurized to 26-D.**

## RESUME STATE
- Phase: C(a).2 — de-risk .sif ligand-modified surface on 1 PDBbind complex (5hls).
- Spend: ~CHF 0 (this workstream). Running jobs: none.
- Next: if ligand surface builds -> C(a).3 graph builder; else diagnose/fallback.

## C(a).2 — PIVOT (2026-08-06): .sif ligand-surface too buggy -> ligand-atoms-as-nodes
The masif-neosurf `score_binder` branch ligand pipeline is WIP/buggy. Fixed 6 integration issues in
`01-pdb_extract_and_triangulate.py` / `ligand_utils.py` (arg name sdf_template->template_ligand; load sdf
into a Mol; bind /scratch into the .sif; drop patched-mol2; AddHs) and STILL hit the `heavy<total`
(protonation) invariant. Chasing further = rabbit hole + fragile at 5k-complex scale.
**DECISION: Path B — protein surface via the normal reliable pipeline; ligand heavy atoms added as GRAPH
NODES (unified 26-D features), ligand bonds as covalent edges, vertex-atom edges protein-pocket-vertex ->
ligand-atom. Encoder emits embeddings for protein surface atoms + ligand atoms.** Training pair = protein-
pocket-surface <-> ligand-atoms (contact); deployment neosurface query = protein-interface-surface +
ligand-atom embeddings. Gives up ligand-derived surface vertices (shape) but keeps the atom-level
protein<->ligand complementarity (the core goal). Robust + self-contained.
## RESUME STATE: C(a).3 Path B — build protein surface for a PDBbind complex (normal .sif) + inject ligand atoms.

## Agent launch (2026-08-07) — headless conductor submitted
Handed off to an autonomous headless Claude agent (conductor) on Jed SLURM to build Workstream C
through C(c). Artifacts: `PHASE6C_HANDOFF.md` (brief), `scripts/phase6C_agent.sbatch` (supervisor
loop, qos=serial 8c/32G/48h, model=claude-sonnet-5, sentinel `logs/PHASE6C_DONE`),
`docs/18-phase6C-user-comment.md` (async steering), `docs/19-phase6C-results.md` (to be written).
Smoke test (job 65979817, qos=debug) PASSED on compute node jst368: internet OK, singularity OK,
`claude -p` returned sentinel rc=0.

### RESUME STATE (for the headless agent — keep this current)
- Phase: C(a).3 — Path B ligand-aware graph builder. Build protein surface via normal `.sif`
  (`scripts/p6_protein_surface_one.sh`, validated 5hls) + inject ligand heavy atoms as graph
  nodes (unified 26-D `atoms.ligand_features`), covalent + vertex<->atom edges. Validate ONE
  PDBbind complex end-to-end (read the output) before scaling.
- Ligand path = **Path B (LOCKED, committed HEAD 410238d)**: ligand atoms as GRAPH NODES, NOT a
  computed ligand surface. Do NOT re-enter the buggy `.sif extract_ligand` chemistry path.
- Spend: ~CHF 0 (workstream C). Running child jobs: none.
- Next after C(a).3: C(b) re-featurize ~3k PPI to 26-D + preprocess PDBbind refined ~5.3k (Jed
  array children) -> cluster-clean split -> C(c) retrain on Kuma GPU -> 3-axis eval -> results doc
  -> touch logs/PHASE6C_DONE.
