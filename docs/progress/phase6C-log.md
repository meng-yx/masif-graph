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

---

## 1. Conductor session start (2026-08-07) — inventory + a blocking data discovery

Read `PHASE6C_HANDOFF.md`, this log, `docs/18-phase6C-user-comment.md`, `docs/16 §5`,
`p6/atoms.py`, `p4/{precompute,dataset,encoder,objective,train_retrieval}.py`,
`graph/{hetero,build}.py`, `p5/retrieval_bench.py`, `io/reference.py`.

### 1.1 DISCOVERY (blocking, changes the plan): the reference surface tree was WIPED
The /scratch 30-day cleanup ate the reference artifacts the PPI corpus was built from:

| artifact | present | needed for the 4,872-complex PPI set |
|---|---|---|
| `01-benchmark_surfaces/*.ply` | 1,171 files (~585 chain pairs) | 9,744 |
| `04b-precomputation_12A/precomputation/<cid>/` | 589 dirs | 4,872 |
| `01-benchmark_pdbs/*.pdb` (chain PDBs) | 1,171 | 9,744 |
| `descriptors/sc05/all_feat/<cid>/` | 590 | 4,872 |

Only **16 / 4,872** Phase-4 training complexes still have both `.ply`s. So the handoff's
"re-featurize the PPI npz to 26-D" cannot just re-run `p4.precompute` — its inputs are gone.
What *did* survive (all on `/work`, which is not on the cleanup timer):
- `/work/upthomae/Meng/phase4/stageA_full_npz/` — **14,614 files = 4,871 complexes** of Phase-4
  14-D hetero-graph npz (atom_feat, vert_feat, all 3 edge types, surf_node_idx, desc, coord, keys).
- `/work/upthomae/Meng/phase5/npz/` — Phase-5 eval npz, **holo 301 / af3 284** of the 304 ids
  (`logs/phase5/eval_clusterclean.txt`; the 287-clean list is `eval_sc304_clean_vs_enc.txt`).
- **Chain PDBs for the Phase-5 eval set DID survive** (301 holo + 284 af3 in `01-benchmark_pdbs`).

### 1.2 The cheap way out (decision D-C3): PATCH `atom_feat` in place, don't rebuild surfaces
The 14 -> 26-D change touches **only the atom node features**. Everything else in the npz (vertex
features, mesh edges, vertex-atom edges, contacts) is dimension-independent. And 23 of the 26 dims
are recoverable with no surface pipeline at all:

| 26-D dim | source |
|---|---|
| [0:10] element 1-hot | 14-D [0:6] (C,N,O,S,P,other) re-indexed — exact for protein |
| [11] backbone, [12] aromatic, [13] degree, [14] is_surface, [22] flex, [23:26] elem-chem | 14-D [6],[7],[8],[9],[10],[11:14] — exact |
| [10] is_ligand | 0 |
| [15] in_ring | `protein_features` defines it == aromatic — exact |
| [16:19] hybridization | rule from (aromatic, element) — exact |
| **[19] donor, [20] acceptor, [21] charge** | need **atom name + residue name** -> the chain PDB |

So the *only* missing input is the chain PDB (for names/resnames). That is recoverable **without
MSMS/APBS/precompute**: download `https://files.rcsb.org/download/{PDB}.pdb` and re-extract the
chain the way the reference did. Cost is a download + a parse per chain instead of a ~6-min `.sif`
surface build. **Correctness is verifiable**: rebuild the 14-D columns from the re-derived atom
table and require an exact match against the stored `atom_feat`; any complex that fails is dropped,
so a mis-ordered atom table can never silently corrupt training.

Divergence from the handoff's step order (it said "re-featurize the ~3,000 PPI complexes"): same
end state, but patching the surviving npz avoids ~500 core-hours *and* the risk that the `.sif`
pipeline no longer reproduces the 2026-07 run. PDBbind still needs the full `.sif` surface build
(no npz exists for it at all).

### 1.3 Cost model (CHF 100 budget)
Anchor: 8 cores x 24 h ~ CHF 1 -> **1 core-hour ~ CHF 0.005**.
- PPI 26-D patch (4,871 complexes): ~2 s/chain parse + download ~ 6 core-h ~ **CHF 0.03**.
- PDBbind refined 5,316 x `.sif` (01 + 04b only; skip 04a-site and the descriptor net — the
  learned encoder never reads descriptors) ~ 5 min/complex ~ 440 core-h ~ **CHF 2.5**.
- Kuma GPU: VICReg + retrieval, a few CHF per run (Phase-4/5 anchor).
Peak /scratch disk for the 04b precompute is ~40 MB/complex -> each array task deletes a complex's
precompute directory as soon as its npz is written.
