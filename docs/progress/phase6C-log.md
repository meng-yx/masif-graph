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

## 2. C(a).3 — Path-B protein-ligand graph builder (DONE, validated on 5hls)

**Key structural decision:** a PDBbind complex is emitted as the **same artefact as a PPI complex** —
`{cid}__holo__p1.npz` (protein: atoms+vertices+3 edge types, 26-D atom feats), `{cid}__holo__p2.npz`
(ligand: atoms + covalent edges, `n_vert=0`, every heavy atom a readout node), `{cid}__contacts.npz`
(pos <=5.0 A / pos_sc <=4.0 A as (protein_surf_row, ligand_row)). Consequence: `p4.dataset.ComplexP4`,
the chain-level retrieval loss and the whole Phase-4 loop consume the mixture with **no new code
path** — the mixture is just a longer id list.

The two sides are encoded **independently**, exactly as the two chains of a PPI complex are. Putting
ligand atoms inside the protein graph (the naive reading of "ligand atoms as graph nodes") would let
protein vertices message into them, so a ligand embedding would already encode its own protein and
retrieval would be free — the metric would look spectacular and mean nothing.

Protein-side prep: PDBbind `_protein.pdb` carries 1-24 chains but the ligand touches few. Keep the
chains with a heavy atom within 6 A of the ligand, merge to one pseudo-chain `A`, renumber residues
sequentially. Cap 8,000 heavy atoms (MSMS cost). Profiled on 150 random refined ids: **148 usable**,
median 2,027 heavy atoms (p90 4,313), ligand median 25 heavy atoms.

Validation (5hls, read the actual output): protein 1,061 atoms / 3,039 verts / 648 surface atoms;
ligand 26 atoms / 58 directed bonds / 20 C, 3 N, 2 O, 1 F, 17 aromatic, 20 in-ring, 1 donor,
3 acceptors; 197 contacts over 40 protein + 26 ligand atoms. Encoder forward **and backward** finite
on both graphs (the 0-vertex ligand graph exercises every empty-edge branch).

Also fixed a latent crash in `p6/atoms.py::_elem_onehot` — it called `list.index` unguarded, so any
element outside the 9-element table (Se in MSE, metals) raised `ValueError`.

## 3. C(b).1 — preprocessing children (submitted 13:46)

| job | what | shape |
|---|---|---|
| 65982544 | PDBbind smoke (2x3) | ok; **87 s/complex**, far below the 6 min estimate |
| 65982545 | PPI 26-D refeat, 4,871 complexes | array 0-7 |
| 65982546 | Phase-5 eval refeat (holo) | **301/301, zero failures** |
| 65982573 | Phase-5 eval refeat (af3) | array 0-1 |
| 65982574 | PDBbind refined, 5,316 complexes | array 0-265%220, 20 ids/task |

`.sif` steps trimmed to **01 + 04b(ppi_search, p1 only)** — 04a/masif_site and the descriptor net are
never read by the learned encoder. Each task uses a private `TMPDIR` (computeMSMS/APBS temp files are
otherwise shared) and deletes the ~40 MB precompute dir as soon as the npz is written.

**Refeat failure analysis (the check earning its keep).** ~2.6% of chains fail, two causes:
(1) a race on the shared RCSB cache — two array tasks wrote the same `.part` file; fixed with a
per-pid temp name; (2) atom-count mismatches (e.g. 1PFF_A: 2,519 re-derived vs 2,511 stored). Traced
to *trailing* residues — the current RCSB entry has C-terminal atoms the 2026-07 snapshot lacked, i.e.
entry remediation, not a systematic per-residue-type bias. Those complexes are **dropped, not
patched**: the 14-D bit-exactness + surface-key checks reject them, which is the entire point of
having them. Losing ~5% of 4,871 still leaves far more than the ~3,000 the Workstream-B verdict asks for.

Cost so far: **~CHF 1** (estimates: pdbbind smoke 0.09, refeat 0.13+0.03, full pdbbind array ~2-6).

## 4. C(b).2 / C(c).1 — split + unified training code (written, debug-run green)

- `p6/split.py` — mmseqs2 @30% id over PPI chains + PDBbind chain sequences, plus RDKit Murcko
  scaffolds. Rule: the frozen Phase-5 287-clean eval set's protein clusters are forbidden in **both**
  corpora (a PDBbind target homologous to an eval chain leaks into the do-no-harm gate just as surely
  as a PPI one). The protein-ligand held-out is carved by connected components of "shares a protein
  cluster OR a ligand scaffold", with a documented fallback if the scaffold graph is degenerate, and
  the scaffold-unseen subset is always reported separately.
- `p6/mixed_bench.py` — chain retrieval over a mixed DB, `median_i max_j z^T T z` + mandatory
  centering, reported per type and per query role (pocket->ligand vs ligand->pocket), with a
  same-type decoy pool (a protein beating a *ligand* decoy is not binder discrimination) and a
  shuffled control.
- `p6/train_unified.py` — Phase-4 recipe (chain InfoNCE + atom InfoNCE + VICReg, freeze-tau 0.1,
  T-wd 1e-3, lr 5e-4 cosine, d64/32/L4, centering) on the combined corpus. Batches are built to hold
  **both** types (`--pl-frac`), since the in-batch chains ARE the hard decoy pool.
  Divergences logged: `--max-patch 128` (Phase 4 was uncapped; the chain score matrix is
  O(N^2*n_a*n_b) and PPI dense patches are 100s of atoms vs a ligand's ~25), and model selection on
  mixed held-out MRR (there is no AF3 state for protein-ligand complexes).

CPU debug run (24 PPI + 39 PL train, 20+30 val): loss 7.99 -> 7.59, in-batch train top-1 0.05 ->
0.16, z_std flat 0.163 (no collapse), |T| flat, gradients finite. **The untrained network sits at
chance on every group** (pl top5 0.15 vs chance 0.167; ppi 0.275 vs 0.25) — the eval is not
accidentally leaking.

## 5. Kuma timing probe (job 4026331, qos=debug, CHF 0.43) + a CUDA bug it caught

800-complex, 2-epoch probe of both stages on an H100.
- **Stage A**: 2m02s for 2 epochs = ~50 s/epoch at 800 complexes -> **~16 complexes/s**; median step
  21 ms. Learning: held-out SC learned 0.493 -> 0.772 in two epochs (frozen ceiling 0.942).
- **Stage B**: crashed — `mixed_bench.build_patches` called `.numpy()` on a CUDA tensor. Fixed there
  and, pre-emptively, everywhere the same pattern existed (`train_unified.iface_idx`,
  `neosurf_bench` cap/seg/index/scatter buffers): index tensors and scatter buffers now inherit the
  device of the tensor they touch. The standalone benches run on CPU so this only bit the GPU path.
  **This is exactly why the probe existed** — the bug would otherwise have surfaced hours into a
  12 h run.

Cost anchors measured on Kuma: **CHF ~0.52 per GPU-hour**; a 12 h pipeline reservation estimates
CHF 6.21. Two full runs (combined + PPI-only control) are therefore ~CHF 10, comfortably inside 100.

**PPI exposure is matched between the two runs by construction.** `make_batches` sizes an epoch by
the larger pool, so with `--pl-frac 0.5` the combined run does 276 batches x 16 PPI = 4,416 PPI
visits/epoch, and the PPI-only control does 138 batches x 32 = 4,416. The combined run costs ~2x the
wall clock per epoch because it additionally does 4,416 protein-ligand visits.

## 6. Random-init control gate (job 65982979) — the eval harness is sound

Ran all three axes with an **untrained** 26-D encoder, both as a pipeline de-risk and as the chance
line for the results doc.

**The important result is the frozen column.** On the re-featurized 26-D eval npz the frozen MaSIF
baseline reproduces the Phase-5 published numbers *exactly*:

| cell | this run (26-D npz) | Phase-5 `gate_fullclean_pos.json` |
|---|---|---|
| HH_frozen top5 / medRank | 0.084 / 110 | 0.084 / 110 |
| AA_frozen top5 / medRank | 0.061 / 128 | 0.061 / 128 |
| n / DB | 269 / 538 | 269 / 538 |

Identical to three decimals on the same denominators. So the `atom_feat` patch left the descriptors,
interface definitions and AF3 join untouched, and the do-no-harm gate is directly comparable to the
Phase-5 bar: **HH_learned top5 = 0.630, AA_learned top5 = 0.639**.

The untrained learned encoder sits at the shuffled-control line (top5 0.009-0.015 vs shuffled 0.011),
and on the neosurface axis it scores exactly chance (medRank 44/108, ligand effect 14 better /
14 worse). Both controls behave.

Also fixed: `p6C_gate.sbatch` died 8 s in (job 65982977) because `set -u` plus conda's gromacs
deactivate hook dereferences unset variables. The gate now calls the env's interpreter directly.

## 7. Final corpus + split (2026-08-07 ~15:20)

PDBbind refined array finished: **5,240 / 5,316 complexes built** (72 skipped by the >8,000-heavy-atom
cap or no chain within 6 A of the ligand; 3 `.sif` 04 failures) = **99.7% of non-skipped**. PPI 26-D
re-featurisation: **4,711 / 4,871**. Phase-5 eval: 301 holo + 284 AF3. Neosurface: 14/14 systems.

Final cluster-clean split (`logs/phase6C/split/`, report in `split_report.json`):

| set | n |
|---|---|
| train_ppi | 4,418 |
| train_pl | 4,546 |
| val_pl (held out) | 300 — of which **198 scaffold-unseen** |
| val_ppi_stageA / stageB | 80 / 197 |
| eval_ppi (frozen Phase-5 287-clean) | 287 |

`verify` (against the ACTUAL train ids): eval-into-train **0**, val_pl-protein-into-train **0**,
PPI-holdout-into-train **0**. **394 PDBbind complexes were dropped for being homologous to the PPI
eval set** — the cross-corpus filter earning its keep; without it those would have leaked straight
into the do-no-harm gate.

The scaffold graph is degenerate (largest component 61% of the corpus once scaffold edges chain
through protein edges), so the split runs on protein clusters and scaffold overlap is reported
rather than pretended away: 102/300 val_pl share a scaffold with train, 198 do not, and the
scaffold-unseen subset is evaluated separately.

## 8. Kuma training children (submitted 15:2x)

| job | mode | corpus | epochs A/B | selection |
|---|---|---|---|---|
| 4026517 | `combined` | 4,418 PPI + 4,546 P-L | 15 / 32 | mixed held-out MRR |
| 4026518 | `ppionly` | 4,418 PPI | 30 / 32 | PPI held-out MRR |
| 4026519 | `plonly` | 4,546 P-L | 30 / 32 | P-L held-out MRR |

**Epoch counts were chosen so per-type exposure matches, not so epoch counts match.** With
`--pl-frac 0.5` the combined run does 284 batches x 16 = 4,544 PPI visits and 4,544 P-L visits per
Stage-B epoch; `ppionly` does 138 x 32 = 4,416 PPI and `plonly` 142 x 32 = 4,544 P-L. So at equal
Stage-B epochs all three models have seen each data type an equal number of times, and any
difference is attributable to the *other* corpus being present rather than to more gradient steps.
Stage A epochs are halved for `combined` for the same reason (it streams 2x the complexes/epoch).

`plonly` is the control that tests Workstream C's actual thesis: if `combined` beats it on the
ligand axis, PPI complementarity transferred through the shared 26-D atom space. `ppionly` is the
control for the do-no-harm claim. Checkpoint selection uses only training-pool holdouts — the frozen
287-complex gate is never used for selection.

Estimated GPU spend: ~CHF 8-9 for all three (measured anchor CHF 0.52/GPU-hour). Jed spend to date
~CHF 3. Running total well under the CHF 100 budget.

## 9. A collapse scare on the combined run — diagnosed, and one real bug found

Watching the Stage-A diagnostic, `combined` (4026517) showed `z_std` falling 0.0083 -> 0.0031 ->
0.0023 -> 0.0018 while the controls stayed put (`ppionly` ~0.014, `plonly` rising to 0.065). That is
numerically the Phase-4 collapse signature (docs/10: "z_std -> 0.003 from ep1").

**It is not collapse.** `p4.train` measures `z_std` on the **L2-normalised** chain-1 embeddings,
and VICReg constrains the **raw** ones. If the raw embeddings acquire a large common mean, raw
per-dim std can sit at the VICReg floor while the normalised vectors all point the same way — the
Phase-4 §21 DC-offset phenomenon, which is exactly what `--center` exists to undo. The decisive
evidence is the epoch-5 eval: combined **learned SC AUC = 0.702** against a shuffled control of
0.501, i.e. the encoder is learning. (It is behind `ppionly` 0.824 @ep5 and `plonly` 0.925 @ep10 —
worth reporting, not worth panicking over, since Stage B centres and does the real work.)

**But the scare surfaced a genuine design hole in Stage B, now fixed.** `train_unified` applied
VICReg to the pooled batch: `vicreg_terms(torch.cat(raws, 0))` over protein *and* ligand atoms
together. Protein and ligand atoms are trivially separable — the `is_ligand` feature alone does it —
so a pooled variance term is fully satisfied by **two tightly-clustered point masses sitting far
apart**. That is precisely the collapse VICReg is there to prevent, and the pooled statistic cannot
see it. VICReg is now computed **per type and averaged**, which forces spread *within* each type.
All three runs were still in Stage A when the fix landed in `/work/.../phase6C/src`, and Stage B is
a separate process, so all three pick it up uniformly.

Decision: let Stage A finish rather than restart. The diagnostic that would have changed this
decision (learned AUC at chance) came back negative.

## 10. User steering (2026-08-07, answered inline in docs/18)

1. **Array concurrency** — the `%220`/`%250` caps were unnecessary; the cluster handles ~500.
   Future arrays go uncapped unless the scheduler actually pushes back.
2. **Ligands have no apo state** — at deployment a ligand-bearing structure always comes from
   experiment, so predicting an apo ligand conformation is meaningless. Recorded as a decision:
   *if* a workflow ever needs a holo/apo pair per entry, the ligand is duplicated unchanged.

   **This workstream does not need it.** C training is holo-only for both corpus types (Stage A and
   Stage B read only `__holo__` npz); the positive pair is the two sides of one complex, never a
   holo-vs-apo pair. AF3/apo appears only in the axis-1 Phase-5 gate, which is protein-protein only.
   Axis-3 ligand geometry is the **experimental bound pose** (RCSB ModelServer instance endpoint),
   matching the deployment assumption exactly.

   Consequence for later phases: the ligand-axis robustness test (AF3-apo protein + experimental
   ligand) is the natural Phase-7 follow-up and now has its rule fixed in advance — protein varies,
   ligand stays at its experimental pose. Out of scope for C per the handoff.

## 11. Resume (session 2) — reattached, no double-submission

`squeue` on Jed: only the agent job. On Kuma, all three training children were still RUNNING
(4026517/18/19, elapsed 1:10) — reattached rather than resubmitted. `ret_plonly_best.pt` already
written (first checkpoint saved at Stage-B epoch 4). Random-init control gate re-run (65983164)
completed and produced the scaffold-deduplicated variant.
