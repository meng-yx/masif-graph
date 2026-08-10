# Phase 7 — build log

Design: `docs/20-phase7-design.md` (D7-1 … D7-8). Motivated by the Phase-6C diagnostic that the
combined model cannot fit the ligand axis on its **own training data** (P–L train top-5 0.095 vs
0.429 for PPI) — a representation/capacity failure, not a generalisation failure.

## S0 — ligand-alone MSMS surface, end-to-end (DONE, 5hls)

`scripts/p7_lig_surface.py`, run inside the `.sif`. Confirms the question Phase 6C never actually
answered: **MSMS processes ligands fine**. What was broken was `extract_ligand`, whose only output
`(rdmol, mol2)` we already have from the PDBbind SDF.

5hls ligand: 42 atoms (26 heavy + 16 H) → **all 42 emitted to xyzrn** (the halogen fix working; the
reference table has no F and would have dropped it) → 990 raw MSMS vertices → `fix_mesh(1.0)` →
**153 vertices / 302 faces**. `vertices_unmatched_to_atom = 0`, so the atom-name join that both
ligand channel helpers depend on is exact.

Three defects found and fixed along the way:
1. **Halogens were being silently dropped.** The reference `radii` table has only N/O/C/H/S/P and
   `output_pdb_as_xyzrn` skips any atom whose type is missing. Our own xyzrn emitter uses Bondi
   radii for F/Cl/Br/I/B/Se/Si/As and takes the element from RDKit rather than `atom_name[0]`
   (which types `CL1` as carbon).
2. **`pdb2pqr` rejects the `am` bond type outright** — the exact failure `amide_to_single_bond()`
   exists for. Applying it makes the reference APBS route work; a self-written PQR + direct APBS is
   the D7-5 fallback.
3. **`read_msms` is locale-dependent.** It reads MSMS output with the default encoding, and the MSMS
   banner contains byte `0x9a`; it therefore succeeds under a UTF-8 locale and raises
   `UnicodeDecodeError` under POSIX/C. That is an environment-dependent failure that would have hit
   an unpredictable fraction of a 5,000-job array. Replaced with an explicit-decode reader.

## S1 — comparability QC, 50 random training ligands (DONE)

**50/50 built.** Median 154 vertices (min 55, max 391), **6.18 vertices per heavy atom** — against
~4.7 vertices per surface atom on the protein side, i.e. the same order, as it must be given both use
`mesh_res = 1.0`.

Channel distributions, ligand vertices (n = 7,585) vs protein vertices (n = 302,658), both after the
*identical* transforms:

| channel | ligand mean / std | protein mean / std | reading |
|---|---|---|---|
| `hbond` | −0.053 / 0.258 | −0.034 / 0.262 | **matches** |
| `charge` | −0.046 / 0.281 | −0.036 / 0.289 | **matches** |
| `si` | +0.453 / 0.423 | +0.061 / 0.574 | ligand far more convex |
| `hphob` | +0.273 / 0.647 | −0.334 / 0.596 | ligand far more hydrophobic |

Two channels match closely; two show large offsets. **The offsets are chemistry, not calibration.**
A small molecule really is mostly convex (84.7% of ligand vertices have si > 0 vs 54.3% for protein;
only 5.4% are concave vs 29.1%) and drug-like molecules really are lipophilic. Complementarity *is*
convex-meets-concave, so this is the signal, and before Phase 7 the ligand side had no si at all.
Neither offset creates a new shortcut: `is_ligand` already tags the type explicitly.

**Not degenerate meshes.** A blob would put ~everything in the top si bin; the ligand histogram
spans all ten bins (0.3 / 1.3 / 2.2 / 4.0 / 7.5 / 11.1 / 15.5 / 17.4 / 16.7 / 24.0 %) and retains
5.4% concave vertices, so MSMS is finding real grooves.

**Risk R2 quantified.** Curvature clamp rate (`H²−K < 0`, visible as |si| > 0.99): **ligand 17.5% vs
protein 12.9%** — higher, same order, not a blocker. Reported rather than assumed away.

**D7-5 RESOLVED → self-PQR for the whole ligand corpus.** `pdb2pqr --ligand` succeeded on only
**15/50** ligands even with the amide fix, so using it would split the corpus across two charge
scales. Where both ran, they correlate at **0.927**, so the uniform choice costs little. One path,
one scale.

## S2 — full ligand-surface pass (DONE)

Array 65990015, 117 tasks. **5,239 / 5,240 ligand surfaces built** (one `AtomValenceException`),
then attached to the ligand npz by `p7/lig_vertices.py` — **5,239 attached**, ~CHF 2.

The attach is a controlled A/B: only the vertex side is added. Atom features, bond edges, the
readout index and the contact arrays are carried through untouched, and the protein npz and contacts
are **symlinked** from Phase 6C rather than copied, so the two corpora are bit-identical everywhere
except the ligand's surface. Vertex-atom edges are built against the npz's stored heavy-atom
coordinates, which makes them immune to the surface molecule's atom ordering (the surface is built
from a mol with explicit hydrogens whose order need not match).

Spot check (5oh3): 10 heavy atoms → 72 vertices, 420 directed mesh edges, 538 vertex-atom edges,
all 10 atoms own a vertex, contacts unchanged at 163 pairs, encoder forward+backward finite.

## S3 — neosurface surfaces (DONE, 42/42)

Job 65990157. For each of the 14 ternary systems: the drug's own surface (14) plus the **composite
protein+drug surface for both subunits** (28). All `ok`.

6QTL_A: 17,020 raw MSMS vertices of which **382 are drug-owned** → 2,401 after `fix_mesh`, curvature
clamp 12.9% (protein-like). So the drug really does contribute a face to one continuous surface,
which is the thing Phase 6C's representation could not express at all.

Composite graphs (`p7/composite_graph.py`, 28/28): protein and drug share one atom set and one
surface; readout = every atom owning a composite vertex. Query-side only (D7-4) — a composite entry
in the DB would be a self-match.

`p6/neosurf_bench.py` gained two extra arms so the two ways a drug can matter are separated:
* `composite` — joint surface **+** the drug's own embeddings;
* `composite_noligand` — joint surface, protein rows only → isolates *the drug reshaping the protein
  surface* from *the drug contributing embeddings*.

## S4 — training (RUNNING): 4 Kuma jobs

| job | tag | data | seed | purpose |
|---|---|---|---|---|
| 4065100 | `p7comb_s0` | Phase-7 (ligand surfaces) | 0 | the deliverable |
| 4065101 | `p7comb_s1` | Phase-7 | 1 | seed spread |
| 4065102 | `p6comb_s1` | Phase-6C (no ligand surface) | 1 | error bar on the OLD result |
| 4065103 | `ppionly_s1` | PPI only | 1 | error bar on the do-no-harm control |

Phase-7 staging symlinks the Phase-6C PPI npz, eval npz and split, so the p7comb-vs-p6comb contrast
is a clean single-variable A/B. `ppionly` seed 0 and `p6comb` seed 0 already exist from Phase 6C, so
each of the three conditions ends up with **2 seeds** (D7-6).

## S5 — AF3-apo on the ligand axis (RUNNING)

300/300 AF3 input JSONs written for the held-out `val_pl` proteins (median 286 residues, p90 653).
MSA probe: **~750 s/chain on 16 cores** → ~1,000 core-hours for all 300, ≈ **CHF 5**. Full MSA array
65990205 running.

R5 (frame mismatch) closed **by reuse, not new code**: `masif_graph.af3.build_pdb` already relabels
an AF3 model to holo residue numbering *and* superposes it into the holo frame by CA correspondence,
so the crystal ligand pose stays valid against the predicted protein.

`p7/pl_af3.py` recomputes contacts on the AF3 structure against the same crystal ligand rather than
reusing holo rows — the AF3 surface has its own atom rows, and the change in the contact set is part
of what is being measured (`contact_ratio_af3_over_holo`).

## S4 — a Stage-A divergence on ONE seed (found, diagnosed, contained)

Watching the Stage-A diagnostics across the four runs:

| run | ep1 gnorm max | ep15 gnorm max | ep15 loss | ep15 val SC AUC |
|---|---|---|---|---|
| `p7comb_s0` | 11.6 | 5.2 | 7.58 | 0.492 |
| **`p7comb_s1`** | **1,114.7** | **1,558,297.9** | **36.82** | 0.533 |
| `p6comb_s1` | 18.5 | 3.8 | 7.58 | 0.600 |

Seed 1 of the **Phase-7** arm diverged; seed 0 on the same data and seed 1 on the Phase-6C data both
stayed clean. So it is a data x seed interaction, not either alone.

**Data checked before blaming the optimiser.** All 5,239 Phase-7 ligand npz: every vertex channel
inside [-1, 1], zero non-finite values, no anomalous `vv`/`va` edge lengths. The new features are
clean, so this is optimisation instability, not a feature pathology.

**Contained by checkpoint selection**: Stage A saves on best held-out AUC, and seed 1's best was
**epoch 5 (0.670)** — before the blow-up — so Stage B initialised from a sane point.

Recorded as a finding rather than papered over: **the Phase-4 Stage-A recipe is not robust on the
combined corpus at every seed.** A single-seed run would have hidden this entirely, which is the
argument for D7-6. If seed 1's Stage B turns out anomalous, a third seed gets added rather than
averaging a broken run into the Phase-7 arm.

Note also that Stage-A held-out AUC *declines* from ~0.70 at epoch 5 to 0.49-0.60 at epoch 15 on the
combined corpus for every arm, with normalised `z_std` at ~0.001. That is the Phase-6C DC-offset
picture again (docs/progress/phase6C-log.md §9), and Stage B is where centering fixes it — but it
means Stage A is over-trained at 15 epochs for this corpus, worth revisiting if Phase 8 reuses it.

## S5 — AF3 MSA complete

**298 / 300** chains have a `_data.json` (0 `MSA_FAIL`; 2 chains produced no output). Inference
probe submitted to Kuma to measure per-chain GPU time before committing the full set.

## S5 — AF3-apo data COMPLETE, axis 4 validated

| step | outcome |
|---|---|
| MSA | **298 / 300** chains (0 `MSA_FAIL`; 2 produced no output) |
| inference | **298 / 298** models, **~60 s/chain** on an H100 |
| AF3 surfaces + contacts | **298 / 298** |
| usable for robustness (>=8 contacts in BOTH states) | **283 / 300** |

**The superposition is sound**, which is the thing that could silently have been wrong: contact ratio
AF3/holo has median **0.99** (p10 0.72, p90 1.41) and **zero** complexes where the AF3 model fails to
contact the crystal ligand at all. A broken frame would have produced ratios at zero across the board.

Cost reality check: inference came in at ~60 s/chain against a ~12 min/chain projection from the
Phase-5 anchor, so the whole S5 arm is **~CHF 8** rather than the ~CHF 39 originally budgeted. Probing
before committing was worth the ten minutes.

**Axis-4 harness validated on the random-init encoder** — all four role cells at chance
(top5 0.011-0.021 against a shuffled control of 0.014; median rank 143-152 of 283, chance ~141).

## Mid-training observation (epoch 8 of 32, NOT a result)

| run | train top1 | PPI top5 | P-L top5 |
|---|---|---|---|
| `p7comb_s1` (ligand surface) | 0.242 | 0.091 | **0.069** |
| `p6comb_s1` (no surface) | 0.363 | 0.500 | **0.052** |

The Phase-7 arm is ahead on the ligand axis and well behind on PPI at matched epoch. Recorded only
so the final numbers can be read against the trajectory — this is epoch 8 of 32 under a cosine
schedule, where most of the movement is still to come, and the Phase-7 arm is also the slower one
per epoch (518 s vs 485 s). No conclusion is drawn from it.

## Visual QC of the ligand surfaces (5 examples) — and one real defect found

`scripts/p7_export_pl_viz.py` + `scripts/p7_pl_viz_figs.py` (the Phase-7 analogue of
`p4_export_graph_viz.py` / `p4_graph_pymol.py`) render a per-complex QC sheet: the molecule, the
surface shaded by each of the four channels, the ligand surface inside the protein pocket surface,
and the actual training contacts. Sheets in `notebooks/figs/p7_ligsurf_*.png`.

Examples chosen to stress different failure modes: `1a99` (6 atoms — smallest), `4ghi` (Cl+F),
`5czm` (Br+Cl+P, 52 atoms), `5b25` (held-out, 7 rings), `3h8b` (71 atoms — largest).

**Four of five are clearly correct.** The mesh follows the molecular skeleton lobe-for-lobe, the
saddle between ring systems shows up as genuine concavity in `si`, halogens carry surface (the D7-2
radii fix working — `5czm` has Br and Cl on the surface, which the reference table would have
dropped), and the ligand interdigitates with the protein pocket (closest ligand-surface vertex to a
protein atom 0.76-0.98 A, with 313-451 contacts spread over the whole molecule).

**Risk R1 is confirmed for very small ligands.** `1a99` (6 atoms) yields a near-spherical blob:
`si` in [-0.16, +1.00] with essentially no concave vertices, and **20.6 A^2/atom against a corpus
median of 11.9**. A 1.5 A probe simply rolls over a 6-atom molecule.

Quantified corpus-wide (`logs/phase7/lig_degeneracy.npy`), concavity rises monotonically with size:

| ligand atoms | n | area/atom (A^2) | frac concave vertices |
|---|---|---|---|
| <8 | 67 | 15.0 | 0.039 |
| 8-12 | 412 | 13.4 | 0.066 |
| 12-16 | 521 | 12.8 | 0.091 |
| 16-20 | 576 | 12.4 | 0.117 |
| 20-30 | 1,855 | 11.9 | 0.143 |
| 30-50 | 1,551 | 11.5 | 0.183 |
| >=50 | 257 | 11.3 | 0.227 |

**45 of 5,239 ligands (0.9%)** have <2% concave vertices, i.e. are blob-like; all have <=13 atoms.
Listed in `logs/phase7/lig_surface_degenerate.txt`.

Flagging by the direct measure rather than by atom count, because atom count is a poor proxy: a
"<8 atoms" rule catches only 19 of the 45, while "<12 atoms" catches 44 but discards 479 complexes
(9.1%) that are fine. At 0.9% the effect is too small to change any headline, but the flag list
exists so the axis-2 numbers can be re-run with those complexes excluded as a sensitivity check.

## RESULTS — see `docs/21-phase7-results.md`

All 4 training runs and all 14 evaluation jobs complete. Headline: **the Phase-7 hypothesis is not
supported.** Train-set P-L retrieval (the primary gate) is unchanged at 0.111 +/- 0.007 vs
0.119 +/- 0.024, so the ligand's missing shape channel was **not** the capacity bottleneck. Held-out
P-L retrieval does improve (0.054 -> 0.084; 0.076 -> 0.111 scaffold-clean), i.e. better
generalisation without better capacity. Cost: PPI 0.644 -> 0.475, a -0.169 do-no-harm regression in
both seeds, with PPI *training* accuracy also down (0.448 -> 0.363), which points at capacity
competition rather than overfitting.

Two secondary results worth carrying forward:
* **The Phase-6C -0.041 do-no-harm gap is not real** — at 2 seeds it is -0.017 +/- 0.019, inside the
  spread. The user's insistence on seeds before believing it was right.
* **The composite-neosurface hypothesis is refuted**, not merely unsupported: composite 337 +/- 2 vs
  separate-surface 347 +/- 15 vs composite-without-drug 335 +/- 13, all below the ~298 chance line.
  That was the strongest remaining explanation for the axis-3 null and it is wrong.

New capability: axis 4 (ligand-axis holo->AF3-apo robustness) shows drops of only +0.007/+0.014
top-5, so conformational change is not the ligand axis's limiting factor either.

Spend ~CHF 22.
