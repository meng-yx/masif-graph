# Phase 7 — Give the ligand a shape channel, and test the ligand axis honestly

> Design-ahead-of-code. Follows `docs/16-phase6-design.md` §5 and `docs/19-phase6C-results.md`.
> Phase 6 Workstream C delivered a ligand-capable unified encoder and three honest axes. Two of
> the three came back weak. Phase 7 attacks the single most likely cause, with the controls needed
> to know whether it *was* the cause.

## 0. One-paragraph summary
Phase 6C represented ligands as **atoms only** (Path B): no surface, no mesh, no shape. Protein–ligand
retrieval came out barely above chance, and the neosurface benchmark came out null. The decisive
diagnostic is that the combined model **cannot fit the ligand axis even on its own training data**
(train top-5 0.095 vs 0.429 for PPI) — a capacity/representation failure, not a generalization
failure. Phase 7 gives the ligand a real molecular surface with the same four MaSIF channels the
protein carries, produced by the **same MSMS pipeline at the same parameters**, so the two sides of a
protein–ligand pair are finally the same kind of object. It also (a) builds the **composite
protein+ligand surface** the neosurface benchmark actually needs, (b) re-runs every claim at **≥2
seeds**, and (c) generates **AF3-apo** models for the held-out protein–ligand proteins so the
project's north star — conformational robustness — is finally tested on the ligand axis too.

## 1. What Phase 6C established (the starting point)

| axis | result | reading |
|---|---|---|
| 1 — do-no-harm PPI gate | combined 0.610 vs PPI-only control 0.651, Phase-5 14-D 0.630 | 26-D features *helped* (+0.021); adding ligands *cost* (−0.041). **One seed each.** |
| 2 — mixed held-out | PPI 0.576 top-5 (medR 1.5); **P–L 0.040 top-5 (medR 76 / chance 138)** | above chance, far below useful |
| 3 — neosurface (28 cases) | top-5 0.036, medRank 267/596 ≈ chance; ligand helps 17 / hurts 11 | null |

**The diagnostic that motivates Phase 7** (`logs/phase6C/results/mixedtrain_*.json`, identical set
sizes train vs held-out):

| model | PPI train → held-out | P–L train → held-out |
|---|---|---|
| combined | 0.429 → 0.576 | **0.095 (medR 45) → 0.040 (medR 76)** |
| ppionly | 0.464 → 0.602 | 0.029 → 0.021 |
| plonly | 0.028 → 0.028 | 0.041 → 0.036 |

The model cannot memorize the ligand axis. Whatever else is true, the current ligand representation
does not have the capacity to express protein–ligand complementarity. The ligand side carries *zero*
shape information, which makes it the obvious first suspect — though not the only possible one (see
§7 R3).

## 2. Can MSMS do this at all? (the question Phase 6C never actually answered)

Yes, and our own Phase-6 C(a).2 attempt already proved it: it produced **220 ligand-owned vertices**
before dying downstream. `computeMSMS` is chemistry-blind — it calls `output_pdb_as_xyzrn`, which
emits `x y z radius 1 name`, then runs MSMS with `-density 3.0 -probe 1.5`. HETATM residues pass
when `residue.get_resname() in keep_hetatms`, and radii come from the same element table protein
atoms use.

**What was actually broken in Phase 6C was `triangulation/ligand_utils.py::extract_ligand`**, whose
only job is to return `(rdmol, mol2_file)`. It re-derives connectivity from PDB HETATM records via
prody/OpenBabel and then trips its own assertion `GetNumHeavyAtoms() < GetNumAtoms()` after
`neutralize_atoms`. Six integration fixes did not save it. Phase 7 does not fix it — Phase 7
**bypasses** it, because PDBbind ships `{id}_ligand.sdf` with correct bond orders in the bound pose,
and we already load **5,240 / 5,240** of them with RDKit. The one input `extract_ligand` fails to
produce, we already have.

Every one of the four channels already has a comparability-preserving ligand implementation in the
reference tree:

| channel | ligand implementation | why it lands on the protein's scale |
|---|---|---|
| `si` shape index | pymesh `vertex_mean/gaussian_curvature` → `arctan((k1+k2)/(k1−k2))·2/π` | pure geometry; the arctan makes it scale-free |
| `hbond` | `ligand_charges.prepare_rdmol` + `computeChargeHelperMol` | same angle-deviation / angle-penalty machinery as the protein branch |
| `hphob` | `computeHydrophobicity` + `kd_from_logp` = `clip(−6.2786 + exp(0.4772·logP + 1.8491), −4.5, 4.5)`, BRICS fragments as the residue analogue | **explicitly maps Crippen logP onto the Kyte–Doolittle range**; both sides then divided by 4.5 |
| `charge` | `computeAPBS(..., mol2_file=…)` | same Poisson–Boltzmann solver, same kT/e units, same `normalize_electrostatics` and `/10` |

Both ligand helpers key on `atom.GetPDBResidueInfo().GetName().strip()`, so we control the join by
writing the ligand PDB and setting the RDKit PDB atom names ourselves.

## 3. Design decisions

**D7-1 — Reuse, don't rewrite.** The ligand surface is built by the reference MSMS + pymesh +
channel code, driven by an RDKit mol supplied from the SDF. We do not write a from-scratch surface
generator and we do not repair `extract_ligand`.
*Rationale:* a from-scratch surface would be a second, differently-calibrated representation, and the
whole point is that the two sides of a pair must be the same kind of object.

**D7-2 — Identical mesh and channel parameters; one documented divergence.** Probe 1.5 Å, density
3.0, `mesh_res = 1.0`, same radii, same normalizations (`hphob/4.5`, `normalize_electrostatics`,
`charges/10`). **Divergence:** the reference `radii` table contains only N/O/C/H/S/P — so `F, Cl, Br,
I` fail the `atomtype in radii` test and are **silently dropped from the surface**. That is harmless
for proteins and unacceptable for drug-like ligands, so Phase 7 writes its own `xyzrn` emitter,
taking elements from RDKit (not from `atom_name[0]`, which also mis-types `CL1` as carbon) and
extending the table with Bondi radii F 1.47, Cl 1.75, Br 1.85, I 1.98.

**D7-3 — Training pair = protein-alone surface ↔ ligand-alone surface.** Each side's surface is
computed from that molecule in isolation, exactly as MaSIF computes each PPI chain's surface from
that chain alone. The two sides stay independently encodable.
*Rationale:* if either side's surface is built in the presence of the other, the partner's shape is
already in the query and retrieval is free — the Phase-6C lesson that made Path B two separate graphs.

**D7-4 — The composite protein+ligand surface is INFERENCE-ONLY.** It is built for the neosurface
benchmark query and never used as a training input.
*Rationale:* a "neosurface" is by definition the surface protein and drug create jointly, and our
Phase-6C protein surface — built without the drug — represents the pocket as an empty cavity, so the
shape a partner actually recognizes never existed in our representation. But building PDBbind protein
surfaces *with* their own ligand would put the ligand on both sides of the training pair (D7-3).
PDBbind has no partner protein, so there is no training data for a composite pair type; extending
this to training is out of scope and is flagged as such rather than quietly done.

**D7-5 — Electrostatics stay APBS on both sides.** If `pdb2pqr --ligand` cannot parameterize a
ligand-only input, the fallback is a self-written PQR (RDKit Gasteiger charges + the D7-2 radii) fed
to the *same* APBS solver — not a Coulomb approximation, which would put the two sides of a pair on
different physical scales. Whichever path is used is recorded per complex.

**D7-6 — Nothing is claimed from one seed.** Every trained condition runs **≥2 seeds**. The Phase-6C
do-no-harm gap (−0.041) is one seed against one seed and is explicitly *not* treated as established
until the seed spread is known.

**D7-7 — AF3-apo scope: the held-out set.** AF3 models are generated for the ~300 held-out `val_pl`
proteins, not the training corpus. Per the standing project rule, **the protein varies and the ligand
stays at its experimental pose** — there is no apo ligand. Placing the crystal ligand on the AF3
backbone requires superposing the AF3 model onto the holo protein first (`align/global_align.py`);
this is a real pipeline step, not a detail.
*Rationale:* held-out AF3 measures the holo→apo drop on the ligand axis (the north star applied to
protein–ligand) at a fraction of the cost of apo-augmented training. Training-set AF3 is a follow-on
decision gated on this result.

**D7-8 — Correction to the Phase-6C write-up.** `docs/19` stated that AF3 generation for PDBbind was
out of scope for Workstream C. That was wrong: the handoff's "no new AF3" line was written when C's
corpus was assumed to be the existing PPI set, and new training data implies its matching apo state.
Corrected in `docs/19` and the running log.

## 4. The gate

**Primary (capacity):** does a ligand surface lift **train-set** P–L retrieval above the Phase-6C
0.095 top-5? This is the question the Phase-6C diagnostic actually poses. If the train number does
not move, the ligand representation was *not* the bottleneck and Phase 7 must say so plainly.

**Secondary (generalization):** held-out P–L retrieval, on the full holdout, the scaffold-unseen
subset, and the scaffold-deduplicated subset.

**Tertiary (neosurface):** axis 3 with the composite query, with the ligand-present/absent contrast
still the control that decides whether any signal is ligand-dependent.

**Floor (do-no-harm):** the PPI gate, re-measured at ≥2 seeds. The Phase-6C −0.041 is re-tested, not
inherited.

**Robustness (north star):** holo→AF3-apo drop on the held-out ligand axis.

A negative result, honestly verified, is a valid finish — as in Phase 6C.

## 5. Stages

| stage | what | est. cost |
|---|---|---|
| S0 | Ligand-alone surface end-to-end on 5hls; read mesh + all 4 channel distributions | ~0 |
| S1 | Channel-comparability QC over ~50 ligands vs protein reference distributions; MSMS failure rate; clamped-curvature rate | ~0 |
| S2 | Full ligand-surface pass: 5,240 PDBbind + 14 ternary. **Incremental** — protein surfaces already exist, we only add vertices to the ligand npz | ~CHF 0.5 |
| S3 | Composite protein+ligand surfaces for the 14 ternary systems | ~CHF 0.1 |
| S4 | Retrain `combined` + `ppionly`, ≥2 seeds each | ~CHF 16 |
| S5 | AF3-apo for the ~300 held-out `val_pl` proteins + ligand-axis robustness eval | cost probe first |

## 6. What changes in the model path
Ligand graphs gain `vert_feat`, `vv_edge`/`vv_feat` and `va_v`/`va_a`/`va_feat`, so they become
structurally identical to protein chain graphs. Two consequences to state up front:
1. Today `is_ligand = 1` and `n_vert = 0` are **perfectly confounded**, and `agg_va` is identically
   zero for every ligand atom. Removing that confound changes the encoder's ligand pathway
   qualitatively — so the do-no-harm gate must be **re-measured**, never assumed to carry over.
2. Nothing about the npz contract changes, so `p4.dataset.ComplexP4` and the training loop are
   untouched; the mixture is still just a longer id list.

## 7. Risks
- **R1 — MSMS degeneracy on very small ligands.** A 1.5 Å probe can roll over a <10-heavy-atom
  molecule and yield a near-spherical or self-intersecting mesh. QC'd in S1; a documented skip rule
  beats a silently bad surface.
- **R2 — discrete curvature on small, highly curved meshes.** pymesh's estimator already needs the
  `H²−K < 0 → 1e-8` clamp; the clamp rate on ligands is measured, not assumed.
- **R3 — the representation may not be the bottleneck.** Protein–ligand complementarity may simply
  be an ambiguous target (many ligands fit many pockets; PDBbind is full of congeneric series), which
  would also depress the train-set number. The primary gate is designed to distinguish these: a
  capacity fix should move the *train* number first.
- **R4 — pdb2pqr may refuse ligand-only inputs** (D7-5 fallback).
- **R5 — AF3 frame mismatch** when placing the crystal ligand on an AF3 model (D7-7).
- **R6 — 28 neosurface cases is a small n.** Per-case ranks are always reported; pooled numbers alone
  would over-claim.
