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
