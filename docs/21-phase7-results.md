# Phase 7 — results: giving the ligand a shape channel

> Design: `docs/20-phase7-design.md` (D7-1 … D7-8). Build log: `docs/progress/phase7-log.md`.
> All numbers come from `logs/phase7/results/*.json` via `scripts/p7_collect_results.py`.
> **Every trained condition was run at 2 seeds (D7-6); nothing below is claimed from one seed.**

## 0. Verdict in one paragraph

**The hypothesis that motivated Phase 7 is not supported.** Giving the ligand a real MSMS surface
did **not** move the capacity bottleneck: train-set protein–ligand retrieval is unchanged
(0.111 ± 0.007 vs 0.119 ± 0.024 top-5). It *did* improve held-out ligand retrieval
(0.054 → 0.084 top-5; 0.076 → 0.111 on the scaffold-clean subset), i.e. it helps the model
**generalise** what it could already fit rather than letting it fit more — the opposite of the
proposed mechanism. And it costs a **large** do-no-harm violation: PPI retrieval falls from
0.644 ± 0.007 to 0.475 ± 0.020, a −0.169 regression present in both seeds. The neosurface axis
stays null, and the composite-surface hypothesis is **refuted** rather than merely unsupported.
Separately, the Phase-6C −0.041 do-no-harm gap that prompted this phase **does not survive two
seeds** (−0.017 ± 0.019). Recommendation: **do not adopt the ligand surface as it stands.**

## 1. What was built (all of it verified, see the log)

* **5,239 / 5,240** ligand MSMS surfaces, same pipeline and parameters as the protein side
  (probe 1.5 Å, density 3.0, `mesh_res` 1.0), attached to the ligand graphs as a controlled A/B —
  only the vertex side differs from Phase 6C; protein npz and contacts are symlinked, bit-identical.
* **42/42** neosurface artefacts: 14 drug surfaces + 28 composite protein+drug surfaces.
* **298/298** AF3-apo proteins for the held-out ligand set, with contacts recomputed against the
  crystal ligand pose (contact ratio AF3/holo median **0.99**, zero failures — the superposition is
  sound).
* Three defects fixed on the way: halogens were being **silently dropped** from surfaces (the
  reference radii table has no F/Cl/Br/I); `pdb2pqr` rejects the `am` bond type; and `read_msms`
  decodes MSMS's non-ASCII banner with the *locale default*, so it works under UTF-8 and dies under
  POSIX — an environment-dependent failure that would have hit part of a 5,000-job array.

## 2. Axis 1 — do-no-harm PPI gate (frozen 287-clean set, n=269, DB=538)

| model | seeds | HH top5 | HH medR | AA top5 | holo→AA drop |
|---|---|---|---|---|---|
| random-init (chance) | 1 | 0.015 | 258 | 0.009 | +0.006 |
| PPI-only (control) | 2 | **0.644 ± 0.007** | 1 | 0.654 ± 0.006 | −0.010 |
| Phase-6C combined | 2 | 0.627 ± 0.018 | 1 | 0.639 ± 0.017 | −0.012 |
| **Phase-7 combined (surface)** | 2 | **0.475 ± 0.020** | **9 ± 3** | 0.493 ± 0.029 | −0.019 |

*frozen MaSIF on identical patches: HH 0.084 (medR 110). Chance top5 0.0093.*

**Two findings.**

1. **The Phase-6C do-no-harm gap is not real.** Combined vs PPI-only is **−0.017 ± 0.019** — inside
   the seed spread. The −0.041 seen in Phase 6C was one seed against one seed. The instinct to
   demand ≥2 seeds before believing it was correct, and it dissolved the finding.
2. **Phase 7 fails do-no-harm decisively.** −0.169 ± 0.021 against the control, ten times the
   Phase-6C gap and far outside any spread. Median rank degrades from 1 to 9. Per seed: 0.494 and
   0.455 — so this is **not** attributable to the Stage-A divergence on seed 1 (log §9); both seeds
   show it.

## 3. Axis 2 — the capacity gate (the question Phase 7 was built to answer)

Phase 6C's diagnostic was that the model could not fit the ligand axis **on its own training data**.
That is what a representation fix should repair first.

### 2a — TRAIN-set retrieval (DB 281, chance top5 0.018)

| model | seeds | PPI top5 | **P–L top5** | P–L medR |
|---|---|---|---|---|
| PPI-only | 2 | 0.447 ± 0.018 | 0.026 ± 0.003 | 134 ± 3 |
| ligand-only | 1 | 0.028 | 0.041 | 105 |
| Phase-6C combined | 2 | 0.448 ± 0.019 | **0.119 ± 0.024** | 38 ± 7 |
| **Phase-7 combined** | 2 | **0.363 ± 0.010** | **0.111 ± 0.007** | 32 ± 1 |

**The primary gate is NOT met.** Train-set P–L top-5 is unchanged — 0.111 ± 0.007 against
0.119 ± 0.024. Whatever limits the model's ability to fit protein–ligand complementarity, **it was
not the ligand's missing shape channel.**

Note also that PPI *training* retrieval drops (0.448 → 0.363). The Phase-7 model fits **both** tasks
less well, which points at capacity competition or under-training rather than overfitting — the
ligand side roughly doubled in graph size while encoder width and epoch budget stayed fixed.

### 2b/2c — held-out retrieval

| model | seeds | P–L top5 (full holdout) | medR | P–L top5 (scaffold-unseen) | medR |
|---|---|---|---|---|---|
| PPI-only | 2 | 0.019 ± 0.002 | 148 | 0.026 ± 0.003 | 96 |
| ligand-only | 1 | 0.036 | 116 | 0.042 | 77 |
| Phase-6C combined | 2 | 0.054 ± 0.015 | 74 ± 2 | 0.076 ± 0.013 | 52 ± 2 |
| **Phase-7 combined** | 2 | **0.084 ± 0.012** | **57 ± 1** | **0.111 ± 0.007** | **40 ± 0** |

Here the surface **does** help: seed ranges do not overlap on either subset, and the effect is
larger on the stricter scaffold-clean subset (+46% relative). So the ligand surface improves
**generalisation** without improving **capacity** — it did not let the model fit more, it let it
transfer more of what it already fit.

**Honesty note.** The ligand-axis metrics have large seed variance (Phase-6C combined: P–L train
0.095 vs 0.143 across its two seeds). Two seeds is the minimum, not a comfortable margin; the
held-out improvement is **suggestive, not established**, and a third seed would be cheap insurance.

## 4. Axis 3 — neosurface benchmark (28 cases, DB 596, chance medR ≈ 298)

| model | seeds | sep-surface medR | composite medR | composite no-ligand medR |
|---|---|---|---|---|
| Phase-6C combined | 2 | 260 ± 6 | — | — |
| **Phase-7 combined** | 2 | **347 ± 15** | **337 ± 2** | 335 ± 13 |

**Null, and worse than Phase 6C.** Both Phase-7 seeds land *below chance* (347, 332 vs ≈298).

**The composite-surface hypothesis is refuted.** I argued in the design that building the protein
surface *without* the drug — leaving the pocket as an empty cavity — was the likelier cause of the
axis-3 null than the ligand's own shape channel. Building real composite protein+drug surfaces
(28/28, with 382 drug-owned vertices on 6QTL_A, so the object is genuinely there) changes nothing:
composite 337 ± 2 vs separate-surface 347 ± 15, and dropping the drug's own rows changes it again
not at all (335 ± 13). That was my strongest remaining hypothesis for axis 3 and it is wrong.

## 5. Axis 4 (new) — ligand-axis holo→AF3-apo robustness, the north star

283/300 held-out complexes usable (≥8 contacts in both states).

| model | seeds | P(holo)→lig top5 | P(AF3)→lig top5 | drop | holo medR | AF3 medR |
|---|---|---|---|---|---|---|
| PPI-only | 2 | 0.027 ± 0.005 | 0.011 ± 0.000 | +0.016 | 139 | 144 |
| Phase-6C combined | 2 | 0.053 ± 0.021 | 0.046 ± 0.004 | **+0.007 ± 0.018** | 72 ± 4 | 86 ± 2 |
| **Phase-7 combined** | 2 | 0.074 ± 0.007 | 0.060 ± 0.000 | **+0.014 ± 0.007** | 48 ± 0 | 62 ± 2 |

**The encoder is conformation-robust on the ligand axis too.** Swapping the crystal protein for an
AF3 prediction costs only +0.007 / +0.014 top-5 — comparable to the seed spread, and consistent with
the Phase-4/5 finding that invariance is a property of the training recipe. Phase 7 is better than
Phase 6C in *both* conformational states, which is the same generalisation effect as §3, not a
robustness effect.

This axis is a genuine addition: it is the project's north star applied to protein–ligand for the
first time, and it says the apo problem is **not** the ligand axis's limiting factor either.

## 6. What this means

* **The ligand representation was not the bottleneck.** Two independent representation upgrades —
  Phase 6C's unified 26-D atom features and Phase 7's full MSMS surface — both failed to move
  train-set P–L retrieval. The limiting factor is somewhere else: the objective, the pair definition,
  or the intrinsic ambiguity of the target (PDBbind is full of congeneric series binding the same
  pocket, so "which ligand binds this pocket" may simply be a many-to-many relation that a
  retrieval loss cannot sharpen). Risk R3 in the design anticipated this; it is now the leading
  explanation.
* **Adding capacity-hungry inputs to a shared encoder is not free.** The PPI regression is large and
  reproducible, and PPI *training* accuracy fell too — so before concluding the two tasks conflict
  fundamentally, the cheap test is a wider encoder / longer schedule, which this phase did not run.
* **Do not adopt the ligand surface as it stands.** It buys a suggestive held-out ligand improvement
  at the cost of a −0.169 PPI regression and a worse neosurface number.

## 7. Honest limitations

* **n = 2 seeds.** Enough to kill the Phase-6C −0.041 gap and to establish the −0.169 Phase-7
  regression, but thin for the held-out ligand improvement.
* **Confounded comparison.** The Phase-7 arm has ~2× the ligand-side graph size at identical encoder
  width and epoch budget, so "surface vs no surface" is entangled with "more compute needed". The
  PPI-train drop is the evidence for this, and it is unresolved.
* **Axis 3 is n = 28** with a single decoy DB; per-case ranks are in `neosurf_*.json`.
* **0.9% of ligands have degenerate surfaces** (45/5,239, all ≤13 atoms, listed in
  `logs/phase7/lig_surface_degenerate.txt`); 5 are in the held-out set. Too few to change anything,
  but the axis-2 numbers were not re-run with them excluded.
* **Stage A diverged on one Phase-7 seed** (gradient norms to 1.5e6). Best-checkpoint selection
  contained it at epoch 5, and both seeds show the same PPI regression, so it does not explain the
  headline — but the Stage-A recipe is **not robust on the combined corpus**, which is its own
  finding.

## 8. Cost

≈ **CHF 22** total: ligand surfaces ~2, neosurfaces ~0.2, AF3 (MSA + inference + surfaces) ~8,
four GPU training runs ~10, evaluations ~2. AF3 inference came in at 60 s/chain against a ~12 min
projection, which is why probing before committing mattered.

## 9. Reproduction

```bash
# surfaces
sbatch --array=0-116%120 scripts/p7_lig_surface_array.sbatch          # ligand MSMS surfaces
sbatch scripts/p7_attach.sbatch                                       # attach to the ligand npz
sbatch scripts/p7_neosurf_surfaces.sbatch                             # 14 drugs + 28 composites
# training (Kuma), DATA_ROOT selects the representation
sbatch --export=ALL,DATA_ROOT=/work/upthomae/Meng/phase7 p7_kuma_pipeline.sbatch p7comb_s0 combined 15 32 0
# evaluation
bash scripts/p7_run_all_evals.sh                                      # 3-axis gate, 7 checkpoints
sbatch scripts/p7_robust.sbatch <tag> <ckpt> <data>                   # axis 4
python scripts/p7_collect_results.py                                  # the tables above
```
