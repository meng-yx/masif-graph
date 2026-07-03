# Phase 1 — Results & Go/No-Go

> **Status:** Complete. Autonomous run 2026-07-02 (headless, CPU, Jed job 64882013).
> **Question (the gate):** Does representing the surface as **per-atom pooled** fingerprints
> (instead of per-vertex) preserve the ability to separate true contacting atom pairs from
> decoys — and can we align proteins globally at atom granularity?
> **Spec:** [`01-phase1-design.md`](01-phase1-design.md). **Brief:** `PHASE1_HANDOFF.md`.
> **Full audit trail:** [`progress/phase1-log.md`](progress/phase1-log.md).

---

## TL;DR — CONDITIONAL GO for Phase 2 (mean pooling; drop max)

- **Mean pooling** of the frozen reference per-vertex descriptors onto surface atoms keeps
  descriptor-separation **ROC-AUC ≈ 0.89–0.94**, versus the **per-vertex baseline ≈ 0.94–0.97**
  on the *same* 40 complexes and the *same* contacts. The cost is **small but real and
  systematic**: paired per-complex deficit **+0.027 ± 0.019** under hard (neg_mix) negatives and
  **+0.052 ± 0.052** under random negatives; atom_mean is worse than vertex in **10–12 of 12**
  complexes.
- **Max pooling** is materially worse (paired deficit **+0.064 to +0.116**) — **do not use it**.
- The gap is **not** caused by poorly-exposed atoms and is **not** recovered by a min-exposure
  filter (the filter *hurts* — see §2). It is the intrinsic **variance-reduction blur** of
  averaging descriptors.
- **M2 global atom-level alignment** (rule-based, no learned scorer) recovers near-native poses
  (**iRMSD ≤ 5 Å, 66–81 % native contacts**) on complexes with large interfaces, and fails on
  small ones — a promising but not-yet-robust prototype.
- **Recommendation:** proceed to Phase 2 **with mean pooling**, treating "graph embeddings
  recover the ≈0.05 pooling gap" as an **explicit Phase-2 success gate**; keep **D1-B**
  (atom-centric surface readout) as the escalation path if they don't.

**Honesty note:** under the strictest reading (random negatives, pooled AUC, which reproduces
the documented ~0.98 baseline regime), the mean-pooling drop is ≈0.057 — right at the design's
"material drop ≳0.05" line. This is a **borderline** result, presented here without spin. It is
a GO because the drop is small in absolute terms, expected by design, and exactly what Phase-2
graph features are meant to buy back — **not** because pooling is free (it isn't).

---

## 0. What was run (methods)

- **Probe inputs (M0):** the frozen reference `masif-neosurf-af2` pipeline was run *as a tool*
  inside its Singularity image (`masif-neosurf_v0.1.sif`) to turn raw PDBs → regularized
  surfaces + per-vertex 80-D descriptors (`desc_straight`, `desc_flipped`), per-vertex interface
  labels, and shape-complementarity (`sc`) labels. Model = **`model_data_paper`** (published
  weights), locked via the `model_data → model_data_paper` symlink and **confirmed loaded** from
  the descriptor log (`Setting model_dir to nn_models/sc05/all_feat/model_data/`).
- **Complexes:** **N = 40**, drawn **deterministically** (seed-0 shuffle of `data/lists/testing.txt`)
  as the first 40 candidates that preprocessed cleanly. **1 skip:** `3KTM_C_F` (reference
  preprocessing produced no surface/descriptors). The set is order-stable (verified: all
  candidates up to the 40th-available were resolved).
- **Per-atom representation (D1-A):** vertex→heavy-atom KDTree map (`a(v)` = nearest heavy atom,
  H excluded); surface atom = any heavy atom owning ≥1 vertex; per-atom embedding = **mean** and
  **max** pool of owned-vertex `desc_straight` / `desc_flipped` (the *precomputed* flipped
  descriptors are pooled directly — never flip a pooled vector, per design §2.2).
- **Metric:** descriptor-separation ROC-AUC, reference convention (score = 1 / L2-distance between
  p1 **straight** and p2 **flipped** embeddings; positives should be close). Reproduced exactly
  from the reference `compute_roc_auc.py`.
- **Positives:** **shape-complementarity-filtered vertex contacts** — p1 vertex with nearest p2
  vertex < 1.0 Å **and** median `sc ∈ (0.5, 1.0)` — are the **primary** definition (this is what
  reproduces the reference ~0.98 baseline; unfiltered <1 Å contacts give only ~0.66, because the
  descriptor net was trained on sc-complementary contacts). Atom positives = owner atoms of those
  vertex contacts, deduplicated (design §4.2). Also reported: **unfiltered <1 Å** and **direct
  heavy-atom <4 Å** as secondary cross-checks.
- **Negatives (neg_mix, ratio 5):** cross-complex 0.5 / within-complex 0.3 / hard 0.2 (reference
  split). "Hard" = spatially closest non-positive cross-chain pairs. Positives are always excluded
  from negatives explicitly. A second **random-negative** scheme (positive p1 entity vs a random
  same-complex p2 entity — the reference sanity scheme, identical construction at both
  granularities) is reported alongside because it has **no hard/within-radius confound** and is
  the cleanest apples-to-apples comparison.
- **Controls (mandatory, all passed):** shuffled-label AUC (§Controls); identical complexes and
  identical pair logic for vertex vs atom; per-complex spread; hand-checkable metric unit tests.

Reproduce: `bash scripts/run_phase1_final.sh 40` (uses env `masif-graph`). Code:
`src/masif_graph/{io,surface,pairs,metrics,align,experiments}`. Raw outputs: `logs/m1/`,
`logs/m2/`, figures in `docs/figures/`.

---

## 1. Per-vertex vs per-atom descriptor separation (the headline)

**SC-filtered positives, same 40 complexes, same contacts.** Pooled = over all pairs from all 40
complexes; per-complex median over the 12–15 complexes with ≥10 positives.

| granularity | pooled AUC (neg_mix) | pooled AUC (random-neg) | per-complex median | per-complex range |
|---|---|---|---|---|
| **per-vertex (baseline)** | **0.944** | **0.946** | **0.972** | 0.846 – 0.996 |
| per-atom, **mean** pool | 0.916 | 0.889 | 0.939 | 0.806 – 0.990 |
| per-atom, **max** pool | 0.879 | 0.830 | 0.890 | 0.809 – 0.972 |

**Paired per-complex deficit (vertex − atom, same complex; 12 complexes with ≥10 sc-positives):**

| comparison | neg_mix Δ (mean ± sd, median) | random-neg Δ (mean ± sd, median) | atom worse in |
|---|---|---|---|
| vertex − atom_**mean** | +0.027 ± 0.019 (med +0.032) | **+0.052 ± 0.052 (med +0.047)** | 10–12 / 12 |
| vertex − atom_**max** | +0.064 ± 0.030 (med +0.065) | +0.116 ± 0.055 (med +0.109) | 12 / 12 |

Per-complex paired detail (random-neg, atom_mean): deficits range from **+0.138** (2QYI) down to
**−0.036** (2A6P, where atom *beats* vertex). 10/12 complexes favor vertex; the effect is
consistent but modest, with two complexes reversing.

**Secondary positive definitions (apples-to-apples sanity):**

| positive def | vertex (pooled negmix) | atom_mean | atom_max | paired vertex−atom_mean (negmix) |
|---|---|---|---|---|
| unfiltered <1 Å | 0.659 | 0.665 | 0.666 | **−0.005** (atom ~equal) |
| direct heavy-atom <4 Å | — | 0.610 | 0.613 | — |

Under the **weak** (unfiltered) positive definition, per-atom ≈ per-vertex (Δ −0.005 under
neg_mix) — the pooling penalty appears **only** in the high-complementarity (sc-filtered) regime
where sub-Ångström shape matters. This is the coherent, expected signature of "pooling blurs the
fine complementarity signal" and is direct evidence there is **no atom-side inflation**.

---

## 2. Exposure stratification + min-exposure filter

`n_owned_vertices` = number of surface vertices a surface atom owns (its "exposure"). Positives
are binned by the **minimum** exposure of the two atoms in the pair. **atom_mean, sc, N=40:**

| min-exposure bin | n_pos | AUC |
|---|---|---|
| 1 | 47 | 0.928 |
| 2–3 | 131 | 0.928 |
| 4–7 | 209 | 0.915 |
| 8–15 | 58 | **0.881** |

**Min-exposure filter sweep** (keep only pairs with min-exposure ≥ T; recompute pooled AUC):

| T ≥ | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| AUC | 0.916 | 0.915 | 0.909 | 0.908 | 0.902 | 0.893 | 0.881 |
| frac pos kept | 1.00 | 0.89 | 0.74 | 0.60 | 0.43 | 0.31 | 0.13 |

**Key (counterintuitive) finding:** for **mean** pooling the trend is the *opposite* of the T1
"barely-exposed atoms are noisy" hypothesis. **Low-exposure atoms give the *highest* AUC** and
the min-exposure filter **monotonically lowers** it. Reason: a 1-vertex atom's mean pool *is* the
raw vertex descriptor (no blur); averaging blur grows with exposure. **The min-exposure filter
does not recover the pooling gap — it makes it worse.** (Max pooling shows the reverse bin trend,
consistent with it distorting the descriptor geometry regardless of exposure.)

---

## 3. Positive-vs-negative distance-distribution overlap (atom vs vertex)

Quantitative summary (SC-filtered; neg_mix negatives; from `docs/figures/overlap_summary.json`).
Plots: `docs/figures/overlap_sc.png` (and `_unfiltered`, `_atom_direct`).

| granularity | overlap coeff. (↓ better) | Cohen's d (↑ better) | pos median | neg median |
|---|---|---|---|---|
| per-vertex | **0.214** | **1.79** | 1.88 | 3.49 |
| atom_mean | 0.304 | 1.61 | 1.84 | 3.24 |
| atom_max | 0.384 | 1.44 | 2.23 | 3.49 |

Mechanism, visible in the numbers: atom_mean's **positive** median (1.84) is essentially
unchanged from vertex (1.88) — true contacts stay close — but its **negative** median drops
(3.49 → 3.24). Averaging reduces embedding variance, so pooled embeddings cluster together and
**negatives move closer to positives**, increasing overlap (0.214 → 0.304). That is precisely the
pooling cost, and it is why random negatives (which sit in this compressed tail) show a larger
gap than spatially-hard negatives.

---

## 4. Milestone-2 — global atom-level alignment prototype

Rule-based, **no learned scorer** (that is Phase 2). Per complex: perturb the binder by a random
rigid pose, build interface-gated (reference MaSIF-site, design D8) embedding correspondences
(complementary: p1 straight vs p2 flipped), fit with **RANSAC-Kabsch**, measure interface-RMSD
of the recovered binder vs native and fraction of native contacts recovered. 10 complexes:

| complex | #corr | corr precision | iRMSD start → recovered (Å) | native contacts recovered |
|---|---|---|---|---|
| 1JXQ_C_D | 149 | 0.24 | 111 → **3.42** | **0.81** |
| 3TDM_A_B | 89 | 0.24 | 68 → **3.54** | **0.80** |
| 2AOB_C_D | 200 | 0.15 | 47 → **3.94** | **0.66** |
| 3HF5_A_D | 144 | 0.28 | 48 → 6.36 | 0.36 |
| 3HTR_A_B | 32 | 0.44 | 99 → 10.27 | 0.15 |
| 2WQ4_A_C | 86 | 0.07 | 44 → 12.11 | 0.16 |
| 3B08_A_B | 19 | 0.05 | 80 → 16.68 | 0.05 |
| 2QLC_C_B | 20 | 0.00 | 35 → 35.52 | 0.00 |
| 1TZI_BA_V | 0 | — | failed (no correspondences) | — |
| 2P4A_A_B | 3 | — | failed (too few correspondences) | — |

**Summary:** 8/10 produce a pose; **3 recover to ≤5 Å with 66–81 % native contacts**; median
iRMSD 8.3 Å. Success tracks interface size / correspondence count: large interfaces → near-native;
small interfaces (few high-iface atoms) → few correspondences → failure.

**Two important engineering findings (documented divergences from the handoff's Open3D recipe):**
1. Contacting atom **centers are ~4–5 Å apart** (van der Waals), *not* coincident. Open3D's
   point-to-point `registration_ransac_based_on_correspondence` is unstable on offset centers —
   it returned **19.6 Å even with ground-truth correspondences**, whereas a direct least-squares
   **Kabsch over the same pairs gives 2.4 Å**. We therefore use a custom RANSAC-Kabsch (inlier
   threshold above the vdW offset, refit on all inliers).
2. Point-to-**plane** ICP lets the binder slide tangentially along the interface and **degrades**
   the pose (11–15 Å); point-to-point ICP over atom centers **overpacks** (co-locating centers
   pulls the binder ~vdW into the target). Both are reported as negative results; the reported
   pose is the RANSAC-Kabsch fit.

This validates that "co-locate complementary atom pairs globally" **can** recover native poses at
atom granularity when correspondences are good enough, and quantifies why a **learned scorer**
(Phase 2/D7) — able to rank many candidate poses instead of trusting one rule-based fit — is
needed for robustness.

---

## 5. GO / NO-GO recommendation for Phase 2

**Recommendation: CONDITIONAL GO.** Proceed to Phase 2 (graph embeddings A/B + contrastive
retraining) using **mean pooling**; **drop max pooling**.

**Reasoning:**
1. **Mean pooling is viable.** Per-atom AUC (0.89–0.94) is far above chance and above the
   unfiltered-vertex regime (0.66); the representation clearly retains most of the
   complementarity signal. Per-complex median 0.939 vs vertex 0.972.
2. **The cost is small, real, and expected.** Paired deficit +0.03 (hard negatives) to +0.05
   (random negatives). The design explicitly anticipates Phase 1 landing "at or slightly below
   baseline" and states a small drop "is not failure — the intended gain is Phase 2." A ~0.05 AUC
   deficit is a plausible target for two orthogonal Phase-2 embedding families to recover.
3. **The cost is not a fixable artifact.** It is the intrinsic variance-reduction blur of
   averaging (§3), it is systematic (atom worse in 10–12/12 complexes), and the min-exposure
   filter does **not** recover it (§2) — so there is no cheap patch; the remedy is added signal
   (Phase 2), exactly as planned.
4. **Global alignment is feasible** at atom granularity (§4).

**Conditions / how to hold Phase 2 accountable:**
- Set an **explicit Phase-2 gate**: with graph embeddings A (bonded chemistry) + B (local
  geometry) fused and retrained contrastively, per-atom AUC must **meet or exceed the per-vertex
  baseline** (recover the ≈0.05 random-neg gap) on this same probe. Keep the ablation (surface-only
  vs +A vs +B vs full) so the recovery is attributable.
- If Phase 2 does **not** recover it, **escalate to D1-B** (atom-centric surface readout) before
  investing further — the pooling blur would then be the binding constraint, not the chemistry.
- Max pooling is out; if a learned pooling (attention) is wanted later it must be re-benchmarked.

**Per-complex spread (not just the pooled number):** vertex per-complex AUC 0.846–0.996
(median 0.972); atom_mean 0.806–0.990 (median 0.939). The paired deficit is consistent
(10–12/12 complexes worse) but modest and reverses on 2 complexes. N=40 (12 with ≥10 sc-positives)
is small — treat the ±0.05 sd on the paired delta as real noise, not false precision.

---

## 6. Decisions & assumptions (judgement calls made autonomously)

1. **SC-filtered positives are primary.** The handoff said "<1.0 Å"; empirically that alone gives
   AUC ~0.66, while adding the reference shape-complementarity gate `sc ∈ (0.5,1.0)` reproduces
   the documented ~0.98 baseline. The descriptor net was trained on sc-complementary contacts, so
   this is the correct apples-to-apples baseline. Unfiltered and direct-atom variants are reported
   too. **This is the single most consequential decision** — the gate is defined against the
   ~0.98 baseline, which requires the sc filter.
2. **Report two negative schemes.** neg_mix (design-specified, spatial-hard) and random-neg
   (reference sanity, confound-free). The gap differs between them (0.03 vs 0.05); both are shown
   and the random-neg one is treated as the more rigorous cross-granularity comparison because its
   construction is identical for vertex and atom (no radius confound).
3. **Vertex coords read from precompute `.npy`, not the `.ply`.** Verified row-aligned with the
   descriptors (reference loops vertices in order with no reordering); avoids a pymesh dependency.
   `.ply` normals (M2 only) are consistency-checked against these coords.
4. **Heavy atoms** parsed from the protonated per-chain PDB (`01-benchmark_pdbs/…`, same frame as
   the surface); H/D and solvent excluded; altloc A/space kept. Self-contained parser (no biopython
   dependency for the gate).
5. **neg_mix hard/within radii:** vertex 3 Å, atom 6 Å (atoms are sparser; "closest non-positive"
   is selected by sorting, so the radius only bounds the candidate search). Because this differs by
   granularity, the confound-free **random-neg** paired delta is used for the primary gate read.
6. **Per-complex spread** computed only over complexes with ≥10 positives (a 1-positive AUC is
   trivially 0/1). Pooled AUC uses all 40 complexes' pairs.
7. **M2 divergences from the Open3D recipe** (custom RANSAC-Kabsch; point-to-point not point-to-
   plane), justified by the offset-center degeneracy — see §4 and the progress log.
8. **Min-exposure bins/thresholds** and desc/RANSAC cutoffs are the documented defaults in
   `run_m1.py` / `global_align.py`; conclusions are qualitative and not sensitive to small changes.

---

## 7. What was NOT tested / caveats (honest limits)

- **"The pipeline ran" ≠ "the science is airtight."** The pipeline ran cleanly and all mandatory
  controls passed (shuffled ≈0.50; apples-to-apples confirmed; hand-checked metric). What is
  *earned*: mean pooling costs ≈0.03–0.05 AUC vs per-vertex on this metric/probe. What is *not*
  proven: that this transfers to full-scale retrieval or that Phase-2 features recover it.
- **Small N.** 40 complexes; only **12** have ≥10 sc-filtered positives (sc-complementary contacts
  are sparse), so the *paired* delta rests on 12 complexes. The paired sd (±0.05) is real noise.
  Not run on a second independent test set.
- **Baseline is my reconstruction**, not the reference's own reported 0.988 (that was on the full
  959-complex test set with the training cache's exact negatives). My per-complex median (0.972)
  is close; the pooled value (0.946) is lower due to cross-complex distance-scale mixing. I did not
  re-run the reference's full training-time AUC.
- **Frozen descriptors only.** No retraining, no graph features, no learned pooling, no learned
  scorer, no ligands (all Phase 2+). The whole point is to measure the *pooling* change in
  isolation; these are out of scope by design.
- **M2 is a prototype**, single random seed per complex, 10 complexes, iface-gated. It is a
  feasibility check, not a benchmarked aligner; contact-recovery uses a 5 Å cutoff and the
  RANSAC-Kabsch pose overpacks by ~vdW. No clash filtering, no scoring of alternative poses.
- **Negative-scheme dependence is a genuine ambiguity**, not resolved here: the gate reads as a
  clean pass (Δ 0.027 < the 0.05 material line) under hard negatives and as borderline-material
  (Δ 0.052) under random negatives. I report both and lean on the more conservative one for the
  recommendation.
- **APBS/electrostatics** survives implicitly (baked into the pooled reference descriptors); its
  cost for v2 ensembles (design D10/T6) was not examined.

---

*Numbers in this document trace to code that was run: `logs/m1/m1_results.json`,
`logs/m1/summary.txt`, `logs/m1/paired_deltas.json`, `logs/m2/m2_results.json`,
`docs/figures/overlap_summary.json`. Commands and config are recoverable via
`scripts/run_phase1_final.sh` and the module defaults. No numbers were hand-entered from memory.*
