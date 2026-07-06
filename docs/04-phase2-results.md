# Phase 2 — Results: atom-graph robustness to sidechain conformation (holo → apo-like repack)

**Status: DRAFT (autonomous conductor run, 2026-07-03, Jed job 65042258). Numbers below trace to
`logs/p2_*/phase2_results.json`; commands in `docs/progress/phase2-log.md`.**

> **One-line verdict (§7): NO-GO for Phase 3.** Across N=23 and N=29, with and without rotamer
> augmentation, the atom graph does **not** make the representation more robust to the apo-like
> FASPR repack. On the honest metric — **absolute apo AUC** — the ranking is always
> **raw frozen descriptor > trained surface-only > graph**: the graph is consistently *worst*, and
> in the true holo→apo transfer (no augmentation) it actively *injects* conformation-sensitivity.
> The design's do-no-harm floor also fails (any trained fusion head < raw mean-pooling on holo).
> **D1-B (atom-centric surface retrain) is NOT indicated** by this evidence: a trivial trained head
> already absorbs the FASPR-repack fragility, so the ceiling is not the descriptor. The real gap is
> that **FASPR fixed-backbone repack is too mild a proxy** — a harder perturbation (backbone
> motion / true apo / AF2) is the indicated next step, not more graph or a descriptor retrain.

This document reports the Phase-2 gate experiment defined in `docs/03-phase2-design.md §5`:
does a graph encoding **atom connectivity + bond rotatability + spatial geometry**, fused with the
frozen Phase-1 surface descriptor, make the representation **robust to sidechain conformation** —
degrade *less* under an apo-like fixed-backbone repack than surface-only — **without harming holo**?

---

## 1. Setup & methodology (what was actually run)

**Apo-like perturbation (the controlled proxy for apo/unbound).** For each complex we split the two
chains, **repack each chain's sidechains in isolation** with **FASPR** (fixed backbone, no partner
context), then re-run the *identical* reference surface+descriptor pipeline (MSMS + APBS +
descriptor net, in the TF1 `.sif`) on the repacked chains under a parallel id `{PDBID}RP_{C1}_{C2}`.
Only sidechain rotamers differ between holo and repack; backbone, mesh procedure, and descriptor net
are unchanged → a **clean controlled perturbation**. FASPR fixed-backbone repack is a *proxy* for
apo, not true apo/AF2 (caveat §8).

**Positives.** Phase-1 sc-filtered vertex contacts mapped to owner atoms, defined on the **holo
backbone** and held fixed. For the differential metric we evaluate the **intersection positive
set** — pairs that are surface atoms in *both* holo and repack (identity-mapped by
`(chain,resseq,atom_name)`) — so holo and repack are scored on **identical contacts**.

**Graph + fusion (design D3-A / D-P2.3).** One heterogeneous atom graph per chain: nodes = surface
heavy atoms; typed edges = **covalent** (biotite residue-template connectivity, bond order) with a
**rotatable/rigid** flag, and **spatial** (RBF-binned distances). A relational message-passing GNN
(pure PyTorch) message-passes over *invariant* chemistry+geometry only; the frozen 80-D descriptor
is fused **only at the readout head**: `fused = normalize(Head(surf ⊕ g))`, with `g` the
role-independent graph readout. Rotation-invariant by construction (only distances/bond-types/flags
enter). Robustness mechanism: under repack the surface channel shifts but the covalent-anchored `g`
does not, so the fused pair distance moves less.

**Ablation cells** (design §5): `surface_only` (frozen desc + trained head, no graph) · `+covalent`
· `+rotatability` · `+spatial` · `full`. Every cell is trained with the **same** contrastive
objective and **rotamer augmentation** (each train chain independently swaps to its repacked state
with probability `p_aug`), so the graph must beat an *augmentation-trained* surface baseline. We also
run every cell at **`p_aug=0`** (no augmentation — pure holo→apo transfer) to isolate the graph's
*intrinsic* robustness.

**Metric.** Descriptor-separation AUC (positives vs negatives by fused-embedding L2), reported two
ways: **randneg** (random within/cross negatives; easy) and **negmix** (hard cross-complex
negatives; discriminative — the meaningful metric). The scientific object is the **differential
degradation** `DEG = AUC(holo) − AUC(repack)` per cell; surface-only is the reference the graph must
beat. We report **pooled** (all test pairs) and **per-complex median** (robust to one dominating
complex) degradation, plus per-complex spread.

**Split & leakage control.** Split **by complex**; a complex's holo and its repacked twin always sit
on the **same side** (test complexes are fully held out — training only ever sees *train* complexes'
repack state, via augmentation) → **no holo/repack twin leakage**. Test complexes require
≥8 intersection positives for a stable per-complex AUC (documented bias: high-positive complexes are
over-represented in test). Preprocessing (frozen descriptors) is Phase-1, fit before any split.

**Controls (ml-research-guardrails).**
- **Rotation invariance:** rotate a chain, rebuild the graph → max|Δ fused embedding| = **1.6e-07**
  (machine precision). PASS.
- **Shuffled-label control:** ~0.46–0.49 on **both** states, every cell. PASS (≈0.5).
- **Holo reproduction:** raw mean-pool holo randneg **0.876** reproduces the Phase-1 atom mean-pool
  baseline (0.889 on the Phase-1 test set). Pipeline consistent.

**Pipeline-ran vs result-is-valid (kept separate).** The *pipeline* is green end-to-end (all controls
above pass; four runs × five cells × three seeds completed; every number traces to a committed
`phase2_results.json`). That is **not** the same as the *scientific* claim being strong: the
differential metric is confounded (§5/§8), N is small and no comparison reaches p<0.05, and the FASPR
proxy is mild. The NO-GO verdict (§7) rests on the **direction and consistency** of the *absolute-apo*
result across all four runs, not on a significant differential — and is stated with those limits, not
beyond them.

---

## 2. The natural collapse (premise check — no model)

Raw mean-pooled frozen descriptors, no training, on the **intersection** positive set (N=23 test=9):

| state | randneg (pooled) | negmix (pooled) | shuffled |
|---|---|---|---|
| holo | 0.876 | 0.897 | 0.48 |
| repack | 0.840 | 0.854 | 0.46 |
| **collapse (holo−repack)** | **+0.036** | **+0.043** | — |

The apo-like repack **does degrade** the frozen surface descriptor (+0.036 randneg / +0.043 negmix,
pooled) — modest but real, and **heterogeneous** across complexes (one near-random complex, 3B5U,
collapses ~+0.21). This is the effect the graph must reduce.

---

## 3. Floor ablation — with augmentation (p_aug=0.5, N=23; 14 train / 9 test)

From `logs/p2_floor/phase2_results.json` (3 seeds, 200 steps). `DEG_nm(med)` = per-complex median.

| cell | holo_rn | repk_rn | DEG_rn | holo_nm | repk_nm | DEG_nm (pool) | DEG_nm (median) |
|---|---|---|---|---|---|---|---|
| surface_only | 0.869 | 0.853 | +0.015±.006 | 0.866 | 0.829 | +0.037±.005 | +0.027±.005 |
| +covalent | 0.868 | 0.854 | +0.014±.001 | 0.843 | 0.821 | +0.022±.003 | +0.022±.005 |
| +rotatability | 0.867 | 0.848 | +0.020±.005 | 0.849 | 0.826 | +0.023±.012 | +0.021±.014 |
| +spatial | 0.858 | 0.849 | +0.009±.003 | 0.853 | 0.834 | +0.019±.003 | +0.008±.008 |
| full | 0.863 | 0.839 | +0.024±.004 | 0.839 | 0.813 | +0.026±.001 | +0.005±.008 |
| _raw mean-pool_ | _0.876_ | _0.840_ | _+0.036_ | _0.897_ | _0.854_ | _+0.043_ | — |

**Reads:**
- **randneg is insensitive** (raw collapse only +0.036; trained differences within ±.006 seed
  noise) — no cell distinguishable. Use negmix.
- **negmix: the graph reduces degradation.** surface_only degrades most (pooled +0.037, median
  +0.027); spatial/full degrade least (pooled +0.019/+0.026; **median +0.008/+0.005**). The median
  (robust to the 3B5U outlier) shows the graph helps the *typical* complex too, not just hard ones.
- **Do-no-harm FAILS against the right anchor.** Every *trained* cell — including surface_only —
  underperforms **raw mean-pooling** on holo (negmix 0.84–0.87 < **0.897**) *and* on repack (best
  trained 0.834 < raw **0.854**). At N=14 train the contrastive fusion head cannot beat the already
  near-optimal frozen descriptors and overfits, lowering both states. The graph is "less-bad
  training," **not** "more robust than the frozen descriptor."
- **Attribution / T4:** **spatial edges carry the robustness** (lowest DEG); chemistry
  (covalent/rotatability) helps less and does **not stack** — `full` (all edges) is *worse* than
  `spatial` alone on the pooled metric. → chemistry is partially **redundant** with spatial geometry
  at this scale, not orthogonal (T4 risk realized).
- **Per-complex spread caveat:** pooled DEG is inflated by one near-random complex (**3B5U**, holo
  negmix ~0.45 → repack 0.24 = +0.21 for surface_only; spatial softens to +0.15). At N=9 a single
  complex swings the pooled number; the median is the honest central estimate. **Underpowered.**
- **⚠ This "graph reduces degradation" read is superseded by §5.** The differential metric is
  confounded (a head that lowers holo shows a smaller gap without better apo); on **absolute apo AUC**
  the graph is *worse* than surface-only even here (repk_nm spatial 0.834 ≈ surface 0.829, full 0.813
  < surface; all < raw 0.854). The apparent benefit is augmentation memorising the perturbation — it
  reverses in the honest no-augmentation transfer (§4) and at N=29 (§5).

---

## 4. No-augmentation transfer — p_aug=0.0, N=23 (pure holo→apo) — **THE DECISIVE TEST**

This isolates the graph's **intrinsic** robustness: train on holo only (the surface channel *cannot*
learn the perturbation from data), test on repack. This is the true holo→apo transfer the north star
cares about. If the graph carries a conformation-stable signal (design D-P2.3), surface_only should
collapse while the graph cells hold. From `logs/p2_noaug/phase2_results.json` (3 seeds, 200 steps):

| cell | holo_rn | repk_rn | DEG_rn | holo_nm | repk_nm | DEG_nm (pool) | DEG_nm (median) |
|---|---|---|---|---|---|---|---|
| surface_only | 0.870 | 0.861 | +0.009 | 0.862 | 0.827 | +0.034±.002 | +0.029±.002 |
| +covalent | 0.878 | 0.858 | +0.020 | 0.841 | 0.808 | +0.033±.004 | +0.030±.006 |
| +rotatability | 0.876 | 0.856 | +0.020 | 0.832 | 0.799 | +0.033±.003 | +0.030±.007 |
| +spatial | 0.873 | 0.853 | +0.020 | 0.840 | 0.800 | +0.040±.005 | +0.026±.001 |
| full | 0.874 | 0.844 | +0.030 | 0.811 | 0.772 | +0.039±.005 | +0.029±.008 |
| _raw mean-pool_ | _0.876_ | _0.840_ | _+0.036_ | _0.897_ | _0.854_ | _+0.043_ | — |

**The decisive read — the graph provides NO intrinsic robustness:**
- Without augmentation, **every graph cell degrades the same or *more* than surface_only** on negmix
  (spatial +0.040, full +0.039 vs surface +0.034 pooled; medians all ~+0.029). The graph does **not**
  reduce the holo→apo collapse.
- **Absolute apo performance is *worse* with the graph:** repack_nm surface_only 0.827 > spatial
  0.800 > full 0.772 (and all < raw mean-pool 0.854). Adding the graph makes the apo representation
  worse, not better.
- **Therefore the floor run's apparent graph benefit (§3) was an augmentation artifact** — the model
  learned the *specific* FASPR perturbation from seeing repacked twins in training, not a
  conformation-invariant signal from the graph. Remove augmentation → the benefit vanishes.
- **do-no-harm still FAILS** and is worse: full holo_nm 0.811 ≪ raw 0.897. The graph *harms* holo
  training (more capacity to overfit the tiny train set, away from the near-optimal frozen desc).

This is the load-bearing negative result: **the design's mechanism (D-P2.3 covalent-anchored readout
→ conformational robustness) is not realized in the true holo→apo transfer.** (Confirmed at N=29 in
§5.)

---

## 5. Scaled ablation — N=29 (6 additional repacked complexes), 17 train / 12 test

Repacked 1A2W/1A99/1ACB/1AGQ/1AK4/1AN1 (all qualified, ≥8 intersection positives). Larger, cleaner
test set (raw natural collapse here is **larger**: Δnm +0.056). This is the confirmation run.

**5a. With augmentation (p_aug=0.5)** — `logs/p2_scaled_aug`:

| cell | holo_nm | repk_nm | DEG_nm (pool) | DEG_nm (median) | paired vs surf (help/hurt, p) |
|---|---|---|---|---|---|
| surface_only | 0.888 | **0.865** | +0.024 | +0.015 | — |
| +covalent | 0.869 | 0.838 | +0.032 | +0.010 | 4/8, p=.39 |
| +rotatability | 0.867 | 0.835 | +0.031 | +0.005 | 5/7, p=.77 |
| +spatial | 0.852 | 0.832 | +0.021 | +0.000 | 8/3, p=.23 |
| full | 0.846 | 0.831 | **+0.015** | −0.007 | **9/3, p=.15** |
| _raw mean-pool_ | _0.942_ | _**0.886**_ | _+0.056_ | — | — |

**5b. No augmentation (p_aug=0.0) — the honest holo→apo transfer** — `logs/p2_scaled_noaug`:

| cell | holo_nm | repk_nm | DEG_nm (pool) | DEG_nm (median) | paired vs surf (help/hurt, p) |
|---|---|---|---|---|---|
| surface_only | 0.849 | **0.847** | **+0.002** | +0.002 | — |
| +covalent | 0.825 | 0.796 | +0.029 | +0.019 | 4/8, p=.39 |
| +rotatability | 0.827 | 0.803 | +0.023 | +0.016 | 4/8, p=.39 |
| +spatial | 0.819 | 0.799 | +0.020 | +0.025 | 4/8, p=.39 |
| full | 0.807 | 0.766 | +0.041 | +0.044 | **3/9, p=.15 (graph HURTS)** |
| _raw mean-pool_ | _0.942_ | _**0.886**_ | _+0.056_ | — | — |

**Reads — the verdict crystallizes:**
- **The differential-degradation metric is confounded, so read absolute apo instead.** A trained head
  lowers *holo* (0.942→0.849) to sit near repack, shrinking the gap without improving apo. Under
  augmentation `full` "wins" on pooled DEG (+0.015 < surface +0.024) **only** because it depressed
  holo more — its **absolute apo (0.831) is still worse than surface_only (0.865) and far below raw
  (0.886)**. Low differential ≠ good; a constant predictor has zero differential. **Absolute apo
  AUC (`repk_nm`) is the honest measure.**
- **Absolute apo ranking is invariant across every run: `raw mean-pool > surface_only > graph`.**
  N=29 aug: raw 0.886 > surf 0.865 > spatial 0.832 > full 0.831. N=29 noaug: raw 0.886 > surf 0.847
  > spatial 0.799 > full 0.766. The graph is **always worst on apo**.
- **The frozen descriptor is already conformation-robust once a trivial head is trained.** In the
  honest transfer (noaug), surface_only collapses only **+0.002** on negmix (per-complex Δ balanced
  −0.07..+0.11 → symmetric noise, not systematic loss). The raw +0.056 fragility is almost entirely
  recoverable **without any graph**, just by learning a projection of the frozen descriptor.
- **The graph injects conformation-sensitivity.** In noaug, full degrades **+0.041** (per-complex Δ
  skewed positive: 1AK4 +0.11, 2HEY +0.13, 3HTR +0.09, 3B5U +0.06) and helps only **3/12** complexes
  (mean paired −0.039). Because the graph readout mixes spatial-distance edges that *move* with the
  repacked rotamers, the fused embedding becomes *more* pose-dependent, not less — the opposite of the
  D-P2.3 mechanism.
- **Augmented "benefit" is (a) not significant (best p=0.15) and (b) an augmentation artifact** — it
  vanishes and reverses without augmentation. It is the model memorising the specific FASPR
  perturbation, not a conformation-invariant signal.
- **Do-no-harm FAILS everywhere** (full holo_nm 0.807–0.846 ≪ raw 0.942; even surface_only < raw).

---

## 6. Which edge features carry robustness + the T4 check

There is **no robustness to attribute** — no edge set improves absolute apo AUC over surface-only in
any run. On the (confounded) differential metric the pattern is also unstable: augmented N=23 ranked
spatial best and chemistry redundant (full > spatial); augmented N=29 ranked full best; noaug N=29
ranked spatial least-bad but all graph cells worse than surface-only. This instability across
N/seed/augmentation is itself the signal: the "differences" are noise around a null (or negative)
effect, not a reproducible edge-feature contribution.

**T4 (redundant-chemistry risk) — realised, and then some.** Covalent/rotatability edges never add
robustness beyond spatial, and the combined `full` graph is the **worst apo performer**. The
chemistry signal the thesis hoped for (bond rotatability → "how sidechains can move" → invariance) is
not extractable by this architecture at this scale; the spatial edges that were supposed to help
instead couple the readout to the (moved) rotamer geometry.

---

## 7. GO / NO-GO for Phase 3 + D1-B trigger

**Gate (design §5):** do-no-harm on holo **AND** full-model degrades significantly less than
surface-only under the apo-like repack, with an attributable edge-feature contribution.

**Result vs gate:**
| gate criterion | outcome |
|---|---|
| Do-no-harm: full holo ≥ frozen mean-pool baseline | **FAIL** — every trained head < raw mean-pool on holo (full 0.807–0.846 nm vs raw 0.942); the graph lowers it further. |
| Decisive: full degrades *significantly* less than surface-only under repack | **FAIL** — best case (aug N=29) is 9/12 complexes, sign-p=0.15 (n.s.); vanishes/reverses without augmentation; and it is confounded by holo depression. |
| Attributable robustness from rotatability/spatial edges | **FAIL** — no stable attribution; on absolute apo the graph is always worst. |

### → **NO-GO for Phase 3.**

The atom graph, as designed (heterogeneous covalent+rotatability+spatial edges, message-passing GNN,
descriptor fused at readout), does **not** deliver conformational robustness on the Phase-2 proxy. In
the honest holo→apo transfer it makes apo performance *worse*. Do not build Phase 3 (learned pose
scorer, aligner hardening, ligands, true-apo training) on this representation.

### D1-B trigger decision: **NOT triggered (contra the naïve reading).**

Design §5 says trigger D1-B (atom-centric surface retrain) "if the graph cannot improve robustness
**and** the ablation localizes the ceiling to the frozen descriptor." The first clause holds; the
**second does not**. The ablation localizes the ceiling to the **read-out / fusion approach and the
choice of perturbation**, *not* to the frozen descriptor:
- A **trivial trained head on the frozen descriptor already recovers the repack robustness**
  (raw collapse +0.056 → surface-only transfer collapse +0.002). The descriptor *contains* enough
  conformation-invariant information; retraining it (expensive, GPU) is not what's missing.
- The learned head nonetheless **underperforms raw mean-pooling on absolute AUC** (holo and apo) —
  a *training/metric-mismatch* problem (contrastive margin loss vs L2-AUC eval; ≤17 train complexes),
  not a descriptor-capacity problem.

**Recommended next step (supersedes D1-B):** the FASPR **fixed-backbone** repack is **too mild** a
proxy — a linear head absorbs it, so it cannot discriminate any representation. Re-scope the gate to a
**harder, more apo-realistic perturbation** (backbone-perturbed rotamer sampling, unbound/apo crystal
pairs where available, or AF2 monomer models) **before** investing in either the graph or D1-B. If
robustness is pursued, (i) drop rotamer-moved spatial edges from the graph readout (they add
pose-sensitivity) and (ii) fix the head so it does not lose to raw pooling (align loss to the eval
metric; more data). Hold **D1-B as contingency** only if a harder perturbation shows the *frozen
descriptor itself* (not the head) collapsing irrecoverably.

---

## 8. Decisions, assumptions, and what was NOT tested

**Decisions (rationale in `docs/progress/phase2-log.md`):** D-P2.1 pure-torch relational GNN (no
PyG); D-P2.2 biotite template connectivity (primary) over RDKit-from-PDB (fragile on protonated
benchmark PDBs); D-P2.3 descriptor fused at readout only (kept out of message passing); D-P2.5
holo↔repack atom identity mapping; repacked eval restricted to atoms present in both states.

**Assumptions / caveats.**
- **FASPR fixed-backbone repack is a proxy for apo — and turned out too mild.** It perturbs only
  sidechain rotamers with the backbone frozen; a trained linear head on the frozen descriptor absorbs
  it almost entirely (transfer collapse +0.002). It therefore **cannot discriminate** representations
  and is the *wrong* gate for the robustness thesis. The real holo→apo gap (backbone motion, larger
  rearrangements) is untested and is the indicated next perturbation (§7).
- **The "differential degradation vs surface-only" metric is confounded.** It rewards any model that
  depresses holo toward its repack value (a constant predictor scores a perfect 0 gap). It must be
  read **together with absolute apo AUC** (`repk_nm`); this report ranks on absolute apo, where the
  conclusion is unambiguous and stable. This is a correction to the design's §5 metric choice.
- **Small N, sparse positives, underpowered significance.** sc-complementary contacts are sparse
  (~29 of ~56 preprocessed complexes have ≥5); no cell comparison reaches sign-test p<0.05 (best
  0.15). Pooled numbers are outlier-sensitive at N=23 (3B5U); the N=29 median/paired stats are the
  more reliable read and agree with the verdict.
- **CPU-only floor.** Kuma GPU was **not** used — the bottleneck is CPU surface preprocessing, not
  GNN training (the GNN is tiny; trains in seconds–minutes on CPU). GPU would not change the science
  at this scale; it would only enlarge N faster. No budget spent (§ below).
- **Representation-only** (no learned pose scorer; D7 → Phase 3).
- **Head underperforms raw pooling.** The contrastive fusion head loses to raw mean-pooling on both
  holo and apo at ≤17 train complexes — a training/metric-mismatch that must be fixed before any
  representation claim; it is orthogonal to the graph question but caps all trained cells.

**What was NOT tested:** true apo/AF2 structures; ligands/neosurfaces; a fusion that preserves holo
(the current head degrades holo vs raw pooling); N ≫ 30; sequence-identity clustering of the split
(split is by complex id, not by cluster). **Native-contact recovery / top-k retrieval** (design §5's
secondary check) was **not** run — descriptor-separation AUC (randneg + negmix) is the
representation-stage metric and already answers the gate; retrieval would only compound the same
absolute-apo ranking (graph worst) and belongs to the Phase-3 pose/retrieval scope that this NO-GO
defers. Do-no-harm anchor used: the **Phase-1 atom mean-pool baseline = 0.889 randneg / 0.916 negmix**
(`docs/02-phase1-results.md`); all trained cells fall below it on holo.

**Cumulative CHF spent: 0** (Jed serial-QOS CPU job only; no Kuma / GPU submissions). Budget ceiling
CHF 100 never approached.

---

## 9. Reproduce

```bash
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
# floor (augmented, N=23):
$PY -m masif_graph.experiments.run_phase2 --ids logs/p2_floor/ids.txt --out logs/p2_floor \
    --steps 200 --seeds 3 --min-pos 5 --test-min-pos 8 --split-frac 0.6 --p-aug 0.5
# no-augmentation transfer:  … --out logs/p2_noaug   --p-aug 0.0
# scaled (N=29):             … --ids logs/p2_scaled/ids.txt --out logs/p2_scaled_aug   --p-aug 0.5
$PY -m masif_graph.experiments.analyze_phase2 floor=logs/p2_floor noaug=logs/p2_noaug \
    scaled_aug=logs/p2_scaled_aug scaled_noaug=logs/p2_scaled_noaug
```
