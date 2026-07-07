# Phase 3 — Results (living; honest, absolute + differential, controls, per-complex spread)

> Deliverable for arc M0–M2. Every number traces to a committed artifact + recoverable command.
> "Pipeline ran" is stated separately from "result is valid." Cumulative CHF logged.

**Status:** ✅ **ARC M0–M2 COMPLETE + self-verified** (2026-07-06). Numbers trace to `logs/phase3/m1_full/*.json`
(`m1_results.json`, `m1_strata.json`, `m1_mismatch.json`), `logs/phase3/m2_full/m2_plddt.json`,
`logs/phase3/m2_ens_full/m2_ensemble.json`. Reproduce via `run_m1_af3` / `run_m1_analyze` / `run_m1_mismatch`
/ `run_m2_plddt` / `run_m2_ensemble` on `logs/phase3/m1_ids.txt`. **M3 is gated on the spend checkpoint in
`06-phase3-user-comment.md` — NOT started.**

**Cumulative CHF spent:** ≈ **3** (Kuma H100 ≈ 1.5 across all inference/smoke jobs + Jed CPU MSA ≈ 1–2).
Far under the CHF-100 arc budget. Kuma job-ids logged in `phase3-log.md`.

---

## Arc deliverable checklist (from handoff §1)
- [x] (a) The measured holo→AF3 gap with controls (absolute AF3 metrics + gap + per-complex spread +
      retention + addressable/unaddressable stratification + shuffled controls + positive control).
- [x] (b) ≥1 tested robustness hypothesis with an honest verdict (TWO: pLDDT-weighting; ensemble soft-min).
- [x] (c) A concrete recommendation + the earned-scale (M3) plan (AtomSurf learnable encoder; checkpointed).

---

## M0 — infra smoke — **PASS** (all four tools run)

| tool | status | evidence |
|---|---|---|
| AF3 `.sif` (a) | ✅ end-to-end | MSA 11 min (Jed, 1AY7_A) + inference 68 s (Kuma H100 job 3786680, rc=0), 5-sample ensemble + pLDDT written |
| AtomSurf (b) | ✅ code+GPU (forward staged) | pkg imports on CPU; `get_default_model`→ProteinEncoder 150k params; H100 torch smoke (job 3786616) OK. Full forward-on-PDB deferred to M2-if-chosen (needs MSMS/pymesh; it's the escalation lever) |
| MaSIF `.sif` (c) | ✅ consumption path | `io.reference.load_complex("1AY7_A_B")` loads 80-D descriptors+atoms; 104 holo complexes already computed |
| Kuma sbatch+cost (d) | ✅ | `sbatch --test-only` prints cost; **1×H100 = CHF 0.52/hr**; Chai env imports OK |

**AF3 timings (project M1 cost):** MSA ≈ 11 min/chain (Jed CPU, mgy 128G dominant); inference ≈ 1 min/chain
(H100, 5 samples). **Relabel** (AF3 mmCIF → holo-numbered PDB) verified 1:1 on 1AY7_A, so AF3 models plug
into the Phase-2 identity-mapping machinery. **Cumulative CHF ≈ 0.1.**

## M1 — the measurement: holo→AF3 gap

**Setup:** 31 Phase-2 complexes; frozen mean-pooled MaSIF descriptor; regimes holo→holo (ceiling),
AF3→holo (deployment), AF3→AF3. Intersection-positive set (surface atoms in both states, identity-mapped
after relabelling AF3 to holo numbering). Frame-free negatives (randneg + cross-complex); absolute AUC is
the headline. AF3 = 1 seed × 5 diffusion samples, top-ranked model.

**DEFINITIVE AF3→holo (full set: 30 complexes generated, N=18 usable with ≥8 intersection positives;
3 seeds; controls valid):**
| regime | randneg pooled | randneg per-cplx median | cross pooled | cross median | shuffled |
|---|---|---|---|---|---|
| holo→holo (ceiling) | **0.902** | 0.908 | **0.914** | 0.934 | 0.48 |
| **AF3→holo (deployment)** | **0.821** | **0.846** | **0.834** | **0.866** | 0.50 |
| AF3→AF3 | 0.766 | 0.817 | 0.789 | 0.837 | 0.51 |
| **gap holo→AF3** | **+0.081** | +0.062 | +0.079 | +0.068 | — |

**The gap is real and moderate: ~+0.08 pooled, ~+0.06 per-complex median.** Absolute AF3→holo ≈ 0.82 pooled
/ 0.85–0.87 median — clearly above chance but a meaningful drop from the ~0.90 ceiling. (Larger than the
early N≈10 estimate of ~0.03 median — the full set includes more divergent complexes; trust N=18.)

**Strata (N=18) — moderate, NOT the near-perfect small-N values.** gap_mean +0.083, gap_median +0.057.
- **corr(gap, pLDDT) = −0.62**; **corr(gap, interface-local Cα-RMSD) = +0.38** (whole-chain fit +0.37).
- ⚠️ **Honesty correction:** at small N (5–10) these read −0.94 / +0.92, but that was inflated by leverage
  points (esp. 1JXQ). At N=18 the relationships are **real but moderate** — lower AF3 confidence / larger
  conformational deviation → larger gap, but with substantial scatter. Do not overclaim −0.94.
- Interface-local superposition (user guidance) is still the honest RMSD stratifier (1JXQ: local 2.4 Å vs
  whole-chain 16.7 Å — a domain-motion artifact removed).

**Two gap components (the honest read, N=18/30):**
1. **Descriptor degradation** on preserved surface atoms: pooled +0.081 / median +0.06. Real, moderate.
2. **Interface-atom divergence (NEW vs Phase-2):** holo interface atoms that stop being AF3 surface atoms.
   Retention mean **0.78**, atom-weighted **0.80**, median 0.91, min 0.00; **5 of 30 complexes lose >50%**
   (1A2W 0, 2AOB 0, 4UDM 0.33, 2Z0E 0.38, 3B5U 0.40). Phase-2's FASPR froze the backbone → retention ≈ 1.0
   always, so this axis is genuinely new. The intersection-AUC (component 1) has a **survivorship bias** (it
   excludes the most-divergent atoms), so +0.08 is a **lower bound** on the true deployment gap.

**Addressable vs unaddressable (structural-mismatch stratification; per user guidance).** Not all of the
gap is a *descriptor* problem. Some AF3 monomers are **structural-mismatch** — domain-swapped / context-
dependent folds where the binding-competent conformation is physically absent (e.g. 1A2W: N-terminal helix
swaps to the partner in the holo dimer; AF3, given a lone monomer, folds it back). No descriptor can fix
these; MaSIF *should* treat them as non-binders. Detector (structure-fixed thresholds): a chain is mismatch
if interface-atom **retention < 0.5** (exposure loss) OR interface-**local** Cα-RMSD **> 4.0 Å** (interface
backbone in a fundamentally different conformation). Complex = mismatch if either chain is. **1A2W is the
verified positive control (flagged; PASS).**
- **7/30 complexes (23%) are structural-mismatch** (1A2W, 2AOB, 2IWP, 2PZD, 2Z0E, 3B5U, 4UDM) — 5 by
  exposure loss, 2 (2IWP 6.5 Å, 2PZD 7.9 Å) by local geometry.
- **Gap, both reported (honesty guardrail):** UNFILTERED (N=18) **+0.075**; **INDUCED-FIT-ONLY (N=16, the
  ADDRESSABLE gap) +0.069**. The two are close because mismatch cases mostly self-exclude via low retention
  (only 2PZD, 3B5U survive the ≥8-intersection bar). So the intersection-AUC gap is *already* mostly the
  addressable induced-fit gap; ~23% of AF3 monomers are unaddressable structural-mismatch (→ non-binders).
- The interface-local view also **correctly clears 1JXQ** (interface-local 2.4 Å → induced-fit): its 16.7 Å
  whole-chain motion is *away* from the interface, so its large gap is genuine descriptor sensitivity, not a
  fold error (corrects my earlier read).

**Secondary metric — top-k RETRIEVAL (the deployment-shaped test; `run_m1_retrieval`, N=18, DB=36 holo
chains).** Query = an AF3 chain's interface descriptors searched against a database of holo chains; does the
true holo partner rank near the top?
| query | top-1 | top-5 | top-10 | MRR | median rank |
|---|---|---|---|---|---|
| holo (ceiling) | 0.50 | 0.78 | 0.92 | 0.63 | 2 |
| **AF3 (deployment)** | **0.44** | **0.64** | **0.81** | **0.55** | **2** |
| drop | −0.06 | **−0.14** | −0.11 | −0.08 | 0 |

AF3-query retrieval is **degraded but not broken**: top-5 recall drops 0.78→0.64 (−14 pts), but the median
true-partner rank stays **2** — the correct partner usually still ranks near the top. Concrete deployment
cost, consistent in magnitude with the ~+0.08 AUC gap. (Descriptor-only matching, no MaSIF-site gating or
geometry, small DB → absolute recalls are modest even at the ceiling; the holo-vs-AF3 *drop* is the signal.)

## M2 — first robustness hypothesis: pLDDT-weighted matching (lever-0)

Motivated by the M1 strata (corr(gap, pLDDT) = −0.62 at N=18). Test: gate the AF3→holo match to positives
whose AF3 query atom pLDDT ≥ T; measure absolute AF3 AUC + same-subset gap. **DEFINITIVE (N=18):**
| pLDDT gate | kept | af3→holo AUC | hh AUC | gap |
|---|---|---|---|---|
| none | 100% | 0.809 | 0.902 | +0.093 |
| ≥80 | 71% | 0.843 | 0.901 | +0.058 |
| ≥90 | 50% | 0.851 | 0.891 | +0.039 |
| ≥95 | 40% | 0.857 | 0.896 | +0.039 |

**Verdict: a real but PARTIAL, test-time-only lever.** pLDDT-gating confirms the gap partly lives in
low-confidence atoms (gap +0.093→+0.039, absolute AF3 AUC 0.809→0.857), but "recovers" it only by
discarding **50–60% of the interface** — it shrinks retrieval coverage, plateaus at gap ≈ +0.04 (doesn't
close), and doesn't fix the representation. Consistent with Phase-2 lesson #3: post-processing a frozen
descriptor has limited headroom; the real lever is an unfrozen/learnable encoder (AtomSurf) → M3.
(N=7 preliminary read +0.083→+0.013 was small-N-optimistic; trust N=18.)

### M2 lever-1: multi-sample ensemble soft-min matching (the better free lever)
Motivated by M1: AF3's 5 diffusion samples span the conformational uncertainty exactly where the gap is
worst (uncertain chains: inter-sample CA-RMSD up to 15.5 Å; confident chains ~0.1 Å). Test: represent the
AF3 query atom by its best-matching diffusion sample (min descriptor distance to target, applied to
positives AND negatives for fairness). **N=10 (samples 0/1/2):**
| | AUC |
|---|---|
| holo ceiling | 0.913 |
| single-sample AF3 | 0.855 |
| **ensemble AF3** | **0.875** |

Gap to ceiling **+0.058 → +0.038** (~34% reduction), **lossless** (discards no atoms — unlike lever-0). Can
also recover interface atoms the single top-ranked model lost (an atom need only be surface in ≥1 sample).
**Verdict: real, lossless, modest.** Better than lever-0 (which must drop ~57% of atoms); smaller raw
magnitude. Neither lever closes the gap. (3-sample; 5-sample untested, likely a touch better.)

## Verdict + recommendation + earned-scale plan (DEFINITIVE, N=18 usable of 30)

**(a) The holo→AF3 gap, measured with controls.** Real and **moderate**. Absolute AF3→holo descriptor-
separation AUC ≈ **0.82 pooled / 0.85–0.87 median** vs a **0.90–0.91** holo ceiling → **gap +0.081 pooled
(3 seeds) / +0.06 per-complex median**. Shuffled controls 0.48–0.51; complex-level; no holo/AF3 leakage;
per-complex spread reported. **Three refinements make it honest:**
- *Two components:* (1) descriptor degradation on preserved surface atoms (+0.08 pooled / +0.06 median),
  and (2) a NEW-vs-Phase-2 axis — **interface-atom divergence** (retention 0.78–0.80; AF3's backbone shift
  moves ~20% of interface atoms off-surface; 5/30 complexes lose >50%).
- *Addressable vs unaddressable:* **23% of AF3 monomers are structural-mismatch** (domain-swap/context-
  dependent fold, e.g. 1A2W — unaddressable by any descriptor → correct to treat as non-binders). The
  **addressable induced-fit gap is +0.069** (N=16); the unaddressable cases mostly self-exclude via low
  retention. Both filtered/unfiltered reported; 1A2W is a verified positive control.
- *Strata are moderate, not the small-N mirage:* corr(gap, pLDDT) = **−0.62**, corr(gap, interface-local
  RMSD) = **+0.38** at N=18 (the −0.94/+0.92 at N≈7 were inflated by leverage points). Real but scattered
  relationships; the deployment risk is the low-confidence / high-deviation tail, not the median (most
  interfaces are predicted locally close, 0.1–2.4 Å, and the descriptor holds up).

**(b) Tested robustness hypotheses (two, honest).** Both are training-free, test-time levers:
- **pLDDT-weighting:** real but partial; halves→closes the gap only by discarding ~57% of interface atoms.
- **Ensemble soft-min over AF3 samples:** real, lossless, modest (gap 0.058→0.038); the better free lever.
- **Neither closes the gap** → confirms Phase-2 lesson #3: a **frozen rigid-holo descriptor has limited
  headroom**; post-processing cannot fully fix conformational fragility.

**(c) Recommendation + earned-scale plan.** The evidence points to the same lever Phase-2 flagged: an
**unfrozen / learnable surface encoder trained for conformation-invariance (AtomSurf)** — M0 verified its
env + H100 path. **M3 (staged, checkpoint-gated):** (1) pilot on ~300–500 training complexes, contrastive
objective pulling holo & AF3(multi-seed) descriptors of the *same* interface atom together / different
interfaces apart, complex-level holdout + untouched PDBBindplus cross-check, judged on this M1 benchmark
(gate: learnable-AF3→holo beats frozen-AF3→holo absolute AUC and approaches ceiling without breaking
holo→holo); (2) full scale only if the pilot passes. **Cost (measured): pilot ≈ CHF 40–50, full ≈ CHF
120–200** — both exceed the CHF-100/48h arc contract → **checkpoint posted; awaiting user go before any M3
spend.** Also adopt **interface-anchored docking** at scale (user guidance) and consider 5-sample ensembles.
See `05-phase3-design.md §M3` for the full plan + anti-circularity.

---

## M3 PILOT — learnable conformation-invariant encoder (AtomSurf DiffusionNet ⊕ chem graph)

**User GO for the pilot** with a key steer: AtomSurf's atom graph is distance-only (no bond chemistry), so
**write our own chemistry graph** (Phase-2 covalent connectivity + bond order + rotatability + electroneg/
valence) and reuse only AtomSurf's learnable **DiffusionNet surface encoder** (the unfreezing lever). Full
analysis: `05-phase3-design.md §M3 ENCODER + GRAPH DECISION`.

**Architecture (`src/masif_graph/m3/`):** per-vertex frozen 80-D desc → DiffusionNet → pool to surface
atoms ⊕ chem-graph GNN (covalent-only, invariant) → fuse → **RESIDUAL** output = normalize(pooled_frozen +
refinement), refinement zero-init ⇒ starts EXACTLY at the frozen baseline (af3→holo 0.821) and learns a
refinement. Losses: complementarity (holo contacts) + **invariance** (af3 atom → holo twin) + refinement-
magnitude penalty (anti-overfit anchor). Train on 72 holo-ready complexes **disjoint from the 30-complex
M1 eval set** (clean complex-level holdout); structural-mismatch complexes excluded from training positives.

**Dynamics de-risk (14-train/10-held-out split of eval-30) — an honest early finding:**
- Without regularization the encoder **overfits**: val af3 0.80→0.89 but **held-out af3 0.72 (−0.11)**, hh
  collapses. With the refinement penalty, held-out no longer collapses but **just reproduces frozen**
  (delta +0.001…+0.006). At ~14–20 train complexes the frozen descriptor's headroom is **not accessible**.
- The 57-train pilot (4× data) is the real test. _[pilot sweep running; results below]_

**PILOT RESULTS (57-complex train, held-out eval-30; frozen-normalized baseline af3=0.801).** The
**regularization strength is the key knob** (mean held-out delta over seeds, all vs frozen 0.801):
| reg-weight | held-out af3→holo delta | stability | hh |
|---|---|---|---|
| ≥1 (strong anchor) | +0.001 | stable (reproduces frozen) | 0.89 |
| 0.1 | +0.007 | 2/3 seeds + | 0.89 |
| **0.05 (sweet spot)** | **+0.014** (all 3 seeds +: .009/.020/.013) | **stable** | 0.90 (preserved) |
| 0 (no anchor) | +0.016 mean but **unstable** (2/4 seeds +0.032, 2/4 = 0) | unstable | 0.88 |

**Modest but ROBUST positive:** with reg=0.05 the learnable encoder beats the frozen baseline on held-out
AF3→holo by **~+0.014** (all seeds positive), **without harming holo→holo**. Best single run +0.020 (0.821,
= M1 raw-frozen). Data-scale trend is strong: reg=0 went from **−0.11 (14 complexes, overfit) → +0.03/0
(52)** — more data reduces overfitting. So the frozen headroom IS partly accessible by unfreezing, and the
trend indicates more data → bigger/robuster win. **→ scaling the training data (evidence-motivated + user
directive).** Graph-vs-no-graph ablation at reg=0.05 + the data scale-up are in progress.

**SCALED RESULT (128 train, disjoint from eval-30; frozen-normalized af3=0.801) — DEFINITIVE M3.**
| config (reg=0.05) | seeds | held-out af3→holo delta | hh | verdict |
|---|---|---|---|---|
| **learnable encoder + chem graph** | 8 | **+0.016** (median +0.017, std .009, 8/8 positive) | 0.90 (preserved) | robust modest win |
| learnable, **no chem graph** (ablation) | 3 | +0.016 | 0.89 | indistinguishable from graph |

**M3 VERDICT — a modest but ROBUST positive; frozen descriptor remains a strong ceiling.**
1. **Unfreezing works (the core M3 hypothesis, Phase-2 lesson #3):** a learnable DiffusionNet surface
   encoder (residual on the frozen descriptor, contrastive holo↔AF3 invariance, reg=0.05) beats the frozen
   baseline on held-out AF3→holo by **+0.016** (8/8 seeds positive, hh preserved). Absolute 0.801→0.817.
2. **The chemistry graph adds NO clear benefit** — graph +0.016 (n=8) ≈ no-graph +0.016 (n=3). The
   *unfreezing* is the driver, not the bond-chemistry graph. (Honest test of the user's hypothesis: not
   supported at this scale. Phase-2 already found the graph's spatial edges inject pose-sensitivity;
   the invariant covalent chemistry, while principled, isn't the lever here.)
3. **Weak data-scaling:** 14→52 complexes escaped overfitting (−0.11→+0.014); 52→128 barely moved it
   (+0.014→+0.016). Diminishing returns — not a scaling law that would close the gap with more data.
4. **Ceiling stands:** M3 recovers only ~20% of the +0.08 gap; absolute AF3→holo 0.817 ≪ 0.90 holo ceiling.
   The frozen descriptor is remarkably strong; unfreezing buys a small, robust, but not gap-closing gain.

**RECOMMENDATION.** The deployment-ready levers remain: (a) accept the modest +0.016 from a learnable
encoder if the retraining cost is justified; (b) the training-free **ensemble soft-min** (M2, lossless
+0.020 gap-to-ceiling reduction) is comparable for far less effort; (c) the biggest wins are *upstream* —
filter structural-mismatch AF3 models (23%, non-binders) and use multi-seed/pLDDT confidence. A materially
gap-closing descriptor would need a fundamentally stronger objective/architecture than tested here, not just
more data. **Cumulative CHF ≈ 7 of 100.**
