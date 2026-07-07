# Phase 3 — Design / plan (living)

> North star: **make MaSIF surface-fingerprint PPI search robust to AF3-model queries** (database =
> holo crystals). This doc holds the evolving plan, hypotheses, and the *why*. Verdicts live in
> `07-phase3-results.md`; real-time reasoning in `phase3-log.md`.

## 0. Problem restatement (the deployment scenario)

MaSIF's 80-D descriptor is a **rigid readout of the holo atomic surface**. In deployment the query is an
**AF3 model** (unbound, predicted — backbone shifts + sidechain rotamers + prediction error), while the
searchable database is **holo crystal** surfaces. We must measure and then shrink the performance drop of
**AF3-query → holo-database** retrieval/separation relative to the **holo→holo** ceiling, *without*
harming holo→holo.

This supersedes Phase 2 (atom graph, NO-GO) and its too-mild FASPR proxy. AF3 monomer models are the
realistic, harder perturbation and — via multi-seed diffusion — give conformational ensembles for free.

## 1. Metrics (guardrail-bound)

- **Primary:** absolute **descriptor-separation ROC-AUC** for AF3-query vs holo, two negative schemes
  (`randneg` = confound-free sanity; `negmix` = spatially-hard, discriminative). Report AF3 **absolute**
  first, then the holo→AF3 gap. (Phase-2 lesson: differential alone is confounded.)
- **Secondary:** small **top-k retrieval** — AF3 query chain's interface descriptors vs a holo database of
  chains; does the true partner rank near the top? This is the deployment-shaped metric.
- **Stratifiers:** AF3 confidence (pLDDT / pTM) and conformational deviation (interface Cα-RMSD, all-atom
  RMSD to holo). Answers *where* the gap lives (low-pLDDT? high-RMSD interfaces?).
- **Per-complex spread** always (not just pooled) — one complex must not swing the verdict.

## 2. Controls (non-negotiable; `ml-research-guardrails`)

- **Shuffled-label ~0.5** reported every eval.
- **Complex-level holdout** for any M2 training; a complex's holo and AF3 twin never straddle the split
  (no holo/AF3 leakage). The **circular trap** (train on AF3 / eval on AF3 of the same complexes) killed by
  complex holdout + (if training) an untouched experimental cross-check subset.
- **≥3 AF3 seeds** where feasible; report which chain/complex a pooled number leans on.
- **"Pipeline ran" ≠ "result valid"** stated separately.

## 3. Milestones

### M0 — infra smoke (hours, ~free). Gate: each tool runs once.
- (a) AF3 `.sif` predicts one chain (MSA offline → model + confidence). **Time it** → projects M1 cost.
- (b) AtomSurf: locate/clone repo (Jed internet), one forward pass on H100.
- (c) MaSIF `.sif`: one holo surface+descriptor (largely proven Phase 1–2; quick re-verify).
- (d) Chai backup env sanity; confirm Kuma sbatch + cost print.

### M1 — THE MEASUREMENT (highest value, cheap-ish). Deliver a number: how bad is the gap, and where?
- Subset of `data/lists/testing.txt`: ~30–60 complexes with strong sc-contacts (reuse Phase-1 sc-filter;
  prefer complexes already holo-preprocessed to save compute).
- Per chain: extract the holo observed sequence → AF3 monomer model (single seed first, then a few seeds).
- Re-run reference descriptor pipeline on AF3 models under a parallel id (repack_one.sh pattern).
- Map holo contacts → AF3 atoms by **sequence position** (AF3 renumbers 1..N; input seq = holo observed
  residues in order → position i ↔ i-th holo observed residue). Intersection-positive eval (atoms present
  in both, like Phase 2).
- Compute absolute AF3 AUC (randneg+negmix), holo→holo, the gap, per-complex spread, stratified by
  pLDDT/RMSD; small top-k retrieval. Controls: shuffled ~0.5, no leakage.

### M2 — first improvement (hypothesize → test on M1). Cheapest levers first.
The lever chosen depends on *where* M1 says the gap lives (pLDDT? RMSD? which regime?). Ladder by cost:

0. **pLDDT-weighted matching (FREE — no extra compute).** Down-weight low-pLDDT query atoms in the
   AUC/retrieval, using the per-atom pLDDT already in the AF3 model. Tests "is the gap driven by
   low-confidence atoms?" Runs directly on the M1 top-ranked-model data. **First lever to try.**
1. **Multi-seed ensemble matching (cheap; needs 5-sample surfaces).** Represent the AF3 query atom by an
   aggregate over its 5 diffusion samples (mean descriptor / soft-min pos-distance / medoid). Tests "does
   averaging over the predicted conformational ensemble recover the holo signal?" **No training.**
2. **Flexibility-weighted surfaces (shares the 5-sample cost).** Down-weight atoms with high per-sample
   descriptor variance (unreliable regions). Reuses lever-1's 5-sample descriptors. **No training.**
3. **AtomSurf learnable encoder fine-tuned for conformation-invariance (expensive, GPU).** Contrastive:
   pull holo & AF3(multi-seed) descriptors of the *same* interface together, different interfaces apart.
   Likely the real lever (Phase-2 lesson #3: the frozen descriptor is the ceiling), but the most work —
   **gated on M1 signal + budget**; a full retrain is **M3 (checkpoint first)**.

**Shared cost note:** levers 1–2 both need the reference surface pipeline run on all 5 AF3 samples per
chain (≈5× the M1 surface compute; Jed-CPU-cheap but a few hours). Lever 0 needs nothing extra. So the M2
sequence is: lever-0 first (free), then decide whether the ensemble compute is warranted by the M1 gap.

Judge every lever on the M1 benchmark with absolute AF3 AUC (headline) + the holo→AF3 gap + per-complex
spread + shuffled control. **Anti-circularity:** levers 0–2 are *test-time* aggregations (no training on
AF3), so no train/test leakage; if lever-3 (AtomSurf) trains, hold out complexes by id and keep an
untouched experimental cross-check.

### M3 — earned scale (CHECKPOINT FIRST — exceeds CHF 100 / 48h).

**What M1/M2 earn (the case for M3):** the holo→AF3 gap is real, conformation-driven (corr(gap,pLDDT)≈−0.94,
corr(gap,RMSD)≈+0.84), and the free test-time lever (pLDDT-weighting) only *partially* mitigates it by
discarding atoms — confirming Phase-2 lesson #3 that a **frozen rigid-holo descriptor has limited headroom**.
The indicated real lever is an **unfrozen/learnable surface encoder trained for conformation-invariance**:
**AtomSurf** (H100-ready env verified in M0; code + GPU path confirmed).

**M3 proposal (staged):**
1. **Pilot (smaller spend):** generate AF3 models for ~300–500 training complexes (reuse this exact
   pipeline), fine-tune AtomSurf with a **contrastive conformation-invariance objective** — pull
   holo & AF3(multi-seed) descriptors of the *same* interface atom together, push different interfaces
   apart — on complex-level-held-out splits. Eval on the M1 benchmark (this held-out set) + an untouched
   experimental cross-check (PDBBindplus subset). Gate: does learnable-AF3→holo beat frozen-AF3→holo
   (absolute AUC) and approach the holo ceiling, without breaking holo→holo?
2. **Full scale (only if pilot passes):** full `training.txt` AF3 generation + retrain + full `testing.txt`
   benchmark + top-k retrieval.

**Cost projection (from measured M0/M1 numbers):**
- AF3 inference: ~1 min/chain on H100 = CHF 0.52/hr → ~**CHF 0.01/chain**. Pilot (≈800 chains) ≈ **CHF ~8**;
  full training.txt (≈9,900 chains) ≈ **CHF ~85** GPU (+ MSA on Jed CPU, cheap but wall-clock-heavy:
  ~11 min/chain, so full set needs a wide Jed array / days).
- AtomSurf training: H100, ~0.5–1 GPU-day/run ≈ **CHF ~6–12/run**; a few runs for tuning ≈ CHF ~30.
- **Pilot total ≈ CHF 40–50** (well-scoped); **full scale ≈ CHF 120–200** (mostly full-set AF3 + more
  training). Both exceed the CHF-100/48h arc contract for M0–M2 → **checkpoint before spending.**

**Anti-circularity for M3 (non-negotiable):** train and eval on **disjoint complexes**; never eval on a
complex whose holo or AF3 twin was in training; keep the PDBBindplus experimental subset untouched by
training as an independent check; report absolute AF3 AUC + shuffled control + per-complex spread.

**Data cleaning for M3 (per user guidance — structural-mismatch handling):** AF3 monomers that are
**structural-mismatch** (domain-swap / context-dependent fold; detector = interface-atom retention < 0.5 OR
interface-local Cα-RMSD > 4.0 Å, thresholds fixed from structure up front, 1A2W as positive control; see
`run_m1_mismatch.py`, ~23% of the M1 set) must NOT be used as training **positives** — forcing a contrastive
match to a physically-absent interface is label noise. Instead **keep them as AF3-side negatives / expected
non-retrievals** (the correct answer is "no confident match"; valuable for calibrating the decision
threshold, and honest about AF3's limits). **Always report both filtered (induced-fit) and unfiltered
numbers.** This makes the M3 gate measure the *addressable* gap (what a better descriptor can close) without
being corrupted by unaddressable structure-prediction failures.

### M3 ENCODER + GRAPH DECISION (user-directed AtomSurf evaluation, 2026-07-06) — JUDGMENT CALL

**Task (user):** read how AtomSurf builds its atom graph; decide reuse-vs-write-own before implementing;
the apo↔holo conformational landscape depends on molecular chemistry (bond order, element, electronegativity,
valence) that AtomSurf's graph may not capture.

**What AtomSurf actually does (read `atomsurf/protein/{atom_graph,graphs}.py`):**
- **Edges = pure distance.** `atom_coords_to_edges(pos, cutoff=4.5)` = KDTree `query_pairs(4.5 Å)`; every atom
  pair within 4.5 Å is an edge, attribute = distance only. A covalent C–C (1.5 Å) and an incidental 4 Å
  contact are the *same* edge type. **No bond order, no covalent connectivity, no rotatability.**
- **Nodes:** element one-hot (12), pdb2pqr charge, radius; residue-level amino-type/SSE/hydrophobicity.
  **No electronegativity, valence, hybridization, aromaticity, or bond-derived features.**
- **Verdict: the user is correct.** AtomSurf's atom graph is a distance/kNN spatial graph with element+charge
  nodes. It encodes geometry, not molecular bonding.

**Why this matters *more* than it first appears — the Phase-2 empirical lesson:** Phase-2's graph
(`src/masif_graph/graph/build.py`) already encodes exactly the missing chemistry — biotite-template
**covalent edges + bond-order one-hot + sidechain-rotatable flag + flex-depth + element/aromatic nodes**,
and its covalent topology is *invariant* to conformational change (bonds don't move). Phase-2 also found that
its **distance/spatial edges MOVED with the rotamers and INJECTED pose-sensitivity — making apo worse**.
AtomSurf uses *only* distance edges → the exact failure mode. So a distance-only graph is not merely poorer;
it is **actively wrong for conformational robustness**. The invariant, bond-based connectivity is the stable
anchor a conformation-invariant encoder needs.

**JUDGMENT CALL (decision):**
1. **Reuse AtomSurf's learnable SURFACE encoder (DiffusionNet) + its surface↔graph communication scaffold**
   (`SurfaceGraphCommunication`, `ProteinEncoder`). This is the *unfreezing lever* M3 is about — Phase-2
   used a FROZEN descriptor (the ceiling); AtomSurf gives a ready, trainable surface encoder, and rebuilding
   one (geodesic-CNN/DiffusionNet) from scratch is wasteful. This is AtomSurf's real value.
2. **Do NOT reuse AtomSurf's atom graph. Write our own = adapt Phase-2's chemistry-aware graph** (covalent
   connectivity + bond order + rotatability), enriched with element-derived **electronegativity / valence /
   hybridization** node features (cheap to add). This is the conformation-invariant chemistry anchor.
3. **Why this is NOT just Phase-2 again (which was NO-GO):** Phase-2 fused a chemistry graph with a **frozen**
   descriptor **at the readout** and it failed. M3 differs on the load-bearing axis: the surface encoder is
   **unfrozen and co-trained end-to-end** with a **contrastive conformation-invariance objective**, and the
   invariant graph informs the *learned* representation (not a late fusion). And we **drop the pose-sensitive
   distance edges** that hurt Phase-2, keeping bond-based invariant edges.
4. **Use AtomSurf as the initial test harness** for the learnable-surface component: first get its
   DiffusionNet surface encoder running on ONE of our MaSIF surfaces (integration smoke), then swap its
   distance atom-graph for our chemistry graph in the `SurfaceGraphCommunication`. **Fallback:** if AtomSurf's
   data-format/dependency integration proves heavier than a lean custom encoder, build a minimal learnable
   surface-feature encoder over the existing MaSIF surface + our graph. Decide at the integration smoke.

**Net:** AtomSurf-surface-encoder ⊕ our-chemistry-graph, unfrozen, contrastive holo↔AF3. This directly
incorporates the user's chemistry point and the Phase-2 pose-sensitivity lesson.

## 4. Open design decisions (resolve as they bind)
- **AF3 monomer vs complex prediction?** → **Monomer** (unbound predicted = realistic deployment query;
  also cheaper). Multi-seed for ensembles.
- **Whole-surface vs interface-only descriptors for the gap?** → report both; the deployment metric is
  interface-region (iface-gated) but whole-surface separation is the cleaner representation probe.
- **AF3 vs Chai/Protenix?** → AF3 primary (assets + multi-seed ensembles); Chai as a cross-check /
  fallback if AF3 wall-clock or MSA cost is prohibitive.
- **Which subset?** → prefer complexes already holo-preprocessed (Phase-1 40 + Phase-2 29) with ≥ enough
  sc-filtered intersection positives; strong-contact bias documented.

## 5. Risks specific to Phase 3
- **AF3 cost blow-out** — MSA is CPU-heavy; 60 chains × MSA could dominate wall-clock. Mitigation: measure
  one in M0, split MSA(Jed)↔inference(Kuma), start single-seed small subset.
- **Sequence-mapping bugs** (holo↔AF3 residue identity) — a silent mismap inflates/deflates the gap.
  Mitigation: build the map from the *exact* extracted holo sequence; assert 1:1; sanity-check on a
  known case; shuffled control catches gross leakage.
- **Confirmation bias toward "the gap is big"** — the handoff warns it may be smaller than assumed. Trust
  the measurement. A small gap is a legitimate, publishable finding (and would re-scope M2).
