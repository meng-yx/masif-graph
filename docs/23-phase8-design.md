# Phase 8 — design contract: apo-native evaluation of biomolecular interactions

> Design-ahead-of-code. Supersedes the partial north-star statements in `docs/00` §1 and `docs/16` §6,
> and **reconciles with `docs/11`** (Phase-4 Stage-C), which specified a three-stage funnel before
> Stage 1 existed. Conditioned on the Phase-7 results (`docs/21`). Agreed with the user 2026-08-11.

## 0. North star (authoritative restatement)

Train a **generalizable model that evaluates biomolecular interactions from apo / predicted (AF2/AF3)
structures, without requiring knowledge of the induced-fit holo conformation.**

* A binding partner may be a **protein, a protein complex, a small molecule, a protein–small-molecule
  complex**, and later a **nucleic acid**.
* The model should learn — **per atom, conditioned on local environment** — *how much shape and
  chemical mismatch is tolerable*, so that it carries an implied **latent conformational landscape**
  reachable from the apo structure. Some atoms are rigid and demand precise complementarity; others
  are flexible and should tolerate mismatch.

**Deployment mode is retrieval/screening** (rank candidate partners for a target). **The training
signal is evaluation**: `P(A and B form a biologically meaningful assembly)`, factorised over two
binary labels (§3). Evaluation subsumes retrieval — rank by P — so Phases 4–7 are not discarded,
they gain a better-defined output.

**Explicitly not the objective:** predicting Kd/Ki. Affinity regression is welcome if it helps, but
1 µM vs 1 nM is not what we are trying to predict.

**Molecular glue is the last deployment target, not the first.** Known glue systems are too few to
train on, and Phase 7 showed glue is *not* the composition of PPI and P–L: real composite
protein+drug surfaces did not move the neosurface number (composite 337 ± 2 vs separate 347 ± 15,
both below the ~298 chance line). It is a third relation and will need its own training data.

## 1. What Phases 1–7 established, and how it constrains this design

| finding | consequence for Phase 8 |
|---|---|
| Holo→apo robustness **achieved** (Phase 5 gate met; Phase 7 axis 4: drops +0.007/+0.014 on the ligand axis) | The precondition is met. Phase 8 builds on it rather than re-proving it. |
| The **atom/chem graph gave five independent nulls** (Phases 2→7) | The method thesis of `docs/00` §2 is **not earned**. Robustness came from invariant features + the contrastive recipe. Do not invest further in chemistry-graph elaboration. |
| **Two representation upgrades failed the capacity gate** (26-D unified atoms, then full ligand surfaces): train-set P–L retrieval flat at ~0.11 | The bottleneck is not the ligand representation. It is the **objective and the label** — which is what this phase changes. |
| Phase 7 **capacity competition**: adding ligand surfaces cost PPI −0.169 *and* hurt PPI training | Multi-molecule-type in one encoder needs an architecture/capacity answer. Do not add features to the shared encoder without one. |
| One seed lies (the Phase-6C −0.041 gap vanished at 2 seeds) | ≥2 seeds per condition remains mandatory. |

## 1.5 Training-corpus composition — CURRENT STATE AND THE GAP

**Every model trained in Phases 1–7 saw holo (crystal) structures only.** All apo data has lived on
the evaluation side. This was never stated in a results summary, and Phase 7's D7-7 deferred
apo-augmented training in a design doc without surfacing the decision. Recorded here so it is
contract, not inference.

| corpus | holo | AI-predicted apo | where apo was used |
|---|---|---|---|
| PPI training (`npz_ppi`) | 4,767 | **0** | — |
| PDBbind P–L (`npz_pl`) | 5,239 | 298 | held-out eval only |
| Phase-5 PPI eval (`npz_eval`) | 301 | 284 | eval only |

Consequence: **the holo→apo robustness reported in Phases 5 and 7 is zero-shot generalisation**, not
learned from apo examples. That is a stronger result than it sounds, but it also means the per-atom
tolerance of D8-9 has never had a training signal — you cannot learn "how much can this atom move"
from a corpus that shows every entity exactly once, in its bound state.

**D8-12 — Corpus composition (target).** Every holo chain is paired with AI-predicted apo
conformer(s). Ratio and source are **UNDECIDED pending Stage A0** (§6) and the user's decision:
* **Option A (1:1)** — AFDB where an acceptable sequence match exists, local prediction otherwise.
  AFDB has one model per sequence, so the ratio is capped at 1:1. Its advantage is that an AFDB model
  is **distribution-matched to deployment** if the screening database is AFDB/TED.
* **Option B (1:N, N≈5)** — locally predict apo conformers for every holo chain. Only this can supply
  the **conformational spread** that D8-9's σ supervision needs; AFDB alone cannot.
These are complementary, not exclusive: AFDB for deployment realism, local ensembles for the σ signal.

**Cost note that changes the arithmetic:** the MSA dominates (~3.3 core-h/chain) and is paid once per
chain regardless of how many conformers are sampled, while AF3's `num_diffusion_samples` reuses the
trunk. **Five conformers therefore cost ~1.5–2× one conformer, not 5×.** Phase 7's use of
`NSAMP=1` was a mistake on those economics.

## 2. Architecture — a three-stage funnel

```
Stage 1  atom-level encoder            → per-atom embeddings → atom×atom complementarity matrix
         (pose-free, molecule-agnostic)   ** EXISTS (Phase 4/5/6C) **
Stage 2  pose prediction               → a binding mode from the complementarity matrix
Stage 3  pose-level scorer             → P(biologically meaningful) for that mode
```

Directly analogous to the reference system (masif-ppi-search patch scoring → RANSAC alignment →
alignment-evaluation NN), and to `docs/11` §2. **Difference from `docs/11`:** its Stage 3 was
*restraint-guided co-folding*; here Stage 3 is a **learned pose scorer**, with co-folding retained
as a later, more expensive precision tier (out of scope, §9).

**Why the funnel is necessary** (`docs/11` §0, unchanged and still the governing argument): *any two
proteins have some complementary atom pairs by chance.* A real interface is a **collective,
geometrically coherent, sufficiently large, physically realisable** set of them. Stage 1 alone cannot
express that, which is why per-atom retrieval plateaus.

### 2.1 What each stage is allowed to see

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| pose | none (pose-free) | produces it | consumes it |
| `biological` label | **not visible** | not visible | **this is its job** (trained on `comp=1` only) |
| conservation | no (D8-5) | no | yes, as interface aggregates |
| molecule type | agnostic (shared 26-D atom space) | agnostic | type-aware head allowed |

**Why this factorisation is the load-bearing idea.** Bio-vs-crystal is *not* a property of a pair of
molecules — it is a property of a **relative pose**. Chains A and B can form a biological interface
at one surface and a lattice contact at another. So the distinction can only live at Stage 3.
Correspondingly, a lattice contact is a perfectly valid **atom-level** positive: the local
complementarity is real physics. This resolves the labelling question cleanly and, as a bonus,
quarantines the buried-surface-area confound (§5) to Stage 3 alone.

## 3. Labels — TWO BINARY LABELS, not an ordinal

Superseded design (2026-08-11, user): an earlier draft used a single ordinal `{0,1,2}`. That was a
design error, and the tell is that **no single stage could own it** — it smeared a Stages-1/2
property and a Stage-3 property onto one axis, contradicting the factorisation of §2.1.

```
complementary_contact ∈ {0,1}    "do these surfaces fit?"          -> Stages 1-2
biological_contact    ∈ {0,1}    "given that they fit, is it real?" -> Stage 3
```

with the hard implication **`biological ⇒ complementary`**. Three states are observable and the
fourth is impossible by construction:

| state | protein–protein | protein–ligand |
|---|---|---|
| `comp=1, bio=1` | interface present in the biological assembly | functional ligand (substrate / cofactor / inhibitor / drug) |
| `comp=1, bio=0` | crystal-lattice contact | **crystallisation additive** (glycerol, PEG, sulfate, MPD, DMSO, acetate, Tris…) or a ligand at a non-functional site |
| `comp=0, bio=0` | non-interacting pair | ligand against a random surface patch |
| `comp=0, bio=1` | — impossible — | — impossible — |

`comp=1, bio=0` is the point of the scheme: a **real, measurable, genuinely complementary contact
that is not biology** — precisely the false-positive mode the project exists to kill.

### 3.1 Why two binaries beat the ordinal

On observed data the encodings are isomorphic, so this is not about expressiveness:

1. **It gives the funnel exact probabilistic semantics.** Because `bio ⇒ comp`,
   **`P(biological) = P(complementary) · P(biological | complementary)`** — Stages 1–2 estimate the
   first factor, Stage 3 the second, and their **product is the deployment score**. The funnel stops
   being an engineering pipeline and becomes a factorisation of the quantity we actually want.
2. **It decouples the training data, which is the practical win.** *Every* crystal contact, additive
   and biological interface is a **complementarity** positive, so that head is not bottlenecked by
   assembly-annotation coverage; only the `bio` head needs curated labels. Under the ordinal, the
   scarce label would have throttled everything.
3. **No false metric.** An ordinal loss asserts that `0→1` and `1→2` lie on one axis at comparable
   distance. We have no reason to believe that. Two Bernoullis assert nothing.
4. **Interpretable failure at deployment.** "Doesn't fit" and "fits but isn't biology" are different
   answers, and only the second deserves a human's attention.

### 3.2 `complementary_contact` is defined at the INTERFACE/POSE level, not the atom level

`docs/11`'s governing argument is that *any two proteins have some complementary atom pairs by
chance*, so `comp=0` would be a **noisy, near-meaningless label at the atom level**. At the pose
level it is well-posed: *does a sufficiently large, geometrically coherent, clash-free complementary
interface exist?* — which is exactly what Stage 2 produces and can be scored on. Stage 1 keeps its
existing atom-level contrastive objective and is never asked to emit this binary.

**Label sources** (to verify on `/work` before mining):
* PPI `bio` — the PDB's own `_pdbx_struct_assembly.details`. Prefer
  `author_and_software_defined_assembly` as the high-confidence positive; software-only as a lower
  tier, reported separately. PISA is a *predictor*, not ground truth, and is not treated as such.
* P–L `bio` — **BioLiP** (curated biologically-relevant ligands, crystallisation additives filtered)
  gives `bio=1`; the complement within the same structures gives `comp=1, bio=0`.
* `comp=0` — two tiers, reported separately: **(a) random pairs** (easy, abundant) and **(b)
  same-organism, co-expressed, non-interacting pairs** (hard, closer to the screening distribution).

**Crystal contacts, staged:** start with **asymmetric-unit chain pairs not in the same biological
assembly** — no symmetry expansion, immediately available. Scale later to full lattice contacts via
symmetry mates (gemmi) once the pipeline and labels are validated.

## 4. Design decisions

**D8-1 — Objective.** Two binary labels, `complementary_contact` and `biological_contact`, with
`biological ⇒ complementary` (§3). The deployment score is the factorisation
**`P(biological) = P(complementary) · P(biological | complementary)`**. Not affinity. Retrieval is
the deployment mode, obtained by ranking on `P(biological)`.
**The constraint is enforced by construction, not by penalty**: the model predicts `P(comp)` and
`P(bio | comp)` and multiplies, which makes `P(bio) ≤ P(comp)` structurally true. Predicting the two
independently and adding a consistency loss would be strictly worse.

**D8-2 — Three-stage funnel** as in §2. Stage 1 is the existing encoder; Stages 2 and 3 are new.

**D8-3 — Label usage by stage.** Stage 1 trains on **all `comp=1` contacts together** — biological
and crystal alike — as atom-level contact positives (pose-free; the local complementarity is real
physics in both). **Under no circumstances does Stage 1 or Stage 2 receive the `biological` label.**
**Stage 3 trains ONLY on `comp=1` examples**, i.e. {crystal contacts, biological interfaces}: that is
the correct conditional likelihood `P(bio | comp)`. `comp=0` pairs go to Stages 1–2 only — if Stage 3
saw them it would spend capacity re-learning complementarity and could cheat via easy negatives.
A useful side effect: the BSA-matching discipline of D8-7 then applies exactly and only where it
belongs.

**D8-4 — Molecule-type generality via the shared 26-D atom space** (`p6/atoms.py`), unchanged.
Nucleic acids deferred to Phase 9+.

**D8-5 — Evolutionary conservation: INCLUDED, at Stage 3 only, as interface-level aggregates**
(mean interface conservation relative to that chain's surface background, and conservation
enrichment). Not a Stage-1 atom channel.
*Rationale:* (i) bio-vs-crystal is a Stage-3 question and conservation is the literature's strongest
discriminator for it (EPPIC-class methods); (ii) it sidesteps the ligand asymmetry entirely — the
ligand has no evolutionary history, but the *pocket* does, and functional sites are conserved while
adventitious ones are not; (iii) Phase 7 showed adding asymmetric, capacity-hungry features to the
shared encoder costs −0.169 on PPI, so new features are paid for only where they are used.
Conservation comes free from the MSAs our AF3 pipeline already computes (~750 s/chain, 298 already
on disk) and from AFDB.
**D8-5b — A structure-only ablation is SCHEDULED, not optional** (Stage E). We must know how much of
Stage 3 is conservation and how much is our representation, and we must be able to state the
structure-only ceiling honestly. If conservation dominates, we benchmark against EPPIC-class methods
rather than claiming novelty.

**D8-6 — Forbidden features.** No ligand identity, no HET-code frequency, no "this ligand appears in
40,000 structures" prior. Such a feature would score well and be worthless in deployment — it is a
lookup table, fails on novel chemotypes, and novel chemotypes *are* the molecular-glue use case.
Splits hold out **protein sequence clusters AND ligand chemotypes** (Murcko scaffolds, as in
Phase 6C; the Phase-7 effect was *larger* on the scaffold-unseen subset, so the discipline pays).

**D8-7 — BSA-matched sampling and a BSA-only baseline gate.** Biological interfaces are on average
larger than lattice contacts, so a model can score well by learning **buried surface area and nothing
else**. `comp=1,bio=0` examples are sampled to match the `bio=1` BSA distribution; every Stage-3 number is
reported **stratified by BSA**; and **"rank by interface area" is an explicit baseline that Stage 3
must beat.** This is the shuffled-label control of this phase.

**D8-8 — Apo inputs come from AFDB/TED wherever possible**, not from generating AF3 ourselves.
Measured cost of self-generation is ~3.3 core-h MSA + 60 s GPU per chain (≈ CHF 250 for 10k chains);
AFDB is free and we already hold **39,637 TED domain surfaces** built. AF3 generation is reserved for
coverage gaps and for the multi-seed ensembles of D8-9.

**D8-9 — Per-atom tolerance via probabilistic embeddings.** Stage 1 emits a mean **and a learned
per-atom variance σ**; complementarity scoring becomes variance-aware, so flexible atoms tolerate
mismatch and rigid atoms demand precision. σ is supervised **primarily by the apo-substituted pairs of D8-14** (a real interface whose apo
partners fit imperfectly is a direct lesson in tolerable mismatch), and secondarily by observed
conformational spread:
**AF3 multi-seed ensembles** (`run_m2_ensemble.py` already measured inter-sample CA-RMSD of ~0.1 Å
for confident chains up to ~15 Å for uncertain ones), **FASPR fixed-backbone repacks**
(`scripts/repack_one.sh`), and pLDDT/PAE as priors.
*Rationale:* a static-structure contrastive loss cannot teach "how much can this atom move"; the
signal has to be supplied. This is the phase's research core and its highest-risk element.

**D8-10 — Stage 2 differentiability.** Hard RANSAC/Kabsch (`align/global_align.py`, already written)
is the **baseline**; a soft-correspondence (Sinkhorn/OT) → weighted-Kabsch variant is the **target**,
because only a differentiable Stage 2 lets Stage 3's biological signal reach Stage 1. "End-to-end
fine-tuning beats frozen Stage 1" is an explicit gate, not an assumption.

**D8-11 — Seeds.** ≥2 per condition for every claim, per Phase-7 (D7-6).

**D8-15 — ONE global split, defined before Stage B, applied to every stage.** Stage 1's encoder sees
interfaces; if Stage 3's evaluation set overlaps Stage 1's training data, **leakage flows through the
encoder** and no per-stage split can detect it. Phase 5 already shipped a leak at the complex level;
this is the same failure one architectural level up.
**Clustering is at the INTERFACE level, not the chain level.** The same protein crystallises many
times, so its lattice contacts recur as near-duplicates across entries; chain-level sequence
clustering will not remove them and a "held-out" set full of near-copies reports inflated numbers.
Cluster on (sequence-cluster pair + interface similarity), and verify against the *actual* training
ids as in Phase 6C.

**D8-16 — Guard against label circularity.** PDB assembly annotations are partly software-derived
(PISA-like). Training on them and then claiming to beat PISA-class methods would be circular. Train
on the broad set, but reserve a **manually curated / author-defined-only benchmark** (PiQSi, the DC
set) — untouched by training — for the headline number.

**D8-17 — Two baselines, not one.** The **BSA-only** baseline (D8-7) is the shuffled-control: it
proves we learned *something*. A **published bio-vs-crystal classifier** (EPPIC-class) is the
comparator: it says whether what we learned is worth anything. Given D8-5b may show conservation
dominates, we need the field's bar on the same benchmark, not just our own.

**D8-18 — Deployment-realistic metrics; AUC is not the headline.** Screening ~40k domains for 1–10
true partners means operating at FPR ~1e-4, where AUC is nearly uninformative. Headline numbers are
**enrichment factor at 0.1% and 1%, and precision@k**. Correspondingly **Stage 1's recall gate has an
operating point**: recall@top-1% of a 40k-entry database, because that is the shortlist Stage 2 can
afford to process (§6.5).

**D8-19 — σ must be shown to be meaningful, not merely present.** The learned per-atom σ is validated
against independent flexibility measures — B-factors, ensemble RMSF, pLDDT — before any claim that
the model has learned adaptive tolerance. Without this, σ can silently absorb noise and "adaptive
tolerance" becomes a story rather than a finding.

**D8-14 — Apo enters training by CONFORMER SUBSTITUTION into crystal-derived poses.** An earlier
draft of this document claimed Stage 3's training data was "crystal-derived by necessity" and that
apo could therefore only be an evaluation state. **That was wrong** (user, 2026-08-11). The label
comes from the crystal, but the *structure* need not: superpose each partner's apo model onto its
holo chain, keep the crystal relative pose, and recompute contacts — the interface label transfers.
This works for biological interfaces and lattice contacts alike.

**The machinery already exists.** `p4.dataset.ComplexP4B` (Phase-4 M2) loads holo *and* AF3 graphs
for both chains and precomputes positives remapped into all four combinations —
`(holo,holo)`, `(af3,holo)`, `(holo,af3)`, `(af3,af3)` — joined on `(chain, resseq, name)`, with a
`min_retention` filter so a badly-mismatched apo model is never injected as a positive. Phase 7 did
the same on the ligand side (`p7/pl_af3.py`; contact ratio AF3/holo median **0.99**, zero failures).
**The barrier to apo-augmented training has always been data, never implementation.**

Consequences:
* The apo generation run **precedes Stage C**, it is not deferred to Stage D. This front-loads the
  CHF 150–560.
* `holo–holo → apo–holo → apo–apo` is a difficulty ladder that doubles as augmentation, with
  **apo–apo being the deployment condition**.
* **This is the primary tolerance signal, demoting ensemble-σ to a secondary source.** An apo–apo
  pair in a crystal pose has clashes and gaps the holo pair does not, and the label still says *real
  interface*. That is task-coupled supervision for "how much mismatch is tolerable here", tied to the
  actual decision rather than to a proxy for flexibility.
* **Mandatory QC:** report the contact-retention distribution for every substituted pair, never
  assume it. Where a global CA superposition misplaces the interface (large domain motion),
  fall back to interface-local alignment; retention is what detects this.

**D8-13 — The Phase-7 ligand surface is carried as an A/B variable inside Phase 8, not re-tested as
a separate phase.** Phase 7's primary verdict (a ligand surface is not a capacity fix) is **not**
confounded by holo-only training: that gate was *train-set* retrieval measured on holo data the model
had been trained on, so apo augmentation cannot explain a failure to fit it. What remains genuinely
open is second-order — whether the surface pays off *once tolerance is being learned*. Since Phase 8
rebuilds the corpus with apo conformers regardless, that question costs one extra training arm here
instead of a whole re-run of Phase 7.

## 5. Metrics — the funnel is evaluated stage-wise, never as one number

| stage | metric | gate |
|---|---|---|
| 1 | **recall@top-1% of a 40k-entry DB** — the shortlist Stage 2 can afford | must not regress vs the Phase-5/6C encoder |
| 2 | pose accuracy (DockQ / interface-RMSD vs native), **reported separately for holo and apo inputs** | the holo→apo gap here is the north-star number for this stage |
| 2 | pose quality → **P(complementary)**: AUC vs `comp=0` pairs (both tiers, reported separately) | must beat a contact-count / size heuristic |
| 3 | **`bio` vs `crystal`, conditioned on `comp=1`**: enrichment@0.1% / @1%, precision@k, BSA-stratified (AUC reported but not the headline, D8-18) | must beat **BSA-only** (D8-7) **and** be situated against a published classifier (D8-17), on the curated held-out benchmark (D8-16) |
| σ (Stage D) | correlation with B-factors / ensemble RMSF / pLDDT | D8-19 — no adaptive-tolerance claim without it |
| end-to-end | screening enrichment@0.1%/1% on the curated held-out benchmark | — |

Reporting only the end-to-end number would hide which stage fails. Chance lines, shuffled controls
and per-item spread are reported throughout, per `ml-research-guardrails`.

## 6. Stages and their gates

**Stage A0 — BENCHMARK THE AI-PREDICTION METHODS. This is the first thing built, and the
phase PAUSES at its end.**

Choose how apo conformers are generated before generating ~14,700 chains' worth of them. Candidates:
**AF3**, **Chai-1** (installed; supports an MSA-free mode), **Boltz-2** (can reuse AF3 MSAs).
**ESMFold is disqualified for the ensemble role** — it is deterministic, one structure per sequence,
so it cannot supply conformational spread at any price; it may still be benchmarked as a cheap 1:1
option under Option A.

Run all candidates on ~30 chains for which we already hold both holo structures and AF3 models, and
report:
1. **wall time and CHF per chain, MSA and inference separately** (MSA dominates and is amortised
   across conformers);
2. **accuracy vs holo** — TM-score, interface RMSD;
3. **inter-sample conformational spread, and whether it tracks real flexibility.** This is the metric
   most easily forgotten and the one we are actually buying: five near-identical models are useless
   for σ however accurate, and five wildly divergent ones are equally useless. Calibrated spread is
   the deliverable. Compare against B-factors/pLDDT and, where the PDB holds multiple entries for the
   same protein, against genuine experimental conformational variation.
4. **AFDB coverage** — what fraction of our holo chains have an acceptable AFDB sequence match, and
   what the residual (mutants, designed and artificial constructs) looks like.

> **PAUSE HERE.** Stage A0 produces a recommendation, not a decision. The user chooses the method and
> the holo:apo ratio (D8-12) before the **apo generation run** and Stages C–E. Estimated Stage-A0
> cost ~CHF 5; the generation run that follows is ~CHF 150–560 depending on the choice, which is why
> it is not made unilaterally.

### Critical path — A0 does NOT block the diagnostics or the mining

Stages A1–A4 use only data we already hold (holo, plus the 284 PPI + 298 P–L apo models), and
Stage B is pure PDB parsing. **None of them depends on A0's outcome**, so serialising them behind the
pause would waste days for nothing:

```
A0  benchmark prediction methods ┐
A1  sidechain sensitivity        │
A2  cryptic pockets              ├─ ALL IN PARALLEL ─→ PAUSE (user decides D8-12)
A3  rigid pose baseline          │                          │
A4  signal-existence probe       │                          ↓
B   label mining                 ┘                   apo generation ─→ C ─→ D ─→ E
```

By the time the method decision is made we will already know whether the encoder is sidechain-blind,
whether pockets survive in apo, what rigid pose accuracy looks like, and whether any bio-vs-crystal
signal exists at all.

**Stage A — diagnostics (cheap; several can invalidate the plan). Run in parallel with A0.**
* **A1 — sidechain sensitivity.** Is the current encoder modelling flexibility, or *ignoring
  sidechains*? Use FASPR repacks: how much do embeddings and PPI retrieval change under sidechain
  scrambling vs the true apo? Correlate per-atom embedding sensitivity with B-factor/pLDDT.
  *Why it matters:* if the encoder is backbone-reading, its atom×atom matrix cannot support pose
  prediction, and the bio-vs-crystal distinction — which lives in fine surface complementarity —
  is unreachable. **This gates everything downstream.** ~CHF 2.
  *Contingency if A1 fails* — this is no longer fatal, because **D8-14 supplies the remedy**: if the
  encoder is robust *by discarding* sidechains, training on apo–apo pairs whose label demands
  discrimination **despite** sidechain differences forces it to model conformational variation rather
  than ignore it. A1 failing therefore promotes the apo-substituted corpus from augmentation to
  prerequisite, and Stage 1 is retrained before Stages C–D rather than the plan being abandoned.
* **A2 — cryptic-pocket probe.** On the 298 AF3-apo models already built, compare pocket
  volume/openness at the known ligand site vs holo. *Why:* for P–L, apo→holo is often
  **backbone-scale** (closed/cryptic pockets), which no amount of per-atom sidechain tolerance
  recovers. This bounds what any model can achieve from apo input. ~CHF 1.
* **A3 — Stage-2 rigid baseline.** RANSAC/ICP on the existing Stage-1 scores; pose accuracy on holo
  and apo. Zero training cost, establishes the starting point. **Also measures seconds-per-pair**,
  which feeds the deployment-compute fork of §6.5. ~CHF 2.
* **A4 — signal-existence probe.** On a few hundred mined interfaces, compute simple aggregates from
  the *existing* Stage-1 embeddings (score distribution, contact count, coherence) and fit a
  **logistic regression** against the BSA-only baseline. *Why:* if a linear model on current
  embeddings cannot beat interface area, the funnel's premise is in doubt — and learning that here
  costs ~CHF 1 instead of discovering it after Stage C. Not a gate to pass, a warning to heed. ~CHF 1.

**Stage B — label mining.** ASU-only crystal contacts + biological assemblies (PPI) and
BioLiP-derived P–L classes; BSA-matched; **the ONE global interface-level split of D8-15 defined
here and frozen**; the curated held-out benchmark of D8-16 set aside untouched; BSA-only (D8-7) and
the literature baseline (D8-17) computed. *Gate:* ≥5k `bio=1` and ≥5k BSA-matched `comp=1,bio=0` interfaces
surviving clustering, or the scheme is reconsidered.

**Stage C — Stage 3 scorer on rigid poses.** Train the pose-level classifier on Stage-A3 poses, with
conservation aggregates (D8-5), trained on `comp=1` only. *Gate:* beats the BSA-only baseline on
`bio`-vs-`crystal` AUC at low FPR, **and** beats it in the small-interface BSA stratum.

**Stage D — tolerance mechanism (D8-9) + differentiable Stage 2 (D8-10).** *Gate:* variance-aware
scoring improves apo pose accuracy and apo `bio`-vs-`crystal` discrimination over the rigid/point-embedding
baseline, at ≥2 seeds.

**Stage E — the conservation ablation (D8-5b)**, the σ-validity check (D8-19), and a deployment
dry-run: a known glue system
(6H0F: CRBN+pomalidomide → IKZF1 ZF2; composite neosurface **already built** in Phase 7) queried
against the TED domainome. Reality check on the real operating point, not a training target.

> **Stage A has its own implementation plan: `docs/24-phase8A-plan.md`** (test sets, methods,
> metrics, job shapes, autonomy boundary, fallbacks). Results land in `docs/25-phase8A-results.md`.

## 6.5 Open forks — decisions NOT yet made, each owned by the stage that needs it

These are named so they cannot be resolved silently, which is the failure mode that produced D8-12
and D8-14.

**F1 — "protein complex" as a partner type.** The north star (§0) explicitly allows a partner to be a
*protein complex*, and nothing in this design addresses it. Is a multi-chain partner one graph with
several chains, or a composed representation? Affects Stage 1's input contract.
*Owner:* must be closed before the corpus is built, since it changes what a "chain pair" is.

**F2 — Stage 3's input representation.** "Consumes the embedding of the predicted pose" is not a
spec. Candidates: (a) an interface-graph GNN over the aligned atoms of both partners with cross
edges; (b) aggregate interface descriptors; (c) both. Whatever is chosen must **not** be handed
BSA-equivalent features naively, or D8-7's control becomes untestable.
*Owner:* Stage C. Informed by A4 — if simple aggregates already carry the signal, (b) may suffice.

**F3 — Stage 2's compute budget at deployment scale.** Unbudgeted, and it can invalidate the
architecture independently of accuracy: at 1 s/pair a 40k-domain screen is ~11 h per target, at
60 s/pair it is ~27 days. **A3 measures seconds-per-pair**, which closes this fork cheaply and early.
If Stage 2 is too slow, either the Stage-1 shortlist tightens (raising the recall bar of D8-18) or
Stage 2 needs a cheaper formulation.

**F4 — What counts as an acceptable AFDB sequence match** (exact, ≥95% identity over the observed
range, truncation policy for constructs and tags). *Owner:* Stage A0, item 4.

### 6.6 Cost envelope

| item | ≈ CHF |
|---|---|
| A0 benchmark | 5 |
| A1–A4 diagnostics + probe | 6 |
| B label mining | 5 |
| **apo generation run** (decision-dependent) | **150–560** |
| C Stage-3 scorer, ≥2 seeds | 15 |
| D tolerance + differentiable Stage 2, ≥2 seeds | 30 |
| E ablation + dry-run | 10 |
| | **≈ 220–630** |

The generation run dominates and is the only item requiring a decision before it is spent, which is
what the A0 pause exists for.

## 7. Risks

* **R1 — the encoder is sidechain-blind.** Would invalidate Stage 1 as a foundation. Tested first (A1).
* **R2 — cryptic pockets** put a hard ceiling on apo P–L screening (A2).
* **R3 — BSA is the whole signal.** Controlled by D8-7; the gate is explicit.
* **R4 — conservation is the whole signal.** Controlled by the scheduled ablation D8-5b.
* **R5 — bio-vs-crystal may not be learnable from structure alone.** Literature says conservation
  dominates; we may find a low structure-only ceiling. That is a publishable negative, and D8-5b is
  designed to state it honestly.
* **R6 — capacity.** Phase 7 showed the shared encoder is already contended. Adding σ heads and more
  molecule types will worsen it; a two-tower/adapter or wider encoder may be forced.
* **R7 — assembly annotations are noisy.** Mitigated by the author+software confidence tiering (§3).

## 8. Explicitly NOT in scope for Phase 8

* Affinity (Kd/Ki) prediction as an objective.
* Molecular glue as a **training** target (benchmark/inference only until P–L works).
* Nucleic acids.
* Restraint-guided co-folding (`docs/11` Stage 3) — the expensive precision tier comes after a
  learned pose scorer is shown to work.
* Any further elaboration of the chemistry graph (five nulls; see §1).
