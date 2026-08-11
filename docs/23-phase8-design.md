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
signal is evaluation**: `P(A and B form a biologically meaningful assembly)`. Evaluation subsumes
retrieval — rank by P — so Phases 4–7 are not discarded, they gain a better-defined output.

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
| bio-vs-crystal distinction | **not visible** | not visible | **this is its job** |
| conservation | no (D8-5) | no | yes, as interface aggregates |
| molecule type | agnostic (shared 26-D atom space) | agnostic | type-aware head allowed |

**Why this factorisation is the load-bearing idea.** Bio-vs-crystal is *not* a property of a pair of
molecules — it is a property of a **relative pose**. Chains A and B can form a biological interface
at one surface and a lattice contact at another. So the distinction can only live at Stage 3.
Correspondingly, a lattice contact is a perfectly valid **atom-level** positive: the local
complementarity is real physics. This resolves the labelling question cleanly and, as a bonus,
quarantines the buried-surface-area confound (§5) to Stage 3 alone.

## 3. Labels — an ordinal 3-class scheme, parallel across molecule types

| class | protein–protein | protein–ligand |
|---|---|---|
| **2** biologically meaningful | interface present in the biological assembly | functional ligand (substrate / cofactor / inhibitor / drug) |
| **1** compatible but not biological | crystal-lattice contact | **crystallisation additive** (glycerol, PEG, sulfate, MPD, DMSO, acetate, Tris…) or a ligand at a non-functional site |
| **0** incompatible | non-interacting pair | ligand against a random surface patch |

Class 1 is the point of the scheme: a **real, measurable, genuinely complementary contact that is not
biology** — precisely the false-positive mode the project exists to kill.

**Label sources** (to verify on `/work` before mining):
* PPI class 2/1 — the PDB's own `_pdbx_struct_assembly.details`. Prefer
  `author_and_software_defined_assembly` as high-confidence class 2; software-only as a lower tier,
  reported separately. PISA is a *predictor*, not ground truth, and is not treated as such.
* P–L class 2/1 — **BioLiP** (curated biologically-relevant ligands, crystallisation additives
  filtered) gives class 2; the complement within the same structures gives class 1.
* Class 0 — two tiers, reported separately: **(0a) random pairs** (easy, abundant) and **(0b)
  same-organism, co-expressed, non-interacting pairs** (hard, closer to the screening distribution).

**Crystal contacts, staged:** start with **asymmetric-unit chain pairs not in the same biological
assembly** — no symmetry expansion, immediately available. Scale later to full lattice contacts via
symmetry mates (gemmi) once the pipeline and labels are validated.

## 4. Design decisions

**D8-1 — Objective.** `P(biologically meaningful interaction)`, ordinal over {0,1,2}. Not affinity.
Retrieval is the deployment mode, obtained by ranking on P.

**D8-2 — Three-stage funnel** as in §2. Stage 1 is the existing encoder; Stages 2 and 3 are new.

**D8-3 — Class usage by stage.** Stage 1 trains on classes **2 and 1 together** as atom-level
contact positives (pose-free; local complementarity is real in both). Stage 3 discriminates 2 vs 1
vs 0. Under no circumstances does Stage 1 receive the bio/crystal label.

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
else**. Class-1 examples are sampled to match the class-2 BSA distribution; every Stage-3 number is
reported **stratified by BSA**; and **"rank by interface area" is an explicit baseline that Stage 3
must beat.** This is the shuffled-label control of this phase.

**D8-8 — Apo inputs come from AFDB/TED wherever possible**, not from generating AF3 ourselves.
Measured cost of self-generation is ~3.3 core-h MSA + 60 s GPU per chain (≈ CHF 250 for 10k chains);
AFDB is free and we already hold **39,637 TED domain surfaces** built. AF3 generation is reserved for
coverage gaps and for the multi-seed ensembles of D8-9.

**D8-9 — Per-atom tolerance via probabilistic embeddings.** Stage 1 emits a mean **and a learned
per-atom variance σ**; complementarity scoring becomes variance-aware, so flexible atoms tolerate
mismatch and rigid atoms demand precision. σ is supervised by observed conformational spread:
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

## 5. Metrics — the funnel is evaluated stage-wise, never as one number

| stage | metric | gate |
|---|---|---|
| 1 | **recall**: does the true interface survive the atom-level screen? | must not regress vs the Phase-5/6C encoder |
| 2 | pose accuracy (DockQ / interface-RMSD vs native), **reported separately for holo and apo inputs** | the holo→apo gap here is the north-star number for this stage |
| 3 | **2-vs-1 AUC at low FPR**, BSA-stratified | must beat the **BSA-only baseline** (D8-7) |
| end-to-end | screening enrichment on a held-out benchmark | — |

Reporting only the end-to-end number would hide which stage fails. Chance lines, shuffled controls
and per-item spread are reported throughout, per `ml-research-guardrails`.

## 6. Stages and their gates

**Stage A — diagnostics (cheap, and either can invalidate the plan). Do these first.**
* **A1 — sidechain sensitivity.** Is the current encoder modelling flexibility, or *ignoring
  sidechains*? Use FASPR repacks: how much do embeddings and PPI retrieval change under sidechain
  scrambling vs the true apo? Correlate per-atom embedding sensitivity with B-factor/pLDDT.
  *Why it matters:* if the encoder is backbone-reading, its atom×atom matrix cannot support pose
  prediction, and the bio-vs-crystal distinction — which lives in fine surface complementarity —
  is unreachable. **This gates everything downstream.** ~CHF 2.
* **A2 — cryptic-pocket probe.** On the 298 AF3-apo models already built, compare pocket
  volume/openness at the known ligand site vs holo. *Why:* for P–L, apo→holo is often
  **backbone-scale** (closed/cryptic pockets), which no amount of per-atom sidechain tolerance
  recovers. This bounds what any model can achieve from apo input. ~CHF 1.
* **A3 — Stage-2 rigid baseline.** RANSAC/ICP on the existing Stage-1 scores; pose accuracy on holo
  and apo. Zero training cost, establishes the starting point. ~CHF 2.

**Stage B — label mining.** ASU-only crystal contacts + biological assemblies (PPI) and
BioLiP-derived P–L classes; BSA-matched; cluster-clean splits on sequence and scaffold; the
BSA-only baseline computed. *Gate:* ≥5k class-2 and ≥5k BSA-matched class-1 examples surviving
clustering, or the scheme is reconsidered.

**Stage C — Stage 3 scorer on rigid poses.** Train the pose-level classifier on Stage-A3 poses, with
conservation aggregates (D8-5). *Gate:* beats the BSA-only baseline on 2-vs-1 AUC at low FPR,
**and** beats it in the small-interface BSA stratum.

**Stage D — tolerance mechanism (D8-9) + differentiable Stage 2 (D8-10).** *Gate:* variance-aware
scoring improves apo pose accuracy and apo 2-vs-1 discrimination over the rigid/point-embedding
baseline, at ≥2 seeds.

**Stage E — the conservation ablation (D8-5b)** and a deployment dry-run: a known glue system
(6H0F: CRBN+pomalidomide → IKZF1 ZF2; composite neosurface **already built** in Phase 7) queried
against the TED domainome. Reality check on the real operating point, not a training target.

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
