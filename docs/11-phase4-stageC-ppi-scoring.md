# Phase 4 Stage-C (design draft) — atom-pairs → protein-level PPI, in the induced-fit regime

> **Status: design brainstorm (2026-07-07), to test AFTER the from-scratch GNN (Stage-A/B) is trained.**
> Not started. Extends `docs/08-phase4-design.md` §5.4 (protein-level aggregation) + reopens D5/D7
> (aligner/scorer). Grounded by two literature sweeps (citations at bottom). This is a *living* draft — the
> exact strategy is not locked; open questions are flagged.

## 0. The problem the GNN does NOT solve by itself

The Stage-A/B GNN gives a per-atom-pair complementarity score `s(X,Y)=zₓᵀ T z_Y`. Protein-level PPI —
*"do these two proteins bind?"* — is a different question, and the gap **is** the false-positive source:
**any two proteins have *some* complementary atom pairs by chance.** A real interface is not "a few good
contacts" but a **collective, geometrically-coherent, sufficiently-large, physically-realizable** set of them.

**Why MaSIF-Neosurf false-positives (root cause, from the lit — validates our escalation):** MaSIF is a
two-tier funnel — fast fingerprint NN screen → RANSAC rigid alignment → **IPA rescorer** (inputs per aligned
vertex pair: 1/3D-dist, normal·normal, 1/fingerprint-dist, a clash/penetration term). The IPA score is real
but computed on **one rigid pose of holo-trained descriptors**. Their own SARS-CoV-2 RBD case: surface
matching *favored the wrong orientation*, and they had to bring in **AlphaFold-Multimer to validate the pose**.
So the field's own SOTA already reaches for co-folding as the precision stage — we systematize that, with
restraint-guidance for induced fit. dMaSIF / PPIretrieval improve speed/recall but **do not** fix the FP
problem (consistent with our north star).

## 1. The three necessary conditions a protein-level scorer must test

A spurious complementary patch fails at least one of these; a real interface passes all three:
1. **Sufficiency/size** — enough matched contacts. Real interfaces bury **~1600 ± 400 Å² total (~800 Å²/
   partner)**; 52/75 complexes in that band (Lo Conte–Chothia–Janin 1999). A handful of contacts ≠ interface.
2. **Geometric coherence** — matched pairs mutually consistent under one (induced-fit-tolerant) transform.
   Random complementary pairs point inconsistently. *This is the strongest FP filter* and the induced-fit-
   correct version of MaSIF's single rigid RANSAC.
3. **Physical realizability** — clash-free, energetically plausible bound pose. For induced fit this **requires
   modeling** (sidechain/backbone relax); a rigid transform can't deliver it → clashes = MaSIF's pose FPs.

These map 1:1 onto the three cascade stages.

## 2. Architecture — a three-stage funnel (predict-then-model)

**Ordering decision: predict-then-model (cascade), not model-then-fold-everything.** Co-folding every
candidate defeats the point of a fast descriptor and doesn't scale. The GNN buys **recall at scale**;
co-folding buys **precision on a shortlist**. This is the canonical PPI-search / docking funnel (MaSIF's own
two-tier design; ZDOCK→ZRANK in docking).

### Stage 1 — GNN descriptor screen (RECALL, scales to millions)
Transform every DB atom once (`T z_d`), max-inner-product retrieval; shortlist candidate partners for query A.
Conformer-invariant (Stage-B's whole point → works on apo/AF3 queries). Threshold **generously** — false
positives fine here; false **negatives are unrecoverable**, so tune for recall/early-enrichment (EF1%, BEDROC).

### Stage 2 — correspondence-level protein scorer (cheap PRECISION, no folding)
From the atom-atom score matrix of a candidate pair, build a protein-level PPI probability from:
- **Optimal transport** over the two interface atom sets → soft correspondence plan + transport cost.
  Respects the one-to-one nature of contacts, so a few great pairs can't dominate (a MaSIF FP mode that a
  naive top-k sum has). *(Design §8 §5.4 already names OT/soft-alignment — this operationalizes it.)*
- **Deformable geometric consistency** — spectral/graph matching that tolerates *local* deformation (or
  cluster correspondences into locally-rigid groups), NOT a single rigid RANSAC. Real interfaces → one large
  mutually-consistent correspondence set; spurious → fragmented.
- **Physical interface features** (cheap, global, *not* captured by a rigid patch score → exactly the
  necessary-but-not-sufficient gap): predicted interface size (# matched atoms / BSA proxy), **contiguity**
  (matched atoms clustered patches vs scattered — crystal-vs-biological signal), shape-complementarity
  distribution (target evolved-interface **Sc ≈ 0.70–0.76**; antibody-like ~0.64–0.68).
- Feed into a light *calibrated* scorer. This is a **better IPA-net**: inputs are learned invariant embeddings
  + OT structure + physical priors, evaluated over the matched *set*, not one rigid superposition. Most FPs
  die here, cheaply.

**Novelty flag:** the lit sweep found **no** work applying OT / spectral-consistency *specifically* to protein-
interface matching (it's standard in generic 3D point-cloud registration — Feydy, a dMaSIF co-author, did
robust-OT registration NeurIPS'21; PointDSC spectral consistency). So Stage 2 is a genuine open opportunity,
not a re-implementation.

### Stage 3 — restraint-guided co-folding (expensive PRECISION + the induced-fit complex)
Only for the Stage-2 shortlist:
- Take the **top OT-matched, high-confidence** GNN contacts (FEW — see below) as **inter-chain contact
  restraints** to a co-folding model.
- Co-fold A+B under those restraints → a **physically realistic induced-fit complex** (sidechains/backbone
  relax → no clash; escapes the rigid-transform trap — the user's core concern).
- Final signal = combine: (a) **restraint satisfaction** (did it form the predicted contacts at high
  confidence?), (b) **PAE-filtered interface confidence** (ipSAE / min-interface-PAE — NOT raw ipTM),
  (c) physical checks on the co-folded complex (BSA, clashes, restraint violation), (d) optionally Boltz-2's
  binder-probability head as a second opinion.

## 3. Locked-ish engine/metric decisions (from the research)
| decision | choice | why |
|---|---|---|
| **co-folding engine** | **Boltz-2** (primary) or **Chai-1** | Native inter-chain contact/distance restraints. Boltz-2: YAML `constraints:` `contact`/`pocket`, `max_distance` 4–20 Å, **`force:true`** steering potential (soft conditioning sometimes ignored w/o it), + a **binder-probability/affinity head**. Chai-1: CSV `contact`/`pocket`/`docking`; 1 restraint lifted Ab–Ag 35%→57%. **AF3 has NO native restraints** (only covalent `bondedAtomPairs` / template / MSA hacks → af3x crosslink trick) → not the engine. Tool `ABCFold` runs all three from one input for A/B. |
| **PPI-likelihood score** | **ipSAE / min-interface-PAE** (+ Boltz-2 binder head) | raw **ipTM is noisy** — documented ~20% FP, disorder-dilution. ipSAE (PAE-cutoff-filtered) & pDockQ2 separate binders/non-binders better. Calibrate threshold on a held-out binder/non-binder set. |
| **# restraints** | **few, high-precision** | *"Restraint quality, not quantity"* (JCIM 2025): a few accurate restraints help, noisy ones mislead → validates using only top OT matches, and makes **Stage-2 precision load-bearing** for Stage-3 trust. |

## 4. Traps to design against
- **Restraint→model→score circularity.** Accurate restraints bias the fold strongly, so a confident forced
  complex ≠ proof of binding. Mitigations: (i) few, high-confidence restraints only; (ii) **weak/no-restraint
  co-folding control** — if an interface only forms when heavily forced, that's evidence *against* binding;
  (iii) the honest signal is *independent* interface-PAE + restraint-satisfaction, not "did the forced complex
  look confident." (iv) physical sanity (clash-free ≠ binding).
- **Benchmark must not be circular.** Co-folding confidence can't be both restraint source and label. Ground
  truth = **experimental binders vs curated non-binders**, incl. **hard decoys** (surface-complementary but
  non-binding — the actual FP test). Pinder is flagged untrustworthy here → a clean binder/decoy set is itself
  a task.
- **Holo-benchmark inflation (north star).** All prior cascades tune thresholds on holo crystal interfaces →
  holo AUC inflated → can't reveal apo FP. **Evaluate with apo/AF3-monomer queries**, protein-level retrieval
  vs decoys, at each stage.

## 5. Staged test plan (cheapest-first, after GNN done)
- **T1 (no folding) — aggregation ablation.** naive top-k sum vs OT-cost vs OT+deformable-consistency+physical-
  feature scorer, on a binder/decoy set, **apo queries**. Establishes Stage 2. Cheap → do first.
- **T2 (small folding set) — restraint-guided co-folding.** Does Boltz-2/Chai-1 with the top GNN contacts yield
  realistic complexes + an ipSAE/min-iPAE that separates binders from decoys? Include the **weak/no-restraint
  control**. Establishes Stage 3 + calibrates circularity.
- **T3 (end-to-end) — full cascade** on a held-out apo binder/decoy benchmark; precision-recall vs the
  MaSIF-Neosurf baseline. **The FP reduction is the headline.**

## 6. Benchmark design (RESOLVED with user, 2026-07-07)

**Deployment = virtual screening, one protein vs thousands** (protein-binder search). → the funnel is
essential (Stage-1 recall-critical), and the **primary metric is retrieval ENRICHMENT**, which makes negatives
largely *implicit* (the DB's thousands of non-partners) — so the eval is NOT blocked on curating a perfect
hard-negative set. Buildable as soon as GNN + apo queries exist.

**Positives:** `data/lists/full_list.txt` (5,902 MaSIF complexes) + **PDBbind+ PPAP** set (~4–5k curated PP
structures, `/work/upthomae/Meng/PDBBindplus/PPAP_dataset.tar.gz` — extract to characterize). Known binders
w/ structures. **Query = apo/AF3 monomers** (induced-fit deployment shape; reuse Phase-3 AF3 pipeline), NOT
holo. **Held out by SEQUENCE CLUSTER (~30% id)** from GNN training — complex-level alone leaks homologs.

**Hard negatives — triangulate (no single ground truth); layered by difficulty:**
1. **Crystal-contact decoys — PRIMARY hard negative** (strongest fit to the FP failure mode): crystal lattice
   packing contacts are **surface-complementary but non-biological** → exactly "real interface vs incidental
   complementary contact." Real structures, curated labels. Sources: EPPIC/PISA bio-vs-crystal, DC (Dockground
   crystal contacts), PRODIGY-cryst, Bahadur/Ponstingl & DeepRank bio-vs-crystal sets, or build from PDB
   symmetry mates.
2. **Swapped-partner decoys** — A's decoys = partners of *other* complexes (real interface-formers, don't bind
   A). Cheap (permute known binders), no model-circularity, harder than random. Minor promiscuity risk.
3. **Negatome** (https://mips.helmholtz-muenchen.de/proj/ppi/negatome/) — validated non-interactors, but sparse
   in structure; use where AF/experimental structures exist as a small high-confidence set.
- **AVOID model-score-mined negatives as a scored set** — circular (labels the model's own complementarity
  hits as negatives, but some are unannotated binders → penalizes a correct model). OK for qualitative error
  analysis / active-discovery only, never for the FP-rate number.

**Benchmark tiers:** (a) apo-query retrieval **enrichment** vs full DB (EF1%, BEDROC, top-k, median rank) —
primary; (b) **hard-negative stress test** (crystal-contact > swapped-partner > Negatome) — FP tail; (c)
**controls**: holo-query upper bound, shuffled ~0.5, sequence-cluster holdout.

## 7. Remaining open question
- **Does Stage 3 even need Stage 2's learned scorer, or do OT-cost + physical features + Boltz-2 binder-head
  suffice?** (parsimony — test in T1/T2.)

## 8. Primary sources
MaSIF (Nat Methods 2020) · MaSIF-seed (Nature 2023, IPA-net) · MaSIF-neosurf (Nature 2024) · dMaSIF (CVPR
2021) · PPIretrieval (arXiv 2402.03675) · Lo Conte–Chothia–Janin 1999 (BSA) · Lawrence–Colman 1993 (Sc) ·
robust-OT registration (Feydy et al., NeurIPS 2021) / PointDSC (spectral) · Boltz-2 (jwohlwend/boltz docs) ·
Chai-1 restraints (chaidiscovery/chai-lab) · af3x crosslinks (bioRxiv 2024.12) · ipSAE (Dunbrack, 2025) ·
"Restraint Quality not Quantity" (JCIM 2025) · ABCFold (Bioinf. Adv. 2025).
