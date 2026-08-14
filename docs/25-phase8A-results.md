# Phase 8 — Stage A results

> Plan: `docs/24-phase8A-plan.md`. Contract: `docs/23-phase8-design.md`. Build log:
> `docs/progress/phase8A-log.md`. Every number traces to a committed artefact under `logs/phase8A/`.
> Stage A is diagnostics only — **nothing was trained**. It ends at the **PAUSE** for D8-12, where
> the choice of apo-prediction method and holo:apo ratio is the user's. What follows is a
> recommendation, not a decision.

## 0. Verdict in one paragraph

Two of the three things Stage A could have invalidated came back **clean**, and one came back
**broken**. The encoder is **not** sidechain-blind (A1) — isolating sidechain atoms costs 70% of
everything the graph carries, so Phase 5's conformational robustness is a real property and not an
artefact of ignoring sidechains. Cryptic pockets **mostly do not close** in AF3 apo models (A2): the
median buried-fraction ratio is 1.001, and the 20% that do degrade do so through **sidechains**,
which a repack step can recover. But **Stage 2 as designed cannot be built on the current Stage-1
scores** (A3): rigid pose prediction succeeds on **0 of 269** complexes in every conformer state and
every checkpoint, while the same fitter given true correspondences succeeds on **100%** — so the
failure is in Stage 1's atom-pair proposals, not in the pose fitter. Its larger component is that the
atom-level objective only ever queried *true interface atoms*, so the encoder was never trained to
tell an interface atom from a non-interface one (§4.2). Fork **F3 is settled
favourably** (≈0.6 s/pair → ~6 core-hours per 40k screen), so the obstacle is the objective, not the
compute. A fourth measurement (A4) says the same embeddings add **nothing** over interface area for
telling a biological interface from a crystal contact. For D8-12, **AFDB covers 68.8% of training
complexes on both sides** but only 13.6% as exact sequence matches; **chai-1 on a shared MSA is
statistically indistinguishable from AF3**; AF3 at five diffusion samples costs the same as one; and
a **FASPR repack reproduces 91% of the AF3 perturbation at zero GPU cost**.

## 1. Provenance (per CLAUDE.md, stated before any result)

Everything below is measured on **encoders trained on HOLO structures only** (PPI 4,767 holo / 0
apo). Apo appears only on the evaluation side. Stage A trains nothing, so it changes that not at
all — it measures the existing holo-trained encoders. D8-12 is the decision that changes it.

## 2. A1 — is the encoder sidechain-blind? **NO.**

Row-preserving ablation ladder (`p8.ablate`), 2 seeds, 287-clean eval, DB=538, chance top-5 0.0093.

| ablation | what it destroys | s0 | s1 | mean | Δ |
|---|---|---|---|---|---|
| `none` | — | 0.651 | 0.638 | **0.644** | — |
| `sc_feat` | sidechain chemistry | 0.353 | 0.257 | 0.305 | **−0.339** |
| `sc_edge` | sidechain connectivity | 0.600 | 0.357 | 0.479 | −0.165 |
| `sc_all` | sidechain atoms isolated | 0.294 | 0.134 | 0.214 | **−0.430** |
| `bb_feat` | backbone chemistry | 0.543 | 0.416 | 0.480 | −0.164 |
| `vert_feat` | the surface channel | 0.387 | 0.301 | 0.344 | −0.300 |
| `all_feat` | **positive control** | 0.043 | 0.020 | 0.032 | −0.612 |

**Both required checks pass.** `none` reproduces the published 0.651 / 0.638 exactly, so this is the
same benchmark and not a lookalike. `all_feat` collapses to ~chance (median rank 166), so the
harness can detect destruction and the other rows mean something.

Sidechain isolation costs **70% of the total** (−0.430 of −0.612), more than backbone chemistry and
more than the entire surface-vertex channel. **The unflattering explanation for Phase 5 —
"conformation-robust because it never reads sidechains" — is refuted.**

**Deviation from the plan, recorded before running.** `docs/24` §3.1 proposed backbone-only atom
nodes. That is unsound: the encoder emits one row per *surface atom* and ~60% of surface atoms **are**
sidechain atoms (measured 0.598), so deleting them renumbers `surf_node_idx`, invalidates
`Rec.inter`, and confounds any drop with patch size. The ladder above is the row-matched version.

**Displacement is spatially specific** (n=52): sidechain ablations move sidechain rows 2–5× more than
backbone rows (`sc_all` 0.833/1.398 vs 0.445/0.296) and `bb_feat` cleanly reverses it
(0.163/0.140 vs 1.387/2.175) — independent confirmation the ablations do what they claim.

**Implicit per-atom σ (D8-9 / D8-19)**: Spearman(displacement, `flex_depth`) = **+0.381 / +0.599**,
p≈0, n=90,121. **Caveat stated up front:** `flex_depth` is input feature column 22, so this shows the
representation is *organised by* flexibility, not that it learned flexibility. It still means a σ
head starts from something rather than nothing.

**What A1 does not show.** It perturbs sidechain *identity*; an apo structure keeps identity and
changes *conformation*. That is A1.2 (§6).

Seed spread is wide (`sc_all` 0.294 vs 0.134). The **ordering** is identical across seeds; the
magnitudes are not, and are reported as ranks.

## 3. A2 — do cryptic pockets close in apo models? **Mostly not; the tail is sidechain-mediated.**

298/298 P–L complexes with AF3 apo protein + crystal ligand pose, **0 failures**.

| quantity | holo | AF3 apo |
|---|---|---|
| ligand buried fraction (median) | 0.851 | 0.840 |
| clashes @2.0 Å (median / p95) | 0 / **0** | 1 / 19 |
| — backbone / sidechain (median) | — | 0 / **1** |

**buried-fraction ratio median 1.001** (p05 0.634, p95 1.124). Pre-registered "collapsed"
(ratio < 0.5 **or** ≥10 clashes at 2.0 Å): **60/298 = 20.1%**.

The holo control is exactly 0 clashes at p95, so the clash metric is calibrated rather than assumed.
The apo failures are **sidechain**, not backbone — recoverable by a repack step, which makes this a
Stage-B design input rather than a ceiling. A `ca_rmsd_in_frame` confound control is recorded so
"collapsed" can be checked against "badly superposed".

## 4. A3 — rigid pose from Stage-1 scores: **0% success**, but compute is not the problem

n=269, 4 checkpoints (Stage-A contrastive ×2 seeds, Stage-B retrieval ×2 seeds), **not**
interface-gated (docs/10 §23 traced part of frozen MaSIF's apparent edge to a gated oracle patch).
Pre-registered success: `fnat ≥ 0.3` **and** `iRMSD ≤ 4 Å`.

| arm | success | fnat med | iRMSD med | corr precision |
|---|---|---|---|---|
| learned, HH → AA (all cells, all ckpts) | **0.000** | 0.031–0.068 | 21–23 Å | 0.001–0.002 (3–9× chance) |
| **ORACLE** (true contacts) | **1.000** | 0.982 | 2.0 Å | 1.0 |
| random correspondences | 0.000 | 0.000 | 26–28 Å | 0.000 |

The oracle succeeds in all four runs, so **the fitter is sound and the failure belongs to the
correspondences**.

### 4.1 What exactly failed: **Stage 1**, not the pose fitter

The test is: score every surface atom of chain 1 against every surface atom of chain 2 with the
learned form `z_i^T T z_j`; take the top-1000 pairs as *predicted interacting atom pairs*; move
chain 2 away with a random rigid transform; ask RANSAC-Kabsch for the transform that best
superimposes those predicted partners; then measure how close chain 2 lands to its true position.

Swapping *only* the predicted pairs for the true contact pairs (the ORACLE arm) takes success from
**0% to 100%**. Everything downstream of the pairs is therefore correct, and **the failure is in
Stage 1's atom-pair proposals** — it is a correspondence failure, not a pose-fitting failure.

### 4.2 Mechanism — correcting two earlier claims

**Claim 1 (wrong), now withdrawn:** I wrote that this "follows from a chain-level `median_i max_j`
objective that never penalised a bad argmax". That is **false**. Atom-level supervision was present
in *both* stages: `p4.objective.info_nce_complex` is an InfoNCE that ranks the true partner atom
against all of the partner chain's atoms, and Stage-B retrieval keeps it as an auxiliary term at
`--w-atom 0.5`.

**Claim 2 (half-right):** hub collapse is a Stage-A pathology only. Stage-B has 342–370 distinct
argmax partners and 210–277 mutual-best pairs and still scores 0%.

**What the objective actually omits.** In `info_nce_complex` the anchor is `z1[pos[:, 0]]` — the
query set is **always a true contacting atom**. The model was trained on *"given that this atom is at
the interface, which partner atom does it touch?"* and **never** on *"is this atom at an interface at
all?"*. Global top-1000 over all pairs mixes the trained axis with the untrained one.

Gating separates them (n=60, oracle gate on **which atoms are interface**, never on which pairs;
labelled per docs/10 §23 and **not** a deployment number):

| correspondence set | Stage-A precision | Stage-B precision | success A / B |
|---|---|---|---|
| global top-1000 (deployment condition) | 0.0020 (7.7× chance) | 0.0020 (8.2×) | 0.000 / 0.000 |
| query gated to true interface atoms | 0.0140 (52×) | 0.0140 (49×) | 0.000 / 0.000 |
| **both sides gated to true interface atoms** | 0.0365 (146×) | 0.0470 (190×) | **0.100 / 0.100** |

So the failure has **two** components, and the larger one is fixable by construction:

1. **Interface localisation (untrained).** Restricting queries to true interface atoms multiplies
   precision by **7×**; gating both sides by **18–24×**. Most of the deployment failure is the model
   scoring non-interface atoms as highly as interface atoms — an axis it was never asked to learn.
2. **Within-interface matching (trained, but weak).** Even with a perfect interface oracle on both
   sides, precision is only 3.7–4.7% and pose success 10%. The trained axis works, but not nearly
   well enough to drive a rigid fit.

**Fork F3 is settled, favourably**: **0.57–0.64 s per pair** with embeddings precomputed →
**~6–7 core-hours per 40k-partner screen**. Stage 2 is affordable; on the current Stage-1 scores it
is not accurate.

## 5. A0 — apo-prediction method benchmark (D8-12 evidence)

### 5.1 AFDB coverage — the number Option A lives or dies on

12,198 chain instances across 5,618 entries, via RCSB GraphQL → UniProt → AlphaFold DB.

| class | n | % |
|---|---|---|
| ident ≥95% | 5,194 | 42.6 |
| subsequence (domain/fragment of the AFDB entry) | 2,222 | 18.2 |
| exact | 1,664 | 13.6 |
| no UniProt cross-reference | 1,181 | 9.7 |
| ident <95% | 1,095 | 9.0 |
| no AFDB model | 740 | 6.1 |
| other (too long to align / not found) | 102 | 0.8 |
| **usable (exact + subsequence + ≥95%)** | **9,080** | **74.4** |

**The decision-relevant number is per COMPLEX, not per chain**: a 1:1 apo training pair needs *both*
partners. But "usable" depends entirely on how much sequence mismatch you will tolerate, and that in
turn is set by how you intend to define training positives on the apo model:

| definition of usable | training complexes, both sides | % |
|---|---|---|
| **exact only** (identical entity sequence) | 625 | **12.6** |
| exact + subsequence (truncate the AFDB model) | 1,253 | **25.3** |
| exact + subsequence + ≥95% identity | 3,400 | 68.8 |

**The 68.8% headline is the permissive reading and should not be used on its own.** Positives are
defined by mapping holo interface atoms onto the apo model through an identity join on
`(chain, resseq, name)`; every mismatched residue drops out of the positive set. If the corpus is
required to have exact atom correspondence, **AFDB covers only 12.6% of complexes** — and 25.3% if
subsequence hits are truncated to the crystal construct.

Definitions, precisely:
* **exact** — the RCSB entity canonical sequence (`pdbx_seq_one_letter_code_can`, i.e. SEQRES) equals
  the AFDB `uniprotSequence` in full: 100% identity **and** 100% coverage. Note this is a *sequence*
  guarantee, not an atom-set guarantee — the crystal may still leave residues unresolved.
* **subsequence** — the entity sequence is a contiguous substring of the AFDB sequence: 100%
  identity over the aligned region, <100% coverage of the AFDB model.
* **≥95%** — biotite `get_sequence_identity(..., mode="shortest")`: matches divided by the length of
  the **shorter full sequence**, not by the aligned region only.

A parsing bug worth recording: a list "side" concatenates chains (`1A14_HL_N` is an antibody
heavy+light against an antigen). Treating `HL` as one chain id put ~11% of the corpus into a fake
"not found" class before it was caught.

### 5.2 AF3 with five diffusion samples

30/30 chains. Test set: 6 balanced strata, length 62–863.

| metric | median | p25 | p75 |
|---|---|---|---|
| TM-score (holo-normalised) | 0.978 | 0.867 | 0.991 |
| TMalign aligned RMSD (Å) | 0.845 | 0.470 | 1.488 |
| inter-sample pairwise RMSD (Å) | **0.220** | 0.168 | 1.540 |
| Spearman(RMSF, holo B-factor) | +0.352 | 0.139 | 0.599 (n=13) |
| Spearman(RMSF, pLDDT) | **−0.613** | −0.686 | −0.459 |

**Five samples cost the same as one**: ~60–70 s per chain at `NSAMP=5`, identical to the Phase-7
`NSAMP=1` timing, because the MSA and trunk dominate and diffusion samples are nearly free. Phase 7's
`NSAMP=1` was leaving four free conformers on the table.

The ensemble spread is **calibrated in sign** — it widens where pLDDT is low (ρ = −0.613) and tracks
crystallographic B-factors (ρ = +0.352) — but it is **small in magnitude** (median 0.220 Å).

**Limitation, measured not assumed:** our corpus has almost no low-confidence chains — pLDDT p01 =
77.5, p50 = 95.3, and **1 of 883 chains below 70**. The test set therefore had to be stratified by
*quartile extremes* rather than a median split, and even the "low-confidence" stratum is ~90 pLDDT.
Conformational diversity from resampling will be limited on this corpus **whatever method is chosen**.

### 5.3 Chai-1, on the identical MSA

30/30 chains in both arms. **The shared-MSA design was verified, not assumed**: chai logs
"MSA found for sequence …" and the runner captures it — 30/30 in the MSA arm, **0/30** in the
MSA-free arm. (This mattered: chai resolves an MSA as `msa_directory / expected_basename(sequence)`,
a sequence hash, and on a miss it only *warns* before falling back to a single sequence. The first
naming would have made "chai + shared MSA" silently identical to "chai without MSA".)

| method | TM-score | aligned RMSD (Å) | ensemble spread (Å) | ρ(RMSF, pLDDT) | s/chain |
|---|---|---|---|---|---|
| AF3, 5 samples, shared MSA | 0.978 | 0.845 | 0.220 | −0.613 | ~60–70 |
| Chai-1, 5 samples, shared MSA | 0.945 | 1.130 | 0.435 | −0.650 | 73.8 |
| Chai-1, 5 samples, **MSA-free** | 0.953 | 1.385 | 0.783 | −0.628 | 74.2 |

Medians mislead here, so the comparison is **paired per chain** (n=30, Wilcoxon):

| comparison | TM-score Δ | aligned RMSD Δ | spread Δ |
|---|---|---|---|
| chai(MSA) − AF3 | **−0.001, p=0.38** | +0.085 Å, p=0.052 | +0.129 Å, p=0.5 |
| chai(no MSA) − AF3 | −0.012, **p=0.011** | +0.590 Å, **p=9.7e-5** | +0.361 Å, p=0.084 |

**Chai-1 on the shared MSA is statistically indistinguishable from AF3 on accuracy** (TM p=0.38;
RMSD borderline at p=0.052). Dropping the MSA costs a small but real amount: TM −0.012 and
+0.59 Å RMSD. The spread differences favour chai but are **not significant at n=30**, so "chai
samples more broadly" is suggestive only.

Inference cost is comparable (~74 s vs ~60–70 s per chain), and both numbers exclude MSA *search*.
The MSA-free arm's real advantage is that it needs no search at all.

## 6. A1.2 — the conformational test: a repack is **91%** of the AF3 perturbation

A1 perturbed sidechain *identity*. An apo structure keeps identity and moves only the rotamer, so
this is the measurement that connects A1 to the north star. 99/100 FASPR fixed-backbone repacks
succeeded; 90 complexes have holo + AF3 + repack.

| quantity | seed 0 | seed 1 |
|---|---|---|
| relative displacement, FASPR repack | 0.322 | 0.206 |
| relative displacement, AF3 | 0.353 | 0.224 |
| **ratio repack / AF3** | **0.914** | **0.917** |

**A fixed-backbone sidechain repack moves the embedding ~91% as far as a full AF3 re-prediction**,
in both seeds. Almost everything the encoder feels when handed an apo structure is **sidechain
rearrangement**, not backbone. Together with A1 (the encoder reads sidechains) and A2 (apo pocket
loss is sidechain-mediated), three independent measurements land on the same lever.

Retrieval with the repack substituted into the AF3 slot (n=90, DB=180, chance top-5 0.028):

| cell | s0 | s1 |
|---|---|---|
| HH holo–holo | 0.733 | 0.694 |
| **AA repack–repack** | **0.700** | **0.706** |
| shuffled control | 0.033 | 0.033 |

The encoder is robust to repacking as it is to AF3 (−0.033 / +0.012, inside seed spread).

### 6.1 Correction: the "implicit σ" from A1 does not survive

Under `sc_all` (feature destruction) Spearman(displacement, `flex_depth`) was **+0.381 / +0.599**.
Under an actual repack it is **−0.18 / −0.18** — **the sign flips**.

So the encoder does *not* move more where sidechains are more rotatable when the perturbation is a
real conformational change; if anything it moves less. One reading is that it has learned to be
*insensitive* at flexible positions — good for robustness, bad for a σ head. Either way **D8-9
cannot assume the σ signal is already present**; the A1 correlation was an artefact of the
perturbation type, and calling it an implicit σ before this test was premature.

## 7. A4 — is bio vs crystal separable, and by what?

393/400 mined entries, **1,755 interfaces (1,460 bio / 295 crystal)**, labelled by the
**identity-operator** rule: a pair is biological only if both chains receive the IDENTITY transform
inside one `BIOMOLECULE`. Chain-list co-occurrence alone would mislabel "chain A plus a symmetry copy
of A" as a biological A–B pair — precisely the case Stage 3 must get right.

BSA median: **bio 1,828 Å², crystal 615 Å².**

| arm | AUROC | AP(crystal) |
|---|---|---|
| **BSA only** | **0.827** | **0.474** |
| + contact count + chain sizes | 0.857 | 0.546 |
| BSA only, shuffled labels | 0.526 | 0.177 |
| structural, shuffled labels | 0.512 | 0.173 |

Grouped 5-fold CV by PDB entry (interfaces from one entry share chains). Shuffled controls collapse
to chance with AP at prevalence (0.168), so the folds and the metric are sound.

**This is the D8-7 hard control, now quantified: Stage 3 must beat AUROC 0.827 / AP(crystal) 0.474.**
Interface area alone is a strong predictor of "biological", which is exactly why it needed measuring
before anything is built on top of it. Adding trivial geometry buys +0.030 AUROC.

### 7.1 Do the Stage-1 embeddings add anything? **No.**

For 80 of the mined entries we built reference surfaces and 26-D graphs (471 pairs surfaced in the
end; graphs were built from the 455 available at that moment, of which 413 were scorable), so the learned bilinear score `z_i^T T z_j` over contacting atom pairs could be summarised
per interface (mean / max / median / p90) and added as a third arm. **All arms are compared on the
identical 413 rows** — the first run silently dropped the embedding arm because 42 interfaces lacked
features, which would have compared arms on different populations.

n=413 (259 bio / 154 crystal; note this subset was deliberately enriched for crystal contacts, so its
prevalence 0.373 differs from the full set's 0.168 and the AUROCs are **not** comparable to §7 above):

| arm | seed 0 AUROC | seed 1 AUROC | AP(crystal) s0 / s1 |
|---|---|---|---|
| BSA only | 0.841 | 0.841 | 0.775 / 0.775 |
| + contacts + chain sizes | **0.863** | **0.863** | 0.794 / 0.794 |
| **+ Stage-1 embedding summaries** | 0.852 | 0.855 | 0.773 / 0.780 |
| shuffled controls | 0.531–0.542 | 0.531–0.542 | ≈ prevalence |

**The Stage-1 embeddings add nothing** — both seeds land slightly *below* the structural arm, which
is what four uninformative features do to a linear model on 413 rows. So the current encoder carries
no usable bio-vs-crystal signal beyond interface area and size.

This is a **warning, not a gate** (a trained pose-level network is a different model). But taken with
A3 — where the same embeddings could not place a pose — it is the second independent indication that
the current Stage-1 representation is tuned for *chain-level retrieval* and not for the atom- or
interface-level judgements Stages 2 and 3 need.

## 8. Recommendation for D8-12 — **the decision is yours**

Stage A produces evidence; the choice of method and holo:apo ratio is yours. What the evidence says:

**1. AFDB coverage is far weaker than the headline once positives must be mappable.** Requiring an
exact sequence match — the only case where holo interface atoms transfer to the apo model with no
loss — leaves **12.6% of training complexes**, not 68.8%. Truncating subsequence hits to the crystal
construct raises it to 25.3%. The 68.8% figure needs the ≥95% class, which is 42.6% of chains and
means point mutants and tagged constructs whose mismatched residues drop out of the positive set.
Option A therefore buys coverage at the cost of a corpus with three different kinds of sequence
relationship — and, at the strict setting that keeps positives clean, it barely covers an eighth of
the corpus.

**2. Local prediction is cheaper than it looked, and uniform.** Chai-1 on the shared MSA is
statistically indistinguishable from AF3 (TM Δ −0.001, p=0.38). **MSA-free** chai costs only
TM −0.012 / +0.59 Å — and needs **no MSA search at all**, which is the dominant real cost. At
~74 s/chain, predicting **every** chain in the corpus (9,886 chain instances × 5 conformers) is
roughly **200 GPU-hours**. That yields one provenance, five conformers per chain, and no
construct-matching ambiguity.

**3. AF3 five samples are free.** ~60–70 s/chain at `NSAMP=5`, identical to `NSAMP=1`, because MSA
and trunk dominate. Wherever we already hold MSAs (~880 chains), five conformers cost nothing extra.

**4. A repack is a 91%-strength apo proxy at zero GPU cost.** This is the finding the plan did not
anticipate. If what the encoder feels from an apo structure is 91% sidechain rearrangement (§6), then
FASPR repacking gives most of the apo signal for ~1 CPU-minute per complex — usable as a **high-volume
augmentation across the entire corpus**, orthogonal to and much cheaper than a predicted-structure arm.

**5. Conformational diversity will be limited whatever you choose.** Our corpus is almost entirely
high-confidence (pLDDT p01 = 77.5, 1 of 883 chains below 70), and AF3's five-sample spread is a
median 0.220 Å. Resampling will not manufacture large conformational variety here.

**My recommendation, for you to accept or reject:** a **hybrid** — FASPR repack as a cheap augmentation
over the whole corpus (it captures 91% of the perturbation), plus **chai-1 MSA-free** for a genuine
predicted-structure arm at ~200 GPU-h, in preference to AFDB. AFDB's 68.8% coverage is not worth the
three-way construct heterogeneity when uniform local prediction is affordable and the accuracy gap to
AF3 is TM 0.012. On the ratio: begin at **1:1 holo:apo** and treat the extra conformers as
augmentation rather than as distinct training pairs, since the spread between them is small.

### 8.1 The finding that should be decided before Stage B, not after

**A3 says the funnel's Stage 2 cannot be built on the current Stage-1 scores.** Rigid pose prediction
scores 0/269 in every conformer state and every checkpoint while the oracle scores 100%. This is not
a compute problem (F3: ~6 core-hours per 40k screen) and not merely hub collapse (the Stage-B
checkpoints are well distributed and still fail). It is that the chain-level `median_i max_j`
objective never constrained *which* partner attains the max.

Per §9 of the plan I am not acting on this unilaterally, because it reorders the phase. The options
are: (a) add the **missing axis** to Stage 1 — an interface-vs-non-interface term, since an atom-level
correspondence loss already exists and the gap is that it only ever queries true interface atoms
(§4.2); (b) replace the RANSAC pose step with a learned pose module trained end-to-end (D8-10 already
anticipates differentiability); or (c) skip explicit poses and score interfaces directly, collapsing
Stages 2 and 3 into one. §4.2 makes (a) the cheapest first move — it is a loss change, not an
architecture change — but even a perfect interface oracle only reaches 10% pose success, so (a) alone
will not be sufficient. **This is worth your decision alongside D8-12.**

## 9. Cost

≈ **CHF 12** total: Jed CPU (A1 ablations, A3 poses, A2, repacks, A4 surfaces/graphs, AFDB sweep)
≈ CHF 6; Kuma GPU (AF3 `NSAMP=5` on 30 chains, chai ×2 arms ×30 chains) ≈ CHF 6. No training.

## 10. What was NOT evaluated

* **Protenix**: the `conda_envs/protenix` env exists (torch 2.7.1, cuequivariance) but the
  `protenix` package is **not installed** — my plan's "zero-install candidate" was wrong.
* **Boltz-2, ESMFold**: not installed; fall under the §9 install timebox.
* **A4's embedding arm covers 413 of 1,755 mined interfaces** (80 entries surfaced of 393 mined),
  deliberately enriched for crystal contacts. The BSA/structural baselines are reported on the full
  1,755; the three-arm comparison on the 413.
* Two of twenty A4 surface tasks were cancelled after 2 h once both seeds of the probe had run.
  **471 of 525** selected pairs were surfaced (29 failed in the reference pipeline, the rest
  cancelled); the embedding arm used the **455** available when graphs were built. The extra 16
  would not move a two-seed null, so the probe was not rerun.
* These omissions narrow A0 to AF3 vs Chai-1. That is enough to answer "is there a cheaper
  alternative to AF3 at comparable quality", which is the D8-12 question, but it is not the full
  five-method sweep the plan proposed.
