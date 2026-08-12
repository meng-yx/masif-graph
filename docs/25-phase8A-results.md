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
every checkpoint, while the same fitter given true correspondences succeeds on **100%**. The
atom-level scores are not spatially discriminative — a direct consequence of training on a
chain-level `median-of-max` objective that never penalised a bad argmax. Fork **F3 is settled
favourably** (≈0.6 s/pair → ~6 core-hours per 40k screen), so the obstacle is the objective, not the
compute. For D8-12, **AFDB covers 68.8% of training complexes on both sides**, and AF3 at five
diffusion samples costs the same as one.

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

**Mechanism, with a correction.** An 8-complex pilot on a Stage-A checkpoint suggested hub collapse.
At full scale that is only half the story: Stage-B checkpoints are **not** hub-collapsed — 370
distinct argmax partners and 211–277 mutual-best pairs, versus 77 and 2–3 for Stage A — and still
score 0%. Plentiful, well-distributed correspondences are **not sufficient**. The atom-level scores
are simply not spatially discriminative, which follows from a chain-level `median_i max_j` objective
that is indifferent to *which* partner attains the max.

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
partners. **3,400 / 4,943 training complexes = 68.8%** have both sides AFDB-usable.

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

### 5.3 Chai-1

*(filled in below when the runs complete)*

## 6. A1.2 — repack sensitivity

*(filled in below)*

## 7. A4 — bio vs crystal signal

*(filled in below)*

## 8. Recommendation for D8-12 (the user decides)

*(filled in below)*

## 9. What was NOT evaluated

* **Protenix**: the `conda_envs/protenix` env exists (torch 2.7.1, cuequivariance) but the
  `protenix` package is **not installed** — my plan's "zero-install candidate" was wrong.
* **Boltz-2, ESMFold**: not installed; fall under the §9 install timebox.
* These omissions narrow A0 to AF3 vs Chai-1. That is enough to answer "is there a cheaper
  alternative to AF3 at comparable quality", which is the D8-12 question, but it is not the full
  five-method sweep the plan proposed.
