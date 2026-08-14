# Phase 8 — Stage R: redesign Stage 1 (and replace Stage 2)

> Inserted **between Stage A and Stage B** on 2026-08-12, with the user, after Stage A found that
> Stage 1's atom-pair scores cannot drive pose prediction (`docs/25` §4). Approved: Stage R and its
> gate, dropping the explicit pose, starting the repack corpus in parallel, and the representation
> change in R1.
>
> Read `docs/25-phase8A-results.md` §4.1–4.3 first — Stage R exists because of those measurements.

## 0. Why Stage R exists

Stage A measured three things that together condemn the current Stage 1 as an input to Stage 2:

| measurement | value |
|---|---|
| rigid pose success, 4 checkpoints × 4 conformer cells | **0 / 269** |
| same fitter, true correspondences (ORACLE) | 100%, fnat 0.982 |
| rank of the true partner, given a true interface atom as query | median **109 of 854** |
| distance from the top-1 predicted partner to the true partner | median **19.4 Å**, 48% > 20 Å |

The failure is **Stage 1's atom-pair proposals**, not the pose fitter, and it is a failure of
**spatial localisation**: the model points at a different part of the partner surface, not at the
wrong atom in the right neighbourhood. A 3–5 Å error would have been harmless.

**Mechanism.** `z_i^T T z_j` asks whether two atoms sit in complementary *local* environments, and
complementary local environments recur all over a protein surface. The encoder is local and
SE(3)-invariant by construction — distances and cosines only, vv edges ≤4 Å, va edges ≤5 Å, 4
message-passing layers ⇒ a receptive field of ~15–20 Å, about one MaSIF patch. **Two chemically
similar patches 40 Å apart are provably indistinguishable to it.** No loss change alone can fix that.

**Stage R runs entirely on the existing holo corpus.** No new data generation is required, so if the
representation cannot be fixed we learn that before spending on an apo corpus.

## 1. R1 — break the locality (representation)

Add **rotation-invariant global context** per atom, so position on the surface is representable:
distance to the chain centroid, and distances to the chain's principal axes. These are invariant
scalars, so the provable rotation-invariance that Phase-4 M0 established is preserved.

**Gate:** `encoder_rotation_maxdiff` stays at float-epsilon (the M0 test, exact 0.0 in Phase 4).

This touches **D2/D4**, the decisions that have consistently worked, which is why it is gated on both
M0 and a chain-retrieval do-no-harm check (§4).

## 2. R2 — redesign the atom-level loss

Current (`p4.objective.info_nce_complex`, used in Stage A and kept in Stage B at `--w-atom 0.5`):

```
for each contact row (i, j):
    s = (z1[i]^T T z2) / tau        # over ALL ~850 partner atoms
    loss = CrossEntropy(s, target = j)      # single hard label
symmetrised over both directions
```

Three changes, in priority order:

**R2a — distance-decayed soft targets.** Replace the hard label with a target
`∝ exp(−d(j′, j_true)² / 2σ²)` over partner atoms, σ ≈ 3–5 Å. This changes the task from *"identify
the exact atom"* — probably not identifiable from local chemistry, and unnecessary — to *"localise
the region"*, which is identifiable and is what everything downstream actually needs. It also
absorbs the multi-positive contamination: an interface atom has a mean of 1.8 true partners
(median 1), and single-label cross-entropy currently treats the other 0.8 as **negatives**.

**R2b — a "no contact" dustbin column, with non-interface queries.** Today the anchor is always a
true contacting atom (`z1[pos[:, 0]]`), so *"is this atom at an interface at all?"* is entirely
unsupervised. Stage A measured that axis: gating queries to true interface atoms multiplies
correspondence precision by **7×**, gating both sides by **18–24×**.

*Priority note.* At deployment the query patch is specified by the user, so the query axis is
supplied externally. R2b is therefore **second** priority — but still required, because the
**candidate** side is never gated and the dustbin is what calibrates the score scale.

**R2c — Sinkhorn normalisation** of the score matrix instead of two independent softmaxes, giving an
assignment with bilateral consistency. Optional; add only if R2a+R2b leave the matching ambiguous.

R2a–R2c together are essentially **SuperGlue's formulation** (match points across two views, with
unmatched points, under a global consistency constraint) — the same problem shape, worth borrowing
rather than reinventing.

## 3. R3 — replace Stage 2 with an inter-chain distogram (D8-2 revision)

**The explicit real-space pose is dropped** (user decision, 2026-08-12). Stage 2 no longer predicts a
rigid transform; it predicts **distance restraints**.

* **Head:** an MLP on pair features `[z_i, z_j, z_i ⊙ z_j, s_ij]` for Stage-1-shortlisted candidates
  → a distribution over distance bins, including an explicit **">20 Å / no contact"** bin.
* **Loss:** cross-entropy against the binned true distance from the crystal.

Why this is better than a pose, and *cheaper*:

1. **It deletes a whole class of bug.** Every superposition-based target — atom-centre Kabsch, vertex
   Kabsch, MaSIF-neosurf's alignment — carries a systematic offset against experimental structures.
   Stage A hit exactly this: fitting on atom centres co-locates contacting atoms that are really
   **3.79 Å** apart, interpenetrating the chains (closest approach 1.06 Å vs 2.55 Å native) and
   imposing a **1.9 Å iRMSD floor** (`docs/25` §4). A predicted distance needs **no alignment**, so
   there is no frame and no frame bias.
2. **Supervision gets ~5 orders of magnitude denser** — one complex yields ~10⁵–10⁶ supervised
   pairwise distances instead of one pose. This is why AlphaFold predicts a distogram.
3. **It subsumes the axis Stage 1 lacks** — a head that can say ">20 Å" *is* a contact classifier.
4. **It refuses to commit prematurely**, which is where information currently dies: RANSAC picks one
   consistent-but-wrong configuration and discards the rest.
5. **It is the north star's quantity.** "How much shape and chemical mismatch is tolerable, per atom,
   conditioned on local environment" *is* the spread of the predicted distance distribution — D8-9's
   σ in ångströms, on the pair, directly supervisable.
6. **It back-propagates a spatially meaningful gradient into Stage 1**, which is the fix R2 alone
   cannot deliver.

### R3b — keep geometric consistency as a *scalar*, not a pose

A rigid pose is a strong prior (6 DOF explaining hundreds of contacts) and mutual consistency is
probably the most discriminative feature available for real-vs-spurious interfaces. Do not discard
it — **measure** it: the residual of a differentiable weighted-Procrustes fit **to the predicted
distances**, handed to Stage 3 as a feature.

If a pose is ever wanted (top hits only, on demand), fit to the predicted distances `d̂ ≈ 3.8 Å`
rather than by co-locating atoms at `d = 0`. That is the unbiased version of the fit Stage A got
wrong, and it comes for free from this formulation.

## 4. R4 — the gate (pre-registered)

**Primary metric: median spatial error of the top-1 predicted partner, target < 5 Å.** Measured
given a true interface atom as query — the deployment condition. Current value: **19.4 Å**.

Chosen deliberately over InfoNCE loss and over chain retrieval: it is the quantity that predicts
whether anything downstream can work, and Stage A showed chain retrieval can be excellent (0.644)
while this is catastrophic.

| check | bar |
|---|---|
| **top-1 spatial error** | **< 5 Å** (from 19.4 Å) |
| chain retrieval, do-no-harm | ≥ 0.644 − 0.02 on the 287-clean set |
| rotation invariance (M0) | maxdiff ~ float epsilon |
| distogram accuracy | reported; no bar pre-registered (first measurement) |
| seeds | **≥ 2** for every claim (D8-11) |

**If the gate fails:** stop before corpus spend. Explicit fallback — abandon explicit correspondence
and score interfaces directly from Stage-1 embeddings, collapsing Stages 2 and 3 into one.

## 5. Cost and ordering

| step | cost | when |
|---|---|---|
| **repack corpus** (4,418 complexes, FASPR + surfaces) | ~CHF 4–9 | **running now, in parallel** (job 66122747) |
| R1–R4 training + diagnostics, ≥2 seeds | ~CHF 20–30 | after the repack corpus lands |
| chai-1 MSA-free apo arm (~200 GPU-h) | ~CHF 100 | **held until R passes** |
| Stage B (corpus), Stage C (Stage 3) | per `docs/23` | after R |

The risk ordering is deliberate: the CHF ~100 item waits on a gate that costs ~CHF 25 to evaluate.

## 6. D-decisions this changes

| decision | change |
|---|---|
| **D8-2** (funnel) | Stage 2 is a **distogram + consistency scalar**, not a pose predictor |
| **D8-9** (σ) | σ becomes the per-pair predicted distance spread. A1.2 showed the current encoder has **no** implicit σ — the `flex_depth` correlation **flips sign** under a real repack — so it must be learned, not assumed |
| **D8-10** (differentiability) | satisfied by construction; no separate requirement |
| **D8-12** (apo corpus) | **RESOLVED**: hybrid — FASPR repack over the whole corpus + chai-1 MSA-free. AFDB rejected (only 12.6% of complexes match exactly; the ≥95% class cannot carry positives cleanly) |
| **D2 / D4** (representation) | R1 adds invariant global context; gated on M0 |
| **new — altLoc** | keep conformer `A` (current behaviour, now an explicit decision). Measured: altLocs in **21% of chains**, 0.53% of residues |
| **new — incomplete sidechains** | **do not repair holo.** Holo geometry is the positive-label ground truth and a rebuilt sidechain would *invent* contacts. Flag `is_complete` and exclude those contacts. Measured 1.13% of residues, **depleted at interfaces** (0.37%, 0.60×) |
