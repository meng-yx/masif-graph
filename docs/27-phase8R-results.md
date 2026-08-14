# Phase 8 — Stage R results: **the gate FAILED**

> Design: `docs/26-phase8R-design.md`. Everything traces to `logs/phase8R/` and
> `/work/upthomae/Meng/phase8R/`.
> **Nothing here should be adopted.** Every Stage-R checkpoint is worse than the Phase-6C/7 encoder
> it was meant to replace.

## 0. Verdict

The R4 gate fails on **both** criteria, at 2 seeds, across 4 ablation arms:

| arm | top-1 spatial error | chain retrieval HH | rotation | gate |
|---|---|---|---|---|
| `full_s0` | 17.15 A | **0.024** | 2.4e-06 | **FAIL** |
| `full_s1` | 20.21 A | 0.032 | 2.4e-06 | **FAIL** |
| `hardtarget_s1` | 17.06 A | 0.028 | 3.1e-06 | **FAIL** |
| gate | **< 5 A** | **>= 0.624** | ~float eps | |

* **Spatial**: 17-20 A against a < 5 A gate. Marginally better than the 19.4 A Stage-A baseline, and
  *not* better than picking a random interface atom (below).
* **Chain retrieval collapsed** from **0.644 -> 0.024-0.032**, i.e. to near chance (0.0093). Stage R
  destroyed the one capability the project actually had.
* Rotation invariance survived (R1's only unambiguous success).

## 1. The control that reframes the whole problem

I had been reporting the encoder's atom matching as "weak but real" (18x chance on rank). The
missing control says otherwise:

| selector | median distance to true partner | within 5 A |
|---|---|---|
| random atom anywhere on the partner chain | 27.69 A | 0.020 |
| **random atom among the partner's true interface atoms** | **19.99 A** | **0.071** |
| Phase-6C/7 encoder (Stage-A baseline) | 19.4 A | **0.071** |
| best Stage-R arm | 17.1 A | - |

**The Phase-6C/7 encoder is indistinguishable from random selection among interface atoms** - the
within-5 A figures agree to three decimals. Reconciling this with the 18x rank enrichment measured in
Stage A gives the precise statement:

> The encoder **finds the interface region** of the partner chain, and has **no discrimination
> whatsoever within it**. Interfaces are 20-30 A across, so "right interface, random atom inside it"
> *is* a ~20 A error.

This also corrects the sharper claim in `docs/25` §4.3 that the failure "is not right-neighbourhood,
wrong-atom". At the resolution that matters it is exactly that - the neighbourhood is just far bigger
than I implied.

## 2. Why Stage R failed: the score matrix goes flat

One mechanism explains both failures. On the overfit checkpoint (1,265 candidates):

| quantity | value |
|---|---|
| softmax entropy | 7.088 nats (log 1265 = **7.14**) |
| **effective number of candidates** | **1,197 of 1,265** |
| within-query score spread | 0.0336 (tau = 0.103 => logit spread ~ 0.33) |
| top-1 softmax probability | 0.00172 (uniform = 0.00079) |

The scores are **essentially uniform**. A near-uniform score matrix cannot rank a partner atom, and
it cannot produce a chain-level `median-of-max` contrast either - so atom matching and chain
retrieval fail together, for the same reason.

The likely driver is the dustbin term: training many non-interface queries to prefer "no partner" is
most cheaply satisfied by shrinking *all* scores toward zero, and with L2-normalised embeddings and a
bounded bilinear form there is nowhere else for the model to go.

## 3. It is not the loss, the context, or the scorer - the ablations are flat

12 epochs, 3,000 complexes, 2 seeds, one variable changed per arm:

| arm | isolates | s0 | s1 |
|---|---|---|---|
| `full` | everything | 21.6 | 24.8 |
| `noctx` | R1 invariant global context | 20.9 | 25.1 |
| `nodisto` | R3 distogram gradient | 26.3 | 27.0 |
| `hardtarget` | R2a distance-decayed targets | 25.0 | 21.2 |

Every arm sits in the same 21-27 A band, and seed spread (+/-2-3 A) exceeds every between-arm
difference. **None of R1, R2a or R3 measurably helps.** That uniformity is itself the finding: the
bottleneck is something all four arms share.

Three further probes rule out the obvious candidates:

* **Can it fit at all?** 20 complexes, evaluated on *the same* 20, 60 epochs, matching loss only (no
  distogram, dustbin or chain term): loss plateaus at **6.95**, against log(1265) = **7.14** - the
  uniform value - and spatial error stays at **21.4 A**. It cannot overfit twenty complexes it has
  seen sixty times. So this is neither a data-volume nor a generalisation problem.
* **Is the bilinear form the bottleneck?** Ranking candidates with the trained MLP distogram head
  instead of `z^T T z` is **worse** (26.7 A vs 23.6 A). Swapping the scorer does not help, so the
  information is not present in the embeddings for either scorer to use.
* **Capacity** (4x larger: d 256, d_out 128, 6 layers, same 20 complexes) does begin to fit - loss
  7.36 -> 6.57 over 33 epochs - but only slowly, and this is memorisation of 20 complexes, not a
  demonstration that the task generalises.

The distogram head itself trained fine (binned MAE 5.03 -> 3.89 A, contact recall 0.20 -> 0.38 after
the class-balance fix), but that reflects the easy separation between random far pairs and true
contacts, not fine-grained spatial discrimination.

## 4. What this means for Phase 8

**The pre-registered fallback in `docs/26` §4 is now the live option:** abandon explicit atom-level
correspondence, collapse Stages 2 and 3, and score interfaces directly from Stage-1 embeddings.

The evidence supports that specifically, rather than as a retreat:

* the encoder's **interface-level** signal is real and strong - Phase-5 retrieval 0.644, robust to
  AF3 and to repacking - and it is what the deployment target (retrieval/screening) actually needs;
* its **atom-level** signal is nil, and three independent attempts to create one (Phase-4->7
  chemistry elaboration, Stage-A's diagnosis, Stage R's redesign) have failed;
* Stage 3's bar is already known and does not need correspondences: BSA-only AUROC **0.827** /
  AP 0.474 (`docs/25` §7).

**Keep the Phase-6C/7 encoder.** No Stage-R checkpoint should be adopted for anything.

## 5. What was NOT tested

* Only 12 epochs on 3,000 of 4,418 complexes per arm; a longer schedule was not tried, though the
  overfit probe argues training length is not the limit.
* Sinkhorn (R2c) was designed but never run - the flat-score diagnosis made it moot, since Sinkhorn
  normalises a matrix that carries no signal.
* The larger-capacity configuration was run only as a 20-complex memorisation probe, never trained
  properly. If the fallback is rejected, that is the one remaining untested direction.
* The repack corpus (4,428 complexes) and its graphs (4,313) were built and used for conformer
  augmentation, but with the gate failing, the apo-augmentation question is untested on its merits.

## 6. Cost

~ **CHF 20**: repack corpus ~4, rp graphs ~3, Kuma sweep (8 runs x 12 epochs) ~7, probes and gate
evaluations ~6.
