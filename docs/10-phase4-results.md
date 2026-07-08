# Phase 4 — Results (arc-1: M0 + M1 feasibility)

> Living, honest. Every number traces to a committed artifact + a recoverable command. "Pipeline ran" is
> stated separately from "result is valid." Cumulative CHF logged. Absolute metrics + controls always.

**Status:** arc-1 M0 COMPLETE + self-verified; M1 Stage-A feasibility run on Kuma H100 (job **3795493**).
**Cumulative CHF:** ≈ 0.3 (Kuma: crashed probe 3795482 ≈ 0.01 + Stage-A 3795493 ≈ 0.15–0.34; a redundant
double-submit 3795496 was detected and **cancelled after ~10 min** ≈ 0.15; Jed CPU ≈ 0). Far under the
CHF-100 arc budget. _(Two conductor instances raced and each launched a Stage-A job; 3795496 cancelled,
3795493 kept — see `docs/progress/phase4-log.md`.)_

---

## Arc-1 deliverable checklist (handoff §1)
- [x] **M0 built + rotation-invariance gate PASS** (geometry + embedding). The make-or-break correctness check.
- [x] **M1 pipeline validated + held-out holo→holo AUC** (Stage-A, 3 seeds: SC best 0.749±0.035 vs frozen
      0.947; dense 0.739 vs frozen 0.682; shuffled ≈0.50).
- [x] **Stage-A full-set GPU cost estimate** — 20 ms/complex/step → ~CHF 1.4–2.1 per 100–150-epoch run.
- [x] **M1 feasibility verdict, honestly framed** — learns + beats pooled frozen on dense, but ~0.75 (unstable)
      vs 0.947 on 90 complexes → promising, data-limited, gated on the full-set scale-up.

---

## M0 — heterogeneous graph builder + SE(3)-invariance gate — **PASS**

`src/masif_graph/graph/hetero.py` builds a per-chain graph with **atom nodes** (all heavy atoms; Phase-2
invariant chem features: element/backbone/aromatic/degree + flex-depth + electronegativity/valence/covalent
radius) and **surface-vertex nodes** (MaSIF input channels `[si, hbond, charge, hphob]`, normalized ~[-1,1]).
Three edge types, **all geometry as SE(3)-invariant scalars**:
- **atom–atom covalent** (bond-order one-hot + sidechain-rotatable flag); no through-space atom edges (design
  §4 — they inject pose-sensitivity).
- **vertex–vertex mesh** (from `.ply` faces): `[dist, cos(normal_i, normal_j)]`.
- **vertex–atom** (radius ball ≤5 Å, ≤8 nearest): `[dist, cos(normal_v, unit(atom−vertex))]`.

**Gate (reproduce: `python -m masif_graph.experiments.p4_m0_gate --n 12 --seeds 1 2 3`, `logs/phase4/m0_gate.log`):**
| check | result |
|---|---|
| chains built | 180/180 across the 90-complex pool (0 failures) |
| geometry rotation gate (12 cplx × 2 chains × 3 SE(3) seeds) | **24/24 PASS**; every edge-feature max-diff **exactly 0.0**; connectivity + node features byte-identical |
| embedding rotation gate (encoder forward, orig vs rotated) | **max\|z0−z1\| = 0.00e+00** |

Invariance is **structural, not approximate**: edge features are pure distances + cos-of-normals and
connectivity is mesh-topology + rotation-invariant radius balls, so the difference is exactly zero; the
encoder reads only these (no coordinate ever enters the net). Scale is tractable (verts/chain median ~3.9k,
p95 ~8.8k, max ~9.4k; ~11.5k mesh + ~30k vertex-atom edges/chain) → **no mesh coarsening needed** for M0/M1
(a `max_vert` subsample knob is implemented for scale-up).

---

## M1 — can a from-scratch GNN re-earn the descriptor? (the feasibility gate)

**Setup.** One shared `HeteroEncoder` (torch-core ops, both-env) → per-surface-atom embedding `z` (d_out=32,
L2-normalized). Symmetric bilinear complementarity `T = ½(A+Aᵀ)` (subsumes MaSIF's flip trick; D3-A). Stage-A
loss = **InfoNCE** with in-complex (partner non-contacting atoms) + cross-complex-bank negatives, symmetrized
over both query directions, holo-only. Train 90 / held-out 60 complexes, **complex-level, mutually disjoint
and both disjoint from the 31-complex Phase-3 `m1_ids` AF3 benchmark** (verified 0 overlap). 3 seeds × 150
epochs, Kuma 1×H100 (job 3795493). Artifacts: `stageA_result_seed{0,1,2}.json`, `stageA_best_seed*.pt`.

_(Run complete — see the DEFINITIVE 3-seed table below. In-situ, held-out SC-learned rose from ~0.48 to a
best of ~0.75 but **oscillated** rather than plateaued (seed 1 collapsed 0.767→0.454 by the final epoch),
while dense-learned beat the 0.682 dense frozen ceiling and shuffled stayed ~0.50.)_

**Eval = descriptor-separation AUC** at the surface-atom level (higher score = contact), reported on TWO
positive definitions with the frozen MaSIF descriptor scored on the **identical** pos/neg pairs as the exact
ceiling:
- **sc-filtered contacts** (MaSIF's shape-complementarity-gated clean set) — frozen ceiling **0.947**,
  reproducing the Phase-3 ~0.90 holo ceiling → **this is the ~0.90 gate**.
- **dense all-vertex contacts** (all touching surface, training distribution) — frozen ceiling **0.682**.
Controls: shuffled-label ≈ 0.50 (confirmed at init on the H100 run); untrained-encoder learned AUC ≈ chance.

### Results (held-out 60, 3 seeds) — DEFINITIVE (job 3795493; `stageA_result_seed{0,1,2}.json`)

Per-seed (best epoch chosen on the held-out SC metric — a **mild optimistic bias**, so final-epoch is also
reported as the unbiased-but-noisy value):

| seed | SC learned best@ep | SC per-cplx median (best) | SC learned final | dense learned best | dense final |
|---|---|---|---|---|---|
| 0 | 0.780 @ 90 | 0.812 | 0.694 | 0.786 | 0.777 |
| 1 | 0.767 @ 50 | 0.868 | 0.454 | 0.735 | 0.472 |
| 2 | 0.700 @ 90 | 0.750 | 0.530 | 0.695 | 0.668 |
| **mean±sd** | **0.749 ± 0.035** | 0.810 | **0.559 ± 0.100** | **0.739** | 0.639 |

| reference (identical pairs) | value |
|---|---|
| **frozen MaSIF ceiling, sc-filtered** | **0.947** (reproduces Phase-3 ~0.90 → harness validated) |
| frozen MaSIF ceiling, dense | 0.682 |
| shuffled-label control | 0.505–0.510 (≈0.5 ✓) |
| untrained-encoder learned (init) | ~0.48 (≈ chance ✓) |
| median step (H100) | **20 ms/complex** |

### Feasibility verdict — PROMISING, NOT YET AT THE CEILING (data-limited; not a kill)

Three honest reads:
1. **The from-scratch GNN genuinely learns interface correspondence.** Held-out SC-filtered separation rises
   from ~0.48 (random init) to a **best-epoch 0.749 ± 0.035**, and on **dense** contacts the learned encoder
   (**0.739**) **beats the mean-pooled frozen MaSIF descriptor (0.682)**. Controls are clean (shuffled ≈0.50,
   complex-level holdout, disjoint from the AF3 benchmark). The architecture + invariant objective work.
2. **It does NOT match MaSIF's specialised clean-contact ceiling.** On the sc-filtered set the frozen
   descriptor scores **0.947**; the learned model reaches only ~0.75 best-epoch → a **~0.20 gap** on the
   metric that defines MaSIF's strength. MaSIF's descriptor is trained/tuned specifically for shape-
   complementary contacts on far more data; 90 complexes does not re-earn that.
3. **Convergence is UNSTABLE at this data scale — the dominant caveat.** Best-epoch (0.749) and final-epoch
   (0.559 ± 0.100) diverge sharply; seed 1 collapsed 0.767→0.454. A 291k-param GNN on 90 complexes (~2% of
   the 4,943-complex train set) overfits/oscillates. The best-epoch numbers carry held-out selection bias, so
   the fair one-line summary is: **held-out SC AUC swings ~0.45–0.78 across epochs/seeds, best ~0.75±0.04,
   well below 0.947, with no stable plateau.**

**Gate call.** The M1 kill-switch ("if a from-scratch GNN can't match MaSIF on holo, stop before invariance")
is **not tripped** — it learns and beats the pooled baseline — but the gate ("held-out holo→holo approaches
~0.90") is **not met at 90 complexes**. The honest diagnosis is **data-limited, not proven architecture-
limited**: the natural and cheap next test is the full-set Stage-A run. **This checkpoints the project-level
decision** (§below); it does not yet license M2 (invariance).

### Post-GO update 1 — instability was a training-recipe issue, FIXED by cosine LR (job 3795542)

Caveat #3 (unstable convergence) was the dominant weakness. Re-ran the identical clean 90/60 split with a
**cosine-LR schedule** (isolates the recipe fix from data size), 3 seeds — a decisive stability win:

| recipe | SC learned **best** (mean±sd) | SC learned **final** (mean±sd) |
|---|---|---|
| baseline (constant LR 1e-3) | 0.749 ± 0.035 | **0.559 ± 0.100** (seed1 collapsed 0.767→0.454) |
| **cosine LR** | 0.743 ± 0.010 | **0.707 ± 0.029** (final ≈ best; no collapse) |

Cosine LR lifts final-epoch AUC 0.559→**0.707** and cuts seed variance ~3–4× (final std 0.100→0.029) with no
loss of best-epoch. **The instability was the optimizer schedule, not the architecture** — the from-scratch
GNN now reaches a **stable, reproducible held-out SC AUC ≈ 0.72–0.74** on 90 complexes. The verdict sharpens:
a solid, stable ~0.73 vs the 0.947 frozen ceiling (~0.21 gap) — the remaining gap is now cleanly a
**data-scale/architecture** question for the full-set run, not a training-noise artifact.

### Stage-A full-set cost — the REAL bottleneck is reference preprocessing, not GPU

**GPU training** (established): **20 ms/complex/step** on 1×H100 → full 4,943 ≈ 99 s/epoch → 100 ep ≈ CHF 1.4,
150 ep ≈ CHF 2.1, 3 seeds ≈ **CHF 4–6**. Trivial.

**The actual prerequisite (Post-GO update 2 — honest correction to the checkpoint):** only **91 of the 4,943
training complexes have MaSIF reference data on disk** (surfaces + 80-D descriptors + input channels). The
literal full-set needs the **`.sif` pipeline (MSMS/APBS/PyMesh/descriptor-net) run on the other 4,852**.
Measured per-complex wall time from the 193 existing `logs/m0/*.log`: **median 304 s (~5 min), mean 351 s,
p90 562 s**; ~6% fail (11/193). So the full preprocessing batch (Jed CPU, **~CHF 0 — no GPU**):
| parallelism | wall-clock for 4,852 complexes |
|---|---|
| P=8 | ~51 hr (2.1 days) |
| P=16 | ~26 hr (1.1 days) |
| P=32 | ~13 hr (0.5 days) |
Cheap in money, **large in wall-clock** — this, not the GPU run, is the "full-set" commitment. Pipeline
re-verified on a fresh un-preprocessed complex (`1CTA_A_B`) this session. **No cheap shortcut:** the ~150 extra
on-disk complexes are RP/AS/AF *variants* of Phase-2/3 (repack/augment/AF3) — training on them would **leak
the eval set** (caught + aborted; the holo filter must exclude `RP`/`AS`/`AF`, not just `AF`).

- **Streaming-loader prereq** (unchanged): `p4.train` loads all complexes upfront (90 ≈ 0.25 GB fine);
  4,943 npz ≈ 14 GB → needs a lazy per-complex loader before the full training run. Small refactor.

---

## Checkpoint before the full ~4,943-complex Stage-A run — POSTED, awaiting go
The full-set run is the budget-gated step (handoff §1). A spend checkpoint + cost projection + the streaming-
loader prerequisite is posted in `docs/09-phase4-user-comment.md`. Not launched this session.

**Arc-1 bottom line:** M0 (heterogeneous invariant graph) is built and its rotation gate passes exactly;
M1's from-scratch GNN trains, is provably SE(3)-invariant, learns correspondence and beats the pooled frozen
descriptor on dense contacts, but reaches only ~0.75 (unstable) vs MaSIF's 0.947 sc-filtered ceiling on 90
complexes. Feasibility = **promising, data-limited, gated on the cheap full-set scale-up** before any
invariance (M2) work.


## SCALE-UP RESULTS — full-set Stage-A (2×2: train-pos × 2 seeds) — AUTO-COLLECTED

Trained on **4809 complexes** (vs 90 in M1), held-out 60 (disjoint), cosine LR + streaming. Frozen ceilings on identical pairs: SC 0.947, dense 0.682. Artifacts: , .

| train-pos | held-out SC best (mean±sd) | SC final | dense best | vs frozen SC 0.947 |
|---|---|---|---|---|
| dense | 0.791 ± 0.011 | 0.415 | 0.808 | gap +0.155 |
| sc | 0.822 ± 0.037 | 0.399 | 0.678 | gap +0.125 |

Shuffled control ≈ 0.50 (✓). Baseline (90 complexes, M1): SC best 0.749±0.035.

**Data-scaling read (auto, numbers-based):** ~~DATA-LIMITED (gap closing): scaling 90→4809 lifted held-out SC
AUC to 0.822 … more data helps → proceed to M2.~~ **SUPERSEDED — over-optimistic; see reconciliation below.**

_(Auto-collected by scripts/p4_scaleup_collect.sh. The auto-read averaged the per-run **best epoch** only; a
conductor self-verification against the full per-epoch histories overturns its interpretation — below.)_

### ⚠️ CONDUCTOR RECONCILIATION (self-verified vs full histories) — TRAINING DIVERGED AT SCALE

Reading the complete per-epoch curves in `scaleup_*_seed*.json` (not just `best_sc_learned_randneg`) shows the
"best" values are **transient spikes of a diverging optimization, not a converged capability**:

| run | stable early phase | then | **final-epoch SC** | train-loss path |
|---|---|---|---|---|
| dense s0 | 0.72→**0.80** by ep20 (loss ~7.6) | collapse ep35+ | **0.42** | 7.6 → 13 → 8 |
| dense s1 | 0.72→0.78 by ep15 | oscillates 0.29–0.72 | **0.41** | 6.9 → 15 → 11 |
| sc s0 | chaotic; spikes 0.86@ep40, 0.83@ep55 | — | **0.37** | 7 → 18 → 7 → 20 |
| sc s1 | 0.74→0.79@ep40 | — | **0.43** | 8 → 17 → 20 |

Three facts kill the auto-verdict: **(1)** all four **final epochs collapse to ~0.37–0.43 — BELOW the 0.50
shuffled control** (the learned score anti-correlates with contacts by the end). **(2)** Train loss repeatedly
**explodes to 12–20** from a stable ~7.6 → the optimization diverges; cosine-LR + grad-clip 5.0 did not contain
it at scale (it *did* stabilize 90 complexes: final 0.707). **(3)** "best" is the max over **48 noisy eval
points** (12 evals × 4 runs) → heavy selection bias. So **0.822 is not a held-out AUC** — it is the luckiest
spike of an unstable run.

**Honest verdict — INCONCLUSIVE (recipe-unstable at scale), NOT "data-limited / gap-closing."**
- *Genuine signal:* the **stable early phase** (first ~15–25 epochs) reproducibly reaches held-out SC
  **~0.78–0.80 on dense across both seeds** — modestly above the 90-complex 0.749, so more data *plausibly*
  helps. But this is suggestive, not established, because the same runs then diverge.
- *Dominant new finding at scale:* a **training-stability failure** (LR / gradient-scale / cross-complex
  negative-bank dynamics over 4809 complexes × 288k steps), **not** a capability ceiling. The naive scale-up
  therefore **cannot** decide data-limited vs architecture-limited.
- *Action:* **do NOT proceed to M2 on the auto read.** A stabilized recipe (lower peak LR, tighter grad-clip,
  early-stop at the stable plateau) is required to get a real converged AUC first. **Stabilization run launched
  — see below / the running log.**

_(Every number above traces to a committed `scaleup_*_seed*.json` + the history-dump command in the log.
Redundant double-submit `3798265-68` was caught on resume and cancelled — not used for any number here.)_

### Stabilization run (jobs 3798320 dense-s0, 3798321 sc-s0) — INSTABILITY CONFIRMED, recipe-robust

Calmer recipe to test whether the stable-early ~0.80 becomes a real plateau: peak **LR 1e-3→3e-4**, **grad-clip
5.0→1.0** (new `--grad-clip` flag), cosine, 40 ep, same data/held-out/guard. It did **not** stabilize:

| run | train-loss path | held-out SC per eval (ep5→40) | best | final |
|---|---|---|---|---|
| dense s0 | 9.7 → 12.4 → 9.7 (never settles to the ~7.6 low) | 0.62, 0.45, 0.72, 0.62, 0.41, 0.40, 0.62, 0.59 | 0.719 | 0.586 |
| sc s0 | 11 → **19** (still exploding at ep40) | 0.77, 0.79, **0.21**, 0.83, 0.78, 0.82, 0.73, 0.54 | 0.827 | 0.538 |

- **No stable plateau in either regime.** Loss never reaches the stable ~7.6; the sc run's loss is *still
  climbing* (→19) at the final epoch. SC still swings chaotically (sc: 0.83→0.21→0.83).
- Lowering LR made **dense strictly worse** (best 0.72 vs 0.80 at lr 1e-3) — it *undertrained*, it did not
  stabilize. So neither "steps too big" nor "gradients too big" is the cause.

## FINAL SCALE-UP VERDICT — data-scaling question **INCONCLUSIVE**; the result is a **training-stability failure**

Across **two independent recipes (lr1e-3/clip5 and lr3e-4/clip1), 6 runs total**, every run diverges: stable for
~15–25 epochs then loss explodes (12–20) and held-out AUC collapses **below the 0.50 shuffled control**. Peaks
touch ~0.80–0.83 in both train-pos regimes — modestly above the 90-complex 0.749, so **more data *plausibly*
helps** — but never as a stable, selectable, deployable result.

- **We CANNOT decide data-limited vs architecture-limited** from the naive scale-up: the optimization breaks
  before it converges. The dominant finding at 4,809 complexes × ~288k steps is that **the InfoNCE +
  cross-complex-negative-bank recipe that worked on 90 complexes does not scale**.
- **Instability is robust to LR + grad-clip** ⇒ it is an **objective/optimization problem**, not a step-size
  problem. Leading suspects (untested): learnable-**temperature runaway**, **stale memory-bank negatives**
  (no momentum encoder), or **representation collapse**; aggravated by 1-complex-per-step + sparse sc-positives.
- Controls stayed valid throughout: shuffled ≈ 0.50; frozen ceilings 0.947 (sc) / 0.682 (dense) reproduced on
  identical pairs every eval; complex-level holdout; the 31 `m1_ids` never entered training (leak-checked by id
  **and** PDB-stem).
- **Do NOT proceed to M2 (invariance).** Required next step is a **recipe redesign, not another knob**:
  freeze/EMA the temperature; drop or shrink the bank / add a momentum encoder; batch several complexes per
  step; add LR warmup; and/or early-stop at the stable-early plateau. A cheap **diagnostic** (log τ, grad-norm,
  and embedding variance on one short run) should name the trigger *before* investing in the redesign.

_Spend this session ≈ CHF 15–17 of 100 (Jed preproc ~7 + precompute ~0.5 + Kuma: 2×2 ~6 + stabilization ~2;
redundant 3798265-68 cancelled early ~0). Every number traces to a committed `*_seed*.json`._

### Diagnostic — ROOT CAUSE NAMED: representation collapse + temperature-floor runaway (jobs 3798692 dense-s0, 3798693 sc-s0)

Instrumented `p4.train` to log per-epoch **τ**, **‖T‖₂** (bilinear spectral norm), **pre-clip grad-norm**, and
**z_std** (embedding spread), then re-ran the *unstable* recipe (lr1e-3/clip5/cosine). The pathology is present
from the earliest epochs and identical in character across both runs (values from ep6–14; the divergence is not
a late surprise — the network is broken from the start and dense merely *masks* it longer):

| metric | healthy | dense (3798692) | sc (3798693) |
|---|---|---|---|
| **z_std** (embedding spread; d=32 sphere) | ~0.18 | **0.003** | **0.0001–0.001** |
| **τ** (learnable InfoNCE temperature) | ~0.1 | **0.0100 = clamp floor**, from ep6 | **0.0100 = floor**, from ep8 |
| **‖T‖₂** (bilinear spectral norm) | O(1) | 13–14, rising | 18→22, rising |
| **grad-norm max** (pre-clip) | ~5 | 55–105 | **up to 5.6×10⁵** |

**Root cause = representation collapse, with temperature runaway + unconstrained T as amplifiers.** The
L2-normalized embeddings collapse to near-identical vectors (`z_std ≈ 0.003` vs a healthy ~0.18) almost
immediately — *despite* the objective's assumption that L2-norm + InfoNCE "can't collapse" (`objective.py`
docstring). The optimizer compensates by driving τ to its **0.01 floor** and inflating ‖T‖ (14→22), which
amplifies the vanishing embedding differences ~100× → catastrophic gradient spikes (sc: up to **5.6×10⁵**) →
divergence. **Dense holds together longer** (plentiful positives; grad max ~100, SC still 0.77@ep10) so it
reaches ~0.80 before tipping; **sc collapses immediately** (sparse positives; SC 0.32@ep10, loss 10–12). This is
why tighter grad-clip (§14) could not fix it — the trigger is *collapse + τ-floor*, not raw gradient magnitude.

**Targeted redesign (priority order) — objective/architecture changes, not knobs:**
1. **Stop the collapse** — add an explicit anti-collapse term (VICReg variance/covariance, or a hypersphere
   uniformity loss), or switch to a stop-gradient/predictor scheme (SimSiam/BYOL). L2-norm alone is insufficient.
2. **Fix the temperature** — a learnable τ rides the floor; freeze it at ~0.1–0.2 (or raise the floor + weight-decay `log_tau`).
3. **Constrain T** — spectral-normalize or weight-decay the bilinear form so ‖T‖₂ stays O(1).
4. Only then revisit LR/schedule. Re-run the diagnostic (same instrumentation) to confirm z_std stays ~0.1 and τ stays off the floor before trusting any new AUC.

**CONFIRMED through ep40 (both runs complete, `diag_{dense,sc}_seed0.json`):** z_std stays **0.001–0.003** the
whole run (dips to 0.0004); τ sits on the **0.01 floor** throughout (dense makes one brief thrashing excursion
to ~0.02 during the blow-up, then returns); **‖T‖₂ grows monotonically and unbounded** — dense 11→28, sc 10→**39**;
grad-norm max spikes to **5.6×10⁵**. Final held-out AUCs are unreliable noise (dense collapses to 0.47; sc's 0.81
"final" coincides with loss 13.5 — a chaotic fluke, reinforcing that best/final are not real capability). The
three suspects are settled: **representation collapse is primary and immediate; temperature-floor + unbounded ‖T‖
are the amplifiers.** Every number traces to the committed `"diag"` array per run.

