# Phase 8 Stage A — build log (append-only)

Plan: `docs/24-phase8A-plan.md`. Contract: `docs/23-phase8-design.md`.
Ends at the **PAUSE** for the D8-12 decision (apo-prediction method + holo:apo ratio) — Stage A
produces a *recommendation*, never the choice.

Spend is tracked per step. Jed CPU ≈ CHF 0.005/core-h; Kuma H100 ≈ CHF 0.52/GPU-h.

---

## 2026-08-12 — step 0: tooling verified, scaffold, A1 launched

**Tooling closed out** (was the open item from the plan review):
* **TMalign INSTALLED** — compiled from source (`zhanggroup.org/TM-align/TMalign.cpp`,
  v20220412) into `/work/upthomae/Meng/conda_envs/masif-graph/bin/TMalign`. Compiling beat a conda
  solve: no dependency churn in a working env. **Verified functionally**, not just "it exists":
  self-alignment TM = 1.00000, an unrelated pair 0.155 / 0.483. `tmtools` is no longer needed.
* **Chai found** — `/home/ymeng/miniconda3/envs/chai`, `chai_lab` 0.6.1 on torch 2.6.0+cu124.
  My earlier "not importable" was a bad search path (I looked only under `/work/.../conda_envs/`).
* **Shared-MSA path verified** — our AF3 `<chain>_data.json` carries the alignments inline as a3m
  (`1bq4_A`: `unpairedMsa` 16,705 seqs / 5.9 MB, `pairedMsa` 50,000 / 19.8 MB, 4 templates), and
  `chai_lab...aligned_pqt.a3m_to_aligned_dataframe` converts a3m → chai `.aligned.pqt`.
  `run_inference(msa_directory=)` consumes it. So A0 compares **inference on one identical MSA**
  instead of accidentally benchmarking five different MSA searches. `docs/24` §2.2 updated.

**Scaffold**: `src/masif_graph/p8/`, `logs/phase8A/{a0..a4}`, this log.

### A1 — design deviation from `docs/24` §3.1, recorded before running

The plan proposed rebuilding eval graphs with **backbone-only atom nodes**. That is unsound as
written: the encoder emits one row per *surface atom* (`z = readout(ha[surf_node_idx])`) and ~60% of
surface atoms **are** sidechain atoms (measured: 0.598 on `1A99_C_D` p1). Deleting them renumbers
`surf_node_idx`, invalidates `Rec.inter`, and leaves the ablated arm retrieving over a smaller,
different patch — so any drop would be confounded with patch size.

Replaced with a **row-preserving ablation ladder** (`p8.ablate`) that cuts edges and destroys
features but never deletes a node, so every arm scores the identical rows:

| ablation | what it destroys |
|---|---|
| `none` | nothing — must reproduce the published number (reproduction check) |
| `sc_feat` | sidechain atom **chemistry** (feature rows permuted among sidechain atoms) |
| `sc_edge` | sidechain **connectivity** (aa edges incident to a sidechain atom cut) |
| `sc_all` | both, plus va edges into sidechain atoms — sidechain atoms fully isolated |
| `bb_feat` | backbone atom features (comparison channel) |
| `vert_feat` | the surface chemistry channel |
| `all_feat` | **positive control** — every node feature in the graph. MUST collapse. |

`all_feat` is the guardrail: if retrieval survives destroying every node feature, the harness is not
measuring what it claims and no sidechain-blindness conclusion may be drawn from the other rows.
Pre-registered before seeing any result.

Verified each ablation actually bites (`1A99_C_D` p1, 2680 atoms / 1193 surface rows):
`sc_feat` 1265 atom rows changed; `sc_edge` aa 5496→2726; `sc_all` aa 5496→2726 **and** va
42256→18810; `vert_feat` 5433 vertex rows; `all_feat` 2511 + 5433. Row count preserved in all
(z stays (1193, 32)); the source graph is never mutated.

Also measured, alongside retrieval: **per-atom embedding displacement** split by whether the surface
row is a sidechain or backbone atom, plus Spearman(displacement, `flex_depth`) under `sc_all` —
the closest thing the current encoder has to an implicit per-atom σ (D8-9 / D8-19).

**Two seeds** (D8-11): `phase6C/ret_ppionly_best.pt` and `phase7/ret_ppionly_s1_best.pt`, whose
published HH top-5 are 0.651 / 0.638 (Phase-7 §2 reports 0.644 ± 0.007 — matches).

**Launched**: `sbatch --array=0-1 scripts/p8a_a1.sbatch` → **job 66063328**, est. CHF 1.41 (that is
the `--time=8h` worst case; the Phase-7 gate ran axis 1 in minutes, so actual will be far less).
Expected signal: `logs/phase8A/a1/{disp_*.json, ret_*_<ablation>.json}`.

**Code change**: `p5.retrieval_bench.run()` gained an optional `transform=` hook so the ablated arms
run through the *identical* benchmark rather than a reimplementation. Non-breaking (default `None`).

Spend so far: ~CHF 0 (login-node work) + job 66063328 pending.

---

## 2026-08-12 — A1 COMPLETE (both seeds), A2 COMPLETE, A3 launched

### A1 — verdict: the encoder is **NOT** sidechain-blind

Job 66063328, both seeds, 7 ablations each. Retrieval (HH top-5, 287-clean, DB=538, chance 0.0093):

| ablation | s0 | s1 | mean | Δ vs none |
|---|---|---|---|---|
| `none` | 0.651 | 0.638 | **0.644** | — |
| `sc_feat` | 0.353 | 0.257 | 0.305 | **−0.339** |
| `sc_edge` | 0.600 | 0.357 | 0.479 | −0.165 |
| `sc_all` | 0.294 | 0.134 | 0.214 | **−0.430** |
| `bb_feat` | 0.543 | 0.416 | 0.480 | −0.164 |
| `vert_feat` | 0.387 | 0.301 | 0.344 | −0.300 |
| `all_feat` | 0.043 | 0.020 | 0.032 | −0.612 |

**Both required checks pass.** `none` reproduces the published 0.651 / 0.638 exactly (Phase-7 §2,
0.644 ± 0.007) — the harness is the same benchmark, not a lookalike. `all_feat` collapses to 0.032
with median rank 166, i.e. ~chance — the ablation harness can detect destruction, so the other rows
are interpretable.

Isolating sidechain atoms costs **70% of everything the graph carries** (−0.430 of −0.612).
Sidechain chemistry matters **more** than backbone chemistry (−0.339 vs −0.164) and more than the
entire surface-vertex channel (−0.300). The unflattering explanation for Phase 5's robustness —
"robust because it never reads sidechains" — is **refuted**.

**Displacement probe** (n=52 complexes, per-surface-atom, dimensionless vs the natural z spread):

| ablation | relDisp sidechain rows | relDisp backbone rows |
|---|---|---|
| `sc_feat` | 0.601 / 0.429 | 0.249 / 0.173 |
| `sc_all` | 0.833 / 1.398 | 0.445 / 0.296 |
| `bb_feat` | 0.163 / 0.140 | 1.387 / 2.175 |

The response is **spatially specific**: sidechain ablations move sidechain rows ~2-5x more than
backbone rows and `bb_feat` cleanly reverses it. Independent confirmation the ablations do what
they claim.

**Implicit σ (D8-9 / D8-19)**: Spearman(displacement under `sc_all`, `flex_depth`) = **+0.381**
(s0) / **+0.599** (s1), p≈0, n=90,121. The encoder already moves more where sidechains are more
rotatable. **Caveat, stated up front:** `flex_depth` is itself input feature col 22, so this shows
the representation is *organised by* flexibility, not that it learned flexibility from physics. Even
so it means a σ head would be predicting something already partly present rather than from scratch.

**What A1 does NOT show.** It perturbs sidechain *identity/chemistry*, whereas AF3 apo models keep
identity and change *conformation*. So A1 says "reads sidechains" and Phase 5 says "robust to
conformation" — compatible, and together the property we want. The direct conformational test is
**A1.2 (FASPR repack)**, which is therefore not optional.

Seed spread is wide (`sc_all` 0.294 vs 0.134); the **ordering** is identical across seeds, the
magnitudes are not. Reported as ranks, not as precise deltas.

### A2 — verdict: pockets mostly do NOT close; the tail is sidechain-mediated

298/298 complexes, **0 failures**.

* buried fraction: holo median 0.851, AF3 median 0.840, **ratio median 1.001** (p05 0.634, p95 1.124)
* clashes @2.0 Å: holo median 0 (p95 **0**) vs AF3 median 1 (p95 19) — the holo control is exactly 0,
  so the clash metric is calibrated rather than assumed
* AF3 clash split: backbone median 0, **sidechain median 1**
* **pre-registered "collapsed": 60/298 = 20.1%**

So apo input does not broadly destroy the pocket — the median complex is unchanged — but a **fifth**
of cases do degrade, and the degradation is **sidechain-mediated**, which a repack step can recover.
That is a Stage-B design input, not a blocker. Added a `ca_rmsd_in_frame` confound control so
"collapsed" can be checked against "badly superposed" rather than assumed distinct.

### A3 — launched (job 66063365, 4 checkpoints)

Built with the controls first, and they changed the design:

* **ORACLE control** (true native contacts as correspondences): fnat 0.971, iRMSD 2.0 Å, **100%
  success**. The pose machinery is sound, so a failure of the learned arm is a result.
* **random control**: 0% success, fnat 0.000.
* **learned arm: 0% success in all four cells**, iRMSD ~25 Å, correspondence precision 0.0045–0.0065
  (15–23× chance, but nowhere near usable).
* **Mechanism found**: the atom-level score matrix is **hub-collapsed** — for ~511 query atoms only
  ~51–64 distinct partners are *ever* the argmax, and there are **2–3 mutual-best pairs** in a
  ~500×500 matrix. Chain-level retrieval works because `median_i max_j` aggregates over many atoms;
  that statistic is indifferent to hubs, so nothing in the training objective ever penalised them.

Both Stage-A (atom-level contrastive) and Stage-B (retrieval-tuned) checkpoints are being run —
scoring only the retrieval-tuned model would have been an unfair test of pose prediction, since only
Stage A was ever trained on an atom-level objective. Stage A is ~4× better on correspondence
precision (0.0055 vs 0.0015) and still fails.

Spend: A1 + A3 + A2 ≈ CHF 1.5 so far.

---

## 2026-08-12 — A3 COMPLETE (4 checkpoints), A0 infrastructure up

### A3 — verdict: rigid pose from Stage-1 scores **fails completely**; F3 compute is **not** the problem

Job 66063365, n=269 complexes, 4 checkpoints (Stage-A ×2 seeds, Stage-B ×2 seeds).

| ckpt | cell | success | fnat med | iRMSD med | corr prec | ×chance | hubs | mutual-best |
|---|---|---|---|---|---|---|---|---|
| stageA_s0 | HH | 0.000 | 0.045 | 21.2 | 0.0010 | 6 | 77 | 3 |
| stageA_s0 | AA | 0.000 | 0.049 | 21.9 | 0.0010 | 3 | 77 | 3 |
| stageB_s0 | HH | 0.000 | 0.059 | 21.3 | 0.0010 | 7 | 370 | 211 |
| stageB_s1 | AA | 0.000 | 0.068 | 21.6 | 0.0020 | 8 | 392 | 277 |

**0% success in every cell of every checkpoint** (pre-registered: fnat ≥ 0.3 AND iRMSD ≤ 4 Å).

Controls, identical in all four runs: **ORACLE** (true native contacts as correspondences) →
**100% success, fnat 0.982, iRMSD 2.0 Å**; **random** → 0%, fnat 0.000. The pose machinery is
correct and the failure belongs to the correspondences, not the fitter.

**Correction to the note above.** From an 8-complex smoke test on a Stage-A checkpoint I wrote that
hub collapse was the mechanism. At full scale that is only half right: **Stage-B checkpoints are
NOT hub-collapsed** — 370 distinct argmax partners and 211–277 mutual-best pairs, versus 77 and 2–3
for Stage A. They still score 0%. So plentiful, well-distributed correspondences are not sufficient;
the atom-level scores are simply not spatially discriminative (precision 0.001–0.002, 3–9× chance).
Hub collapse is a Stage-A pathology, not the general explanation.

**Fork F3 is settled, and favourably**: **0.57–0.64 s per pair** with embeddings precomputed →
**~6–7 core-hours for a 40k-partner screen**. Stage 2 is affordable. What it is not, on the current
Stage-1 scores, is *accurate* — which moves the problem from the compute budget to the objective,
the same conclusion Phase 7 reached from a different direction.

### A0 — infrastructure complete, GPU arms launched

* **Test set**: 30 chains, 6 balanced strata (5 each), length 62–863. Measured the candidate pool
  first — pLDDT p01 = 77.5, p50 = 95.3, **1 of 883 chains below 70** — so the corpus has
  essentially no low-confidence chains, and a median split would have given the calibrated-spread
  metric no dynamic range. Switched to **quartile extremes**. The small contrast is reported as a
  limitation, not hidden.
* **Shared MSA exported and verified**: 30/30 chains → 60 a3m → 30 chai `.aligned.pqt`,
  merged depth median 43,014 (min 28, max 69,491). 28 unique files because two chain pairs share a
  sequence (2NXM_A/B, pl4b2i/pl4b32) — correct, not a bug.
* **Bug caught before it mattered**: chai resolves an MSA as `msa_directory /
  expected_basename(sequence)` — a sequence hash — and when the file is missing it only *logs a
  warning* and falls back to a single-sequence MSA. My first conversion wrote
  `{cid}.unpaired.a3m.aligned.pqt`, which chai would never have found: the "chai + shared MSA" arm
  would silently have been an MSA-free arm and the benchmark would have been meaningless. Fixed to
  use chai's own `expected_basename`, merging unpaired+paired into one frame, and the runner now
  captures chai's own "MSA found … depth=" log line into the result so the claim is checkable.
* **AF3 NSAMP=5** submitted on Kuma (job 4077883, 5 array tasks, est. CHF 5.17 worst case).
  Writes to a new dir; the Phase-5/7 NSAMP=1 models are untouched.
* **Chai** weights (1.1 GB) + ESM2-3B fp16 pre-staged to `/work/upthomae/Meng/chai_downloads`
  because compute nodes may lack internet.
* **Protenix is NOT installed** — the `conda_envs/protenix` env exists with torch 2.7.1 but no
  `protenix` package, so my plan's "zero-install candidate" was wrong. It falls under the §9
  install timebox and will be reported as "not evaluated" if it does not come up cheaply.

Spend: Jed ≈ CHF 1.5; Kuma AF3 pending (≤ 5.17 worst case).

---

## 2026-08-12 — A0 complete, A1.2 complete, A4 mined + probed

### A0 — AF3 vs Chai-1 on ONE shared alignment

| method | TM | aligned RMSD | spread | ρ(RMSF,pLDDT) | s/chain |
|---|---|---|---|---|---|
| AF3, 5 samples | 0.978 | 0.845 | 0.220 | −0.613 | ~60–70 |
| Chai-1, 5 samples, shared MSA | 0.945 | 1.130 | 0.435 | −0.650 | 73.8 |
| Chai-1, 5 samples, MSA-free | 0.953 | 1.385 | 0.783 | −0.628 | 74.2 |

Paired per chain (n=30, Wilcoxon), because the medians mislead: chai(MSA) − AF3 is
**TM −0.001, p=0.38** — indistinguishable. MSA-free costs TM −0.012 (p=0.011) and +0.59 Å
(p=9.7e-5). The spread advantage for chai is **not** significant (p=0.5 / 0.084).

Shared-MSA consumption verified rather than assumed: **30/30** `msa_found` in the MSA arm and
**0/30** in the MSA-free arm.

**AF3 at NSAMP=5 costs the same as NSAMP=1** (~60–70 s/chain), because MSA + trunk dominate.
Phase-7's `NSAMP=1` was leaving four free conformers on the table.

### A1.2 — the conformational test: repack ≈ 91% of the AF3 perturbation

99/100 repacks succeeded; 99/99 graphs rebuilt. **A 14-D/26-D bug was caught here**: `p4.precompute`
built Phase-4 14-D atom features, which the 26-D Phase-6C/7 encoders cannot consume
(`mat1 711x14 vs mat2 26x64`). Added `--unified` and verified `atom_feat (2680, 26)` on one complex
before rebuilding all 99.

| quantity | seed 0 | seed 1 |
|---|---|---|
| rel. displacement, FASPR repack | 0.322 | 0.206 |
| rel. displacement, AF3 | 0.353 | 0.224 |
| **ratio repack / AF3** | **0.914** | **0.917** |

**A fixed-backbone sidechain repack moves the embedding ~91% as far as a full AF3 re-prediction**,
in both seeds. Almost everything the encoder feels when handed an apo structure is **sidechain
rearrangement**, not backbone. Combined with A1 (the encoder reads sidechains) and A2 (apo pocket
loss is sidechain-mediated), three independent measurements point at the same lever.

Retrieval with the repack substituted into the AF3 slot (n=90, DB=180, chance top-5 0.028):

| cell | s0 | s1 |
|---|---|---|
| HH holo–holo | 0.733 | 0.694 |
| AA repack–repack | 0.700 | 0.706 |
| shuffled control | 0.033 | 0.033 |

Robust to repacking as well as to AF3 — the drop is −0.033 / +0.012, inside seed spread.

**Correction to the A1 "implicit σ" reading.** Under `sc_all` (feature destruction) Spearman
(displacement, `flex_depth`) was **+0.38 / +0.60**. Under an actual repack it is **−0.18 / −0.18** —
the **sign flips**. So the encoder does *not* move more where sidechains are more rotatable when the
perturbation is a real conformational change; if anything it moves less. One reading is that it has
learned to be *insensitive* at flexible positions, which is good for robustness and bad for a σ
head. Either way, **D8-9 cannot assume the σ signal is already present** — the A1 correlation was an
artefact of the perturbation type, and I should not have called it an implicit σ without this test.

### A4 — mining and the BSA baseline

393/400 entries mined, **1,755 interfaces (1,460 bio / 295 crystal)**. BSA median: bio 1,828 Å²,
crystal 615 Å².

| arm | AUROC | AP(crystal) |
|---|---|---|
| BSA only | **0.827** | 0.474 |
| + contacts + chain sizes | 0.857 | 0.546 |
| BSA only, shuffled labels | 0.526 | 0.177 |
| structural, shuffled labels | 0.512 | 0.173 |

Shuffled controls collapse to chance with AP at prevalence (0.168), so the folds and the CV are
sound. **This is the D8-7 hard control, now quantified: Stage 3 must beat AUROC 0.827 / AP 0.474.**
Interface area alone is a strong predictor of "biological", which is exactly why it had to be
measured before building anything on top of it.

Labelling used the identity-operator rule (both chains under IDENTITY inside one BIOMOLECULE), not
chain-list co-occurrence, so "chain A plus a symmetry copy of A" is not mislabelled as a biological
A–B pair.

Surfaces for the embedding arm are building (80 entries / 525 interfaces / ~441 chains).

---

## 2026-08-12 — A4 embedding arm, Stage A COMPLETE

471/525 selected pairs surfaced in the end (29 pipeline failures, rest cancelled); graphs were
built from the 455 available at that moment, of which 413 interfaces were
scorable with the Stage-1 bilinear score. **All arms compared on the identical 413 rows** — the
first run silently dropped the embedding arm because 42 interfaces lacked features, which would
have compared arms on different populations.

| arm | s0 | s1 |
|---|---|---|
| BSA only | 0.841 | 0.841 |
| + contacts + chain sizes | **0.863** | **0.863** |
| + Stage-1 embedding summaries | 0.852 | 0.855 |
| shuffled | 0.531–0.542 | 0.531–0.542 |

**The embeddings add nothing** (both seeds slightly below the structural arm). Second independent
indication, after A3, that the current Stage-1 representation serves chain-level retrieval and not
the atom- or interface-level judgements Stages 2 and 3 require.

The last two A4 surface array tasks were **cancelled** after 2 h once both probe seeds had run —
more pairs would not have changed a two-seed null.

**Stage A is complete.** `docs/25-phase8A-results.md` written; `logs/PHASE8A_DONE` touched.
Total ≈ CHF 12 (Jed ≈ 6, Kuma ≈ 6), no training, nothing irreversible.

**PAUSED for the user**, per plan §9, on two decisions:
1. **D8-12** — apo-prediction method and holo:apo ratio. Recommendation in docs/25 §8.
2. **The A3 consequence** — Stage 2 cannot be built on the current Stage-1 scores. Options in
   docs/25 §8.1. Not acted on unilaterally because it reorders the phase.

---

## 2026-08-12 — A3 harness bias found on user challenge

The user pointed out that RANSAC-Kabsch on *atom centres* co-locates contacting atoms, which is
physically impossible — MaSIF gets away with it on surface vertices because two contacting surfaces
are ~1 Å apart, but atom centres are ~4 Å apart and must never overlap.

**Measured, and they are right** (40 complexes):

| quantity | value |
|---|---|
| true separation of "contacting" atom-centre pairs | median **3.79 Å** |
| native inter-chain closest approach | 2.55 Å |
| closest approach after the ORACLE Kabsch fit | **1.06 Å** (interpenetrated) |
| ORACLE iRMSD vs true pose | **1.90 Å** ← a floor the harness imposes |

So the oracle's 1.9 Å is the *bias*, not its accuracy, and the pre-registered 4 Å threshold really
allowed only ~2.1 Å of correspondence-driven error.

**Attempted fix failed and was reverted.** A global de-clash (slide the partner out along chain 1's
interface normal until contact distance is restored) took the oracle from 1.9 → **10.3 Å** and
success 1.000 → 0.125. The co-location bias is per-contact along each pair's own local normal, not a
single global translation, so one shift wrecks a curved interface. Removed rather than shipped; the
reason is recorded in the module docstring so it is not retried blind.

**Correct fix, for when it matters:** fit at the VERTEX level. Contacts were defined by vertex
proximity at 1.0 Å (`precompute`: `vertex_contacts(..., pos_cutoff=1.0)`), so vertex-level alignment
carries ~4× less bias. Vertex coordinates are in the reference precomputation, not our npz.

**Conclusion unchanged:** the learned arm is at 21–23 Å against a 1.9 Å floor. But the harness is
stricter than intended and must be fixed before iRMSD is used to judge an improved Stage 1.
