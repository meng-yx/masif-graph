# Phase 8 — Stage A: implementation plan

> Implements §6 Stage A0–A4 of `docs/23-phase8-design.md`. Stage A is **diagnostics + a method
> benchmark**: nothing is trained, no corpus is generated. It ends at the **PAUSE** where the user
> chooses the apo-prediction method and the holo:apo ratio (D8-12).
> Written to be executed autonomously; §9 states exactly what I decide alone and what I stop for.

## 0. What Stage A delivers

| id | question it answers | can invalidate |
|---|---|---|
| **A0** | which method generates apo conformers, at what cost and quality | the CHF 150–560 generation run |
| **A1** | is the Stage-1 encoder modelling flexibility, or **ignoring sidechains**? | Stage 1 as a foundation |
| **A2** | do **cryptic pockets** close in apo models, bounding apo P–L screening? | the P–L branch's ceiling |
| **A3** | how good — and how *fast* — is rigid pose prediction from Stage-1 scores? | Stage 2 / fork F3 |
| **A4** | does *any* bio-vs-crystal signal exist in current embeddings? | the funnel's premise |

**Done** = `docs/25-phase8A-results.md` written with every number traceable to a committed artefact
under `logs/phase8A/`, each sub-stage's verdict stated honestly, a **recommendation** (not a
decision) for D8-12, and `logs/PHASE8A_DONE` touched.

## 1. Verified tooling (checked 2026-08-12)

| need | status | plan |
|---|---|---|
| FASPR (sidechain repack) | `/work/upthomae/Meng/tools/FASPR/FASPR` **exists** | A1 secondary test, as in Phase 2 |
| SASA / BSA | **biotite `sasa` works**; `freesasa` absent | use biotite; no install |
| biological assemblies | **`REMARK 350` present** in our cached PDBs (19 lines/entry) | parse from PDB format — **no gemmi needed** for ASU-only |
| gemmi | absent | not required for Stage A (only for full symmetry expansion, Stage B) |
| TM-align | **INSTALLED** — `conda_envs/masif-graph/bin/TMalign` (v20220412, compiled from source) | verified: self-alignment TM = 1.00000, cross-pair 0.155/0.483. No `tmtools` needed |
| AF3 | pipeline exists (`p7_af3_msa/infer.sbatch`), **MSAs already on disk for 298 P–L chains** | re-run inference with `NSAMP=5`; MSA cost already paid |
| Chai-1 | **conda env `chai` at `/home/ymeng/miniconda3/envs/chai`** — `chai_lab` 0.6.1, torch 2.6.0+cu124 | zero-install. (My earlier "not importable" was a wrong search path: I looked only under `/work/.../conda_envs/`.) |
| Protenix | **conda env exists** (`conda_envs/protenix`) — open AF3 reproduction | include as a zero-install candidate |
| Boltz-2 | absent | `pip install boltz` into a fresh env; can consume the AF3 a3m |
| ESMFold | absent | install only if cheap; **1:1 candidate only** (deterministic → no ensemble) |
| PocketMiner | conda env exists | *optional* cross-check for A2, not a dependency |

Environment work is done **first**, timeboxed (§9); any candidate that will not install in the box is
reported as "not evaluated" rather than blocking the stage.

## 2. A0 — apo-prediction method benchmark

### 2.1 Test set (n = 30 chains)
Drawn from chains where we already hold **holo structure + AF3 model + MSA**, stratified so the
spread metric is measured in both regimes:
* length: 10 short (<150 aa), 10 medium (150–350), 10 long (>350);
* AF3 confidence: half high-pLDDT, half low — Phase-3 M2 measured inter-sample CA-RMSD of ~0.1 Å for
  confident chains and up to ~15 Å for uncertain ones, so a set of only-confident chains would make
  every method look identically "good".

Written to `logs/phase8A/a0/testset.txt` with the stratification recorded.

### 2.2 Methods, on a shared MSA

**Verified 2026-08-12: every MSA-based method can be run on the *same* MSA, so the benchmark
isolates inference from search.** Our AF3 `<chain>_data.json` carries the alignments inline as a3m
strings (`1bq4_A`: `unpairedMsa` 16,705 seqs / 5.9 MB, `pairedMsa` 50,000 seqs / 19.8 MB, plus 4
templates), and `chai_lab.data.parsing.msas.aligned_pqt.a3m_to_aligned_dataframe` converts a3m →
chai's `.aligned.pqt`. `run_inference(..., msa_directory=)` then consumes it; the user's earlier
`Chai_MSA/test/out/msas` run confirms the path works end-to-end.

This matters because MSA search, not inference, is the dominant cost — comparing methods that each
ran their own search would measure the search tool, not the structure predictor.

| method | MSA | samples | notes |
|---|---|---|---|
| AF3 | **already on disk** | 5 | `NSAMP=5`; marginal cost is inference only |
| Protenix | shared AF3 MSA | 5 | zero-install (`conda_envs/protenix`) |
| Chai-1 | shared AF3 MSA **and** MSA-free | 5 | `num_diffn_samples=5` is already its default; the MSA-free number is the interesting one — it prices the 1:5 option without any search cost |
| Boltz-2 | shared AF3 a3m | 5 | only method still needing an install |
| ESMFold | none | 1 | 1:1 (Option A) candidate only; deterministic → no ensemble |

Step 0 therefore writes `a0_msa_export.py`: AF3 `_data.json` → `{chain}.a3m` → `{hash}.aligned.pqt`,
once, shared by all arms. If a method rejects the shared MSA, it runs in its native mode and the
report says so rather than silently comparing unlike things.

### 2.3 Metrics — `src/masif_graph/p8/a0_bench.py`
1. **Cost**: wall-clock and CHF per chain, **MSA and inference reported separately** (MSA dominates
   and amortises across conformers — the economics that made `NSAMP=1` a Phase-7 mistake).
2. **Accuracy vs holo**: TM-score (`TMalign`, installed), CA-RMSD, and **interface RMSD** over the
   holo interface residues (reuse `align.global_align.kabsch`). Report TM-score normalised by the
   **holo** chain length — with the apo model as chain 2, normalising by the prediction would reward
   a method that simply predicts fewer residues.
3. **Calibrated spread** — the metric we are actually buying:
   * pairwise CA-RMSD among the 5 samples (mean, max);
   * **per-residue RMSF across samples**;
   * **Spearman(per-residue RMSF, holo B-factor)** and **Spearman(RMSF, pLDDT)**.
   A method whose 5 samples are near-identical scores 0 here regardless of accuracy; so does one whose
   spread is uncorrelated with real flexibility.
4. **AFDB coverage** (fork F4): map PDB chain → UniProt via the RCSB REST API
   (`data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}`), fetch
   `alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb`, and report the fraction of our **full**
   corpus (not just the 30) with (a) exact sequence match over the observed range, (b) ≥95% identity,
   (c) no match — with the residual characterised (mutants / designed constructs / tags).

### 2.4 Cost and job shape
AF3/Protenix/Boltz inference: Kuma `h100`, array over the 30 chains, ~2–5 min each → well under
1 GPU-h per method. Chai MSA-free likewise. **≈ CHF 5 total**, dominated by nothing.

## 3. A1 — is the encoder sidechain-blind?

Two tests, cheapest first. `src/masif_graph/p8/a1_sidechain.py`.

### 3.1 Primary: backbone-only ablation (no new surfaces needed)
Rebuild each eval graph with **atom nodes restricted to backbone atoms** (`is_backbone == 1`),
dropping `aa` edges to sidechain atoms and re-pointing `va` edges to surviving atoms only. The
surface (vertices, `vert_feat`, `vv`) is untouched — it was computed with sidechains present, which
is the point: we are asking *how much the encoder's output depends on sidechain atom nodes*, not
rebuilding the molecule.

Run the Phase-5 retrieval gate (`p5.retrieval_bench`, 287-clean, `--center`) on the ablated graphs.

* **If retrieval survives ≈ intact → the encoder is backbone-reading.** That is the A1 failure, and
  D8-14 is the remedy (apo-substituted training promotes from augmentation to prerequisite).
* If retrieval collapses → the encoder genuinely uses sidechains, and the P–L ceiling has another
  cause.

Cost: ~CHF 2, no new preprocessing.

### 3.2 Secondary: FASPR repack sensitivity (graded version)
For 100 complexes of the eval set, FASPR-repack each chain **in isolation** (Phase-2's apo proxy,
`scripts/repack_one.sh`), rebuild surfaces + graphs, and measure:
* per-atom `‖z_holo − z_repack‖` vs `‖z_holo − z_af3‖` — how much of the AF3 gap is sidechain-only;
* **Spearman(per-atom embedding sensitivity, flex_depth / B-factor / pLDDT)** — does the encoder
  already move more where the structure is more flexible? This is the closest thing we have to an
  implicit σ, and it directly informs D8-9/D8-19;
* retrieval on repacked structures.

Cost: 200 chains × ~90 s surfacing ≈ 5 core-h ≈ **CHF 3**.

## 4. A2 — cryptic-pocket probe

`src/masif_graph/p8/a2_pockets.py`, on the **298 P–L complexes that already have AF3 apo models**
superposed into the holo frame. No new structure generation.

For each: place the crystal ligand against the AF3 protein and measure
1. **clash count** — AF3 protein heavy atoms within 2.0 Å / 2.5 Å of a ligand heavy atom;
2. **ligand buried SASA**, holo vs apo (biotite `sasa`), and the buried-fraction ratio;
3. **contact retention** — already computed in Phase 7 (median 0.99), reported alongside as the
   complementary measure: retention says *contacts survive*, clashes say *the pocket did not close*.

"Pocket collapsed" is pre-registered as **buried-fraction ratio < 0.5 or ≥10 heavy-atom clashes at
2.0 Å**, and the *distribution* is reported, not just the fraction beyond the threshold.

Cost: ~CHF 1. Optional PocketMiner cross-check only if it runs out of the box.

## 5. A3 — rigid pose baseline, and the F3 compute number

`src/masif_graph/p8/a3_pose.py`, reusing `align.global_align` (`ransac_kabsch`, `kabsch_icp`) driven
by the existing Stage-1 atom–atom scores.

* **Conditions** (the 284 PPI eval complexes have all four states): `holo–holo`, `apo–holo`,
  `holo–apo`, `apo–apo`. The last is the deployment condition.
* **Metrics**: `fnat` (fraction of native contacts recovered) and **interface RMSD** vs native —
  most of DockQ without the dependency. Success is pre-registered as `fnat ≥ 0.3` **and**
  `iRMSD ≤ 4 Å` (standard "acceptable" docking quality).
* **Timing**: seconds per pair, which closes fork **F3**. Extrapolate to a 40k-domain screen and
  state plainly whether Stage 2 is affordable; if it is not, the Stage-1 shortlist must tighten and
  D8-18's recall bar rises accordingly.

Cost: ~CHF 2.

## 6. A4 — signal-existence probe

`src/masif_graph/p8/a4_probe.py` + a **mini-mining** step (a rehearsal for Stage B, deliberately
small).

### 6.1 Mini-mining (~300 entries)
1. Choose PDB entries from our existing PPI corpus that have **≥3 chains in the asymmetric unit**
   (our stored pairs are all `bio=1`, so extra chains are needed to get any `bio=0`).
2. Parse **`REMARK 350`** for the biological assemblies (present in our cached PDBs — no gemmi).
3. Enumerate ASU chain pairs in contact (any heavy-atom pair ≤ 5 Å). Label
   **`bio=1`** if the two chains co-occur in one `BIOMOLECULE` record, **`comp=1, bio=0`** otherwise.
4. Build surfaces + graphs for the extra chains (~600 chains × 90 s ≈ 15 core-h ≈ CHF 0.1).
5. Compute **BSA** per interface with biotite `sasa` (`SASA_A + SASA_B − SASA_AB`).

### 6.2 The probe
Features from the **existing** Stage-1 embeddings, per interface: score distribution summary
(mean / max / median of `zᵀTz` over contacting pairs), contact count, and a coherence statistic.
Fit a **logistic regression** and compare against **BSA-only**, with **grouped 5-fold CV** (grouping
by PDB entry *and* sequence cluster, per D8-15 in miniature).

Reported as a warning, not a gate: if a linear model on current embeddings cannot beat interface
area, the funnel's premise is in doubt and that is worth knowing for ~CHF 1 rather than after
Stage C.

## 7. Execution order

```
step 0  shared-MSA export (AF3 json → a3m → .aligned.pqt); try Boltz-2 / ESMFold  (timeboxed, §9)
        [TMalign and Chai are already in place — nothing left to install for them]
        ────────────── then everything below in parallel ──────────────
A0  AF3(5) / Protenix / Chai / Boltz / ESMFold on 30 chains  + AFDB coverage sweep
A1  backbone-only ablation ─→ FASPR repack (needs surfacing)
A2  cryptic-pocket probe                       (existing data only)
A3  rigid pose baseline + timing               (existing data only)
A4  mini-mining ─→ logistic probe
        ───────────────────────────────────────────────────────────────
write docs/25-phase8A-results.md  →  touch logs/PHASE8A_DONE  →  PAUSE for D8-12
```

A1's repack surfacing and A4's mini-mining are the only steps needing new preprocessing; both are
small Jed arrays and neither blocks the others.

## 8. Deliverables

* `src/masif_graph/p8/{a0_bench,a1_sidechain,a2_pockets,a3_pose,a4_probe}.py`
* `scripts/p8a_*.sbatch` job shapes; chunk lists under `logs/phase8A/`
* raw results `logs/phase8A/{a0..a4}/*.json`
* **`docs/25-phase8A-results.md`** — per-sub-stage verdicts, the D8-12 recommendation, honest
  statements of what was *not* evaluated
* `docs/progress/phase8A-log.md` — append-only build log with spend and job ids

**Total ≈ CHF 15**, no GPU training, nothing irreversible.

## 9. Autonomy boundary

**I decide alone:** all implementation detail; stratification of the A0 test set; which candidate
methods install within the timebox (**2 h per method**, then reported as "not evaluated"); thresholds
already pre-registered above; how to fix bugs; whether to add a diagnostic that costs < CHF 2.

**I stop and ask:**
* the **D8-12 decision itself** — A0 produces a recommendation, never a choice;
* anything that would spend **> CHF 25** in Stage A, or any GPU *training*;
* if A1 shows the encoder is backbone-reading — I will report it and *not* unilaterally start
  retraining Stage 1, because that reorders the whole phase;
* if a result contradicts a locked D-decision in `docs/23`.

**Guardrails throughout** (`ml-research-guardrails`): thresholds pre-registered *before* looking at
results (done, above); distributions reported, not just summary statistics; per-item spread; "the
pipeline ran" stated separately from "the result is valid"; commit + push after each working step
because `/scratch` has a 30-day cleanup.

## 10. Risks and fallbacks

| risk | fallback |
|---|---|
| Boltz-2 / ESMFold will not install in the timebox | report "not evaluated"; the benchmark still decides among AF3, Protenix and Chai, none of which need an install |
| A method rejects the shared AF3 MSA (format/depth limits) | run it in its native MSA mode and **label the row** — a method benchmarked on a different alignment is not comparable on accuracy, only on cost |
| `pairedMsa` (50k seqs) is too deep for a method's limit | subsample deterministically to that method's cap and record the depth actually used for every arm |
| Backbone-only ablation is ambiguous (partial degradation) | fall back on A1.2's graded per-atom sensitivity, which gives a continuous answer rather than a binary |
| REMARK 350 missing/malformed for some entries | skip those entries, report the fraction skipped; do not silently substitute a predictor |
| A4 mini-set too small for a stable CV estimate | report the confidence interval and treat the probe as directional only |
| AF3 5-sample inference behaves differently from `NSAMP=1` | it is the same code path with a different flag; verify on 2 chains before the array |
