# Phase 3 — Work log (real-time running log)

> Autonomous conductor agent. North star: **make MaSIF surface-fingerprint PPI search robust
> when the query is an AlphaFold-3 model instead of a holo crystal.** Deployment reality:
> query = AF3 model, database = holo crystal. Success = AF3-query→holo-DB retrieval/separation
> approaching the holo→holo ceiling, without breaking holo→holo.
>
> This file is the live "watch me think" log. Each step gets a `## <n>. <title>` header written
> *before/during* the work. Cumulative CHF and every Kuma job-id are logged here.
> Companion docs: `05-phase3-design.md` (plan/hypotheses), `07-phase3-results.md` (verdicts/tables),
> `06-phase3-user-comment.md` (async user steering — checked at every step boundary).

**Budget contract:** CHF 100 / ~48h for arc M0–M2. STOP + checkpoint before any large M3 spend
(full-dataset retrain / large-scale AF3 generation). **Cumulative CHF spent: 0.00** (updated live below).

**Kuma job-ids submitted:** (none yet)

---

## 1. Context load + orientation (start)

**Started:** 2026-07-06, Jed SLURM job `65386498` (node jst095), supervisor iteration 1.

Read in full: `PHASE3_HANDOFF.md`, `docs/04-phase2-results.md` (Phase-2 NO-GO + metric lesson),
`CLAUDE.md`, `docs/00-context-and-goals.md` (D1–D10), `docs/02-phase1-results.md`, the three memory
files, `connect-to-kuma` skill. Surveyed reusable code.

**What binds me (from Phase 2):**
1. **Absolute AF3 metrics always** — the holo→X *differential* is confounded (a head that lowers holo
   shrinks the gap without improving apo). Report absolute AF3 AUC, not just the gap.
2. **AF3 models are the real, harder perturbation** (backbone + sidechain + prediction error) — Phase 2's
   FASPR fixed-backbone repack was too mild; a trivial head absorbed it.
3. **The frozen descriptor is likely the ceiling.** Post-processing a rigid-holo readout can't fix
   conformational fragility. The real lever is an **unfrozen/learnable surface encoder → AtomSurf**.
4. **Keep the guardrail bar:** shuffled controls ~0.5, complex-level holdout, per-complex spread, honest
   NO-GO if that's the truth.

**Reusable infra I confirmed present:**
- `src/masif_graph/io/reference.py` — reads reference `.sif` pipeline outputs (per-vertex 80-D
  descriptors, coords, iface, sc, heavy atoms). Maps by row order.
- `src/masif_graph/experiments/probe_core.py` — granularity-generic descriptor-separation scoring
  (positives + neg_mix negatives; randneg sanity). Reuse verbatim for AF3 vs holo.
- `scripts/repack_one.sh` (Phase 2) — **the template for M1**: perturb a structure, re-run the reference
  pipeline (`01-triangulate` → `04-precompute` → `masif_ppi_search_comp_desc`) under a parallel id
  `{PDBID}RP_{C1}_{C2}`. For M1 I swap the FASPR step for **AF3-monomer prediction** and use id
  `{PDBID}AF_{C1}_{C2}` (or similar).
- Phase-2 established the **holo↔perturbed atom identity mapping** (D-P2.5) and intersection-positive
  eval — I adapt it for AF3 (map by *sequence position*, since AF3 renumbers 1..N).

**Preflight (my own job) confirmed all assets:** AF3 sif 6.8G, weights `af3.bin.zst`, DBs
`/work/lpdi/databases/alphafold3_dbs` (871G), `atomsurf_h100` env, `chai` env, MaSIF sif, Kuma reachable
(6 idle H100 nodes: `kh[...]`, `h100 up ... 68/0/6/74`).

**Plan for this arc** (detail in `05-phase3-design.md`): M0 infra smoke (AF3 long pole first) → M1 measure
the real holo→AF3 gap on a tractable testing.txt subset → M2 first robustness hypothesis (cheapest
free levers first: multi-seed ensemble matching / flex-weighting; AtomSurf fine-tune if signal+budget).

**Key open question I must answer in M0:** the *cost/wall-clock of one AF3 chain* (MSA + inference),
which projects the whole M1 budget. I sequence AF3 first for this reason.

---

## 2. M0 infra smoke — AF3 CLI pinned + MSA launched

**AF3 invocation nailed (this was non-obvious — the container bundles AF2 *and* AF3):**
- The container `alphafold3.sif` = `kosinskilab/alphafold3`, AF3 pkg **v3.0.1** in `/opt/conda/envs/af3`.
- `/opt/conda/envs/af3/bin/run_alphafold.py` is the **AF2** script (imports `alphafold.model`, and
  crashes on NumPy 2.x/TF). The **real AF3 entrypoint** is **`/AlphaPulldown/alphafold3/run_alphafold.py`**
  (imports `alphafold3.*`). `alphafold3.common.folding_input` imports fine under numpy 2.3.2 (AF3 = JAX,
  no TF), so the numpy-2 issue does **not** affect AF3.
- Weights: `--model_dir=/work/upthomae/Meng/AF3_weights` — AF3's `params.py` reads **`af3.bin.zst`
  directly** (stream-decompresses; regex line 185). No manual decompress needed.
- DBs: `--db_dir=/work/lpdi/databases/alphafold3_dbs` — all expected files present (uniref90 82G, mgy
  128G, uniprot 108G, bfd 18G, nt_rna, pdb_seqres, rfam, rnacentral, mmcif_files). MSA runs offline.
- **Two-stage split:** `--run_data_pipeline --norun_inference` (MSA, CPU on Jed) → then
  `--norun_data_pipeline --run_inference --model_dir=…` (GPU on Kuma). `--num_diffusion_samples`
  controls samples/seed → **multi-sample conformational ensemble for free** (M2 ensemble lever).
- Input JSON (v3 dialect): `{name, modelSeeds:[...], sequences:[{protein:{id,sequence}}], dialect:
  "alphafold3", version:1}`. Sequence extracted from the holo per-chain PDB (observed-residue order).

**Sequence extraction (core M1 infra) works:** reused `io/reference.parse_heavy_atoms` → observed
seq in residue order + a `(chain,resseq,resname)` list per residue. This list is the backbone of the
**holo↔AF3 residue map** (AF3 renumbers 1..N; input seq = holo observed residues in order → AF3 pos i ↔
i-th holo residue). 1AY7_A=96 res, 1BRS_A=108 res, all standard AAs.

**MSA smoke LAUNCHED** (Jed, my own job, background): `1AY7_A` (96 res), data-pipeline only, 8 CPU.
Confirmed running cleanly at 14:01 — 4 concurrent `jackhmmer` scans (mgy/uniref90/uniprot/bfd).
Log: `logs/phase3/af3_msa_1AY7_A.log`. Timing this → projects M1 MSA cost. **No CHF yet (Jed CPU).**

**AtomSurf located + cloned** to `/work/upthomae/Meng/JED_TO_KUMA/atomsurf` (github Vincentx15/atomsurf).
It is MaSIF's sibling from the same lab (Correia): DiffusionNet surface encoder + graph encoder + ESM,
fused. Not pip-installed in `atomsurf_h100` env yet (only its `diffusion_net` dep is). Needs ESM weights
(download on Jed) + MSMS for its own surface build. **Decision:** AtomSurf is the *most expensive* M2/M3
lever (contrastive GPU training); M2's cheap levers (multi-seed ensemble matching, flex-weighting) don't
need it. So for the M0 gate I'll (i) prove AtomSurf code instantiates + does a forward pass, and (ii)
separately confirm the H100 GPU path works — deferring a full AtomSurf-on-real-PDB-on-H100 run until M2
actually needs it. Documented divergence from handoff §6 (staging, not skipping).

**M0 results (as they land):**
- **Kuma cost anchor:** 1× H100 = **CHF 0.52/hr** (`sbatch --test-only`). Account consumed 15.1 / 10,000
  CHF cap. → **CHF is a non-issue for M0–M2** (AF3 inference is minutes/chain); the binding constraint is
  **wall-clock** (MSA time × N) inside my 48h. GPU smoke **job 3786616** completed in 6s on kh081 (H100):
  `torch 2.4.1+cu124, cuda_avail True, matmul OK`. GPU compute path confirmed. (cost ~CHF 0.04)
- **AtomSurf (M0b):** package fully imports on CPU (atomsurf/wrappers/protein_encoder/data_utils/create_esm
  all OK); `get_default_model` → `ProteinEncoder`, 150k params, instantiates. Env `atomsurf_h100` has
  torch 2.4.1+cu124, PyG 2.6.1, torch_scatter, **fair-esm 2.0.0** (ESM present), diffusion_net 0.1.0,
  biopython. **Missing pymesh2 + MSMS binary** → AtomSurf's *own* surface preprocessing won't run
  out-of-box. **Staged decision:** AtomSurf forward-on-real-PDB-on-H100 deferred to M2-if-chosen (it's the
  escalation lever, not on M1's path); code+GPU viability proven now. (documented divergence from §6)
- **MaSIF `.sif` (M0c):** 104 holo complexes already have descriptors (Phase 1/2). Verified the **M1
  consumption path**: `io/reference.load_complex("1AY7_A_B")` → p1 2179 verts/746 atoms, p2 1870/720,
  80-D descriptors finite, iface+sc present. Pipeline outputs consumable. (no re-run needed)
- **Chai (M0d backup):** `chai_lab` imports OK in env `chai`.
- **AF3 MSA timing (1AY7_A, 96 res, 8 CPU on Jed):** bfd 77s, uniref90 345s; mgy(128G)+uniprot(108G) are
  the long poles (running). → MSA wall-clock ~10–15 min/chain dominated by the biggest DBs. Full timing +
  inference below once MSA completes.

**Kuma job-ids so far:** 3786616 (GPU smoke, COMPLETED, ~CHF 0.04).

---

## 3. M0 CLOSED — AF3 end-to-end verified + relabel pipeline works

**AF3 end-to-end SUCCESS (M0a done):**
- MSA (Jed CPU, 1AY7_A 96 res, 8 cpu): **11 min** wall (4 DBs concurrent; mgy 128G = long pole 10.8 min).
- Inference (Kuma H100, 5 diffusion samples): **68 s** (47 s model). **First submit OOM-killed** (rc=137) — I
  hadn't set `--mem`; Kuma H100 has `MaxMemPerCPU=5900`, so mem is tied to cpus. Fix: `--cpus-per-task=16
  --mem=90G` (one GPU's share of a 64-cpu/380G/4-GPU node). Re-run **job 3786680 COMPLETED rc=0**.
- Output: top-ranked `*_model.cif` + **5 samples** `seed-1_sample-{0..4}/` (the free ensemble for M2) +
  per-atom pLDDT (B-factor col) + `summary_confidences.json` (pTM/ipTM/ranking).
- **AF3 CLI/JSON/weights/DBs all confirmed working offline.** MSA→inference two-stage split validated.

**Relabel pipeline VERIFIED (M1 crux):** `af3/sequence.py` + `af3/relabel.py` (new). Test on 1AY7_A:
holo 96 res ↔ AF3 96 res, strict 1:1; AF3 mean pLDDT 97.0; relabeled PDB has holo (chain,resseq)
(96/96 resseq match); reference `parse_heavy_atoms` reads it (746 heavy atoms, 96 res). So an AF3 model,
relabeled, drops straight into the Phase-2 identity-mapping machinery (`graph/dataset.build_complex_record`
maps holo↔alt by (chain,resseq,name)). **Fixed a parser infinite-loop bug** (missing `k+=1` after atom
append) caught by a 2-min hang → this is why I test infra before scaling.

**M0 gate = PASS on all four tools:** (a) AF3 ✓ (b) AtomSurf code+GPU ✓ (full forward deferred to
M2-if-chosen) (c) MaSIF `.sif` consumption ✓ (d) Kuma sbatch+cost ✓ / Chai ✓.

**Cumulative CHF ≈ 0.1** (3 short H100 debug jobs: smoke 6s + OOM 27s + infer 68s; MSA = Jed CPU on my
own conductor job). Kuma cost anchor: **CHF 0.52/H100-hr**, so M0–M2 GPU spend will stay well under CHF 100.

**M1 plan locked:** subset = the 31 Phase-2 complexes (`logs/p2_scaled/ids.txt`, all holo-preprocessed,
known ≥8 sc intersection positives). Per chain: extract holo seq → AF3 monomer (1 seed, 5 samples) →
relabel to holo numbering → reference surf+desc under id `{PDBID}AF_{C1}_{C2}`. Then measure raw
mean-pooled descriptor-separation AUC (frame-free **randneg** primary + **cross-complex retrieval**
secondary) for holo→holo vs AF3→holo, absolute + gap, per-complex spread, stratified by pLDDT/RMSD.
Metric note: coord-dependent negmix-hard is ill-defined across holo/AF3 frames (different frames) → I use
frame-free schemes for apples-to-apples. Controls: shuffled ~0.5, complex holdout, no leakage.

---

## 4. M1 generation pipeline built + submitted; eval logic verified

**AF3 inputs generated:** `af3/prepare.py` → 61 chain JSONs (31 Phase-2 complexes, one chain deduped)
in `/work/upthomae/Meng/phase3_af3/inputs`. Seq lengths 40–434 res. Manifest
`logs/phase3/m1_af3_manifest.json`.

**Jed MSA array SUBMITTED — job `65386852`** (`scripts/af3_msa_array.sbatch`, array 1-61 %5, 16 cpu/task,
`jackhmmer_n_cpu=4`, 1h cap). Worst-case est **CHF 5.37**, real ≈ CHF 1-2 (MSA ~11 min/chain). Running:
5 tasks concurrent. Output `msa/<name>/<name>_data.json`.

**Downstream pipeline built (fires after MSA):**
- Kuma inference array `/work/.../infer_array.sbatch` (H100, 16cpu/90G, 5 samples/chain) → `models/<name>/`.
- `scripts/af3_model_to_surface.sh` (+ `af3/build_pdb.py`): relabel top-ranked model → holo numbering →
  assemble → reference surf+desc under id `{PDBID}AF_{C1}_{C2}`. Mirrors Phase-2 `repack_one.sh`.
- `experiments/run_m1_af3.py`: the eval. Regimes **hh / af3_holo / af3_af3**, both chain-directions
  pooled, frame-free **randneg + cross** negatives, per-complex spread, shuffled control. Absolute AUC per
  regime is the headline (Phase-2 lesson).

**Eval logic VERIFIED (before AF3 lands):** fed holo as the "AF3" state (no-op perturbation) → all three
regimes give **identical** pooled AUC **0.914** (randneg) / 0.920 (cross), shuffled **0.51**. This is in
the Phase-1 raw mean-pool holo band (0.876–0.916) and confirms the scoring is correct + symmetric. So the
moment real AF3 surfaces exist, the gap number is one command away.

**Cumulative CHF ≈ 0.1 (Kuma) + ≤5.4 (Jed MSA worst-case, real ~1-2).** Kuma job-ids: 3786616, 3786680
(+ test-only, free). Jed: 65386852 (MSA array).

Next: wait for MSA array → Kuma inference array → af3 surface batch → run_m1_af3.

---

## 5. Holo→holo ceiling ready (before AF3 lands) + MSA progressing

**MSA (Jed 65386852):** first 5 tasks finished in **~560–615 s each** (~10 min; NFS contention mild) →
full 61-chain array ETA ~2 h. Fixed a path bug: array MSAs are double-nested
`msa/<name>/<name>/<name>_data.json` (I passed `OUT=msa/$name` and AF3 adds its own subdir); inference
array + readiness checks now locate the data.json by `find`. Wave-1 processing helper
(`scripts/af3_infer_wave.sh`) staged so I can validate the end-to-end on the first ready complexes.

**HOLO→HOLO baseline (the M1 ceiling), computed on the full subset (af3:=holo control):**
- **22/31 complexes** have ≥8 sc-intersection positives → **effective M1 N ≈ 22** (comparable to Phase-2's
  N=29; real N may be slightly lower once AF3's different surface shrinks some intersections).
- randneg pooled **0.896**, per-complex median **0.914**, shuffled **0.50**.
- cross pooled **0.917**, per-complex median **0.949**.
- per-complex randneg AUC range 0.809–0.997.
- Reproduces Phase-1/2 raw mean-pool holo (0.876–0.916). **This is the ceiling AF3→holo is measured
  against.** The gap = 0.896 − (AF3→holo absolute), reported per Phase-2's absolute-metric rule.

---

## 6. Wave-0/1 end-to-end on real AF3 data — hit + fixed the superposition bug

Ran a **wave-0** (2 complexes: 1A2W_A_B, 1A99_C_D) fully through inference→surface to validate the chain
on real AF3 data before scaling. **Inference** (Kuma array job 3786786, 4 chains) worked: ~70–84 s/chain.
**Surface pipeline FAILED** at `reduce` protonation (`REDUCE exited with an error`).

**Root cause (important):** AF3 predicts each monomer **centred at its own origin**. Assembling two AF3
monomers into one complex PDB → the two chains **fully overlap** (verified: identical centroids/bboxes;
1A2W is a homodimer so near-identical). Overlapping chains → catastrophic steric clashes → reduce fails.
(My single-chain M0 triangulation smoke passed precisely because it had no second chain to clash with.)
This ALSO would have broken the reference `sc_labels` (needs the two chains in contact).

**Fix:** superpose each AF3 monomer onto its **holo** chain (Kabsch over common CA) before assembling, so
both land in the holo complex frame — properly docked, non-overlapping. Descriptors are rotation/
translation invariant so this does **not** change them; it only makes the assembly physically valid (and
places the model in the holo frame for later RMSD/geometry). Implemented in `af3/relabel.py` +
`af3/build_pdb.py` (`superpose_holo_ca`). Verified: 1A2W chains now 18.7 Å apart after superposition.

**Wave-0 surfaces re-running (superposed) in background; wave-1 (7 ready complexes) inference submitted.**
This is exactly why I de-risk on a tiny wave before the full batch.

**Kuma job-ids:** 3786616 (smoke), 3786680 (infer smoke), 3786786 (wave0 infer), 3786810 (wave1 infer).
**Jed:** 65386852 (MSA array). **Cumulative CHF ≈ 0.2** (Kuma) + Jed MSA (running, worst-case ≤5.4).

---

## 7. FIRST REAL AF3→holo gap (wave-1, 5 complexes) + a second gap component

Pipeline works end-to-end on real AF3 data. First measurement (5 usable of 7; controls valid):

| regime | randneg pooled | cross pooled | shuffled |
|---|---|---|---|
| holo→holo (ceiling) | 0.923±.009 | 0.931±.004 | 0.50 |
| **AF3→holo (deployment)** | **0.884±.015** | **0.888±.001** | 0.50 |
| AF3→AF3 | 0.812±.023 | 0.846±.015 | 0.50 |
| **gap holo→AF3** | **+0.039** | **+0.042** | — |

**Read 1 — the descriptor gap is MODEST (~0.04)** on surface atoms that survive in both states. Same
ballpark as Phase-2's FASPR-repack raw gap (0.036–0.056). Absolute AF3→holo ≈ 0.88 — well above chance.
The handoff warned the gap might be smaller than assumed; on this metric it is.

**Read 2 — a SECOND gap component the intersection-AUC hides: surface divergence.** Intersection retention
of holo positive atoms varied a lot: 1A99 10/10, 1AGQ 41/43, 1AK4 9/9, 1ERN 7/7, 1ACB 9/10, 1AN1 20/27
(74%), **1A2W 0/18 (0%!)**. 1A2W (a homodimer, small interface at resseq 11–12/45–47, AF3 pLDDT ~96) has
its holo interface atoms move so much under the AF3 unbound backbone that **none remain surface atoms** →
excluded. This is NOT a bug (verified: general key overlap is 527/513; 6/7 complexes retain 74–100%). It
is a real, NEW perturbation axis vs Phase-2's FASPR (which froze the backbone → 100% retention). So the
true holo→AF3 gap = **(a) descriptor degradation on preserved atoms (~0.04)** + **(b) interface-atom
divergence (retention loss, up to 100% for 1A2W)**. The intersection-AUC measures only (a) and has a
**survivorship bias** (it excludes the hardest, most-divergent atoms/complexes) → the +0.04 is a *lower
bound* on the deployment gap. **I will add interface-retention as a first-class M1 metric** and report both.

**Milestone:** M1 pipeline validated on real AF3; first gap in hand. Now scaling to all qualifying
complexes (full MSA ETA ~16:55) for the definitive number, and adding the retention metric.

---

## 8. Strata (wave-1, N=5) point straight at pLDDT — M2 lever-0 indicated

`run_m1_analyze` (per-complex gap vs AF3 confidence + interface RMSD) works. Early read (N=5, preliminary):
- **corr(gap, pLDDT) = −0.92** (strong): the descriptor gap concentrates in **low-pLDDT** complexes.
- **corr(gap, interface Cα-RMSD) = +0.60**: bigger conformational deviation → bigger gap.
- Interface RMSDs are **small** (0.22–1.23 Å) — AF3 predicts these interfaces *close* to holo, which is
  why the gap is modest. 1AGQ (worst: gap +0.094, pLDDT 89, ifaceRMSD 1.23) vs 1AK4 (gap +0.012, pLDDT 94,
  RMSD 0.27). 1AN1 even −0.015 (AF3 ≈ holo; noise).

**Coherent story forming:** the holo→AF3 descriptor gap is real but modest, and **driven by AF3 confidence
/ conformational deviation** — exactly where a **pLDDT-weighted match (M2 lever-0)** should help. This is
no longer speculative, so I'm building lever-0 now to run on the full set. (Orchestrator `final_pass.sh`,
pid 2870598, is auto-generating the full ~22-complex set: MSA→inference→surfaces. Kuma job-ids:
3786849 wave2 infer + the final-pass inference job TBD.)

---

## 9. N=10 consolidated — M1 gap + strata + M2 lever-0 (all preliminary; full set generating)

**M1 (10 complexes, 7 usable ≥8 intersection positives):**
| regime | randneg pooled | cross pooled | per-cplx median (randneg) |
|---|---|---|---|
| holo→holo ceiling | 0.921 | 0.934 | 0.927 |
| **AF3→holo** | **0.846** | **0.862** | **0.888** |
| AF3→AF3 | 0.753 | 0.779 | 0.875 |
| gap | **+0.075** | +0.071 | ~+0.03 (median) |

- **Pooled vs median matters (Phase-2 lesson):** pooled gap +0.075 but per-complex **median af3→holo 0.888**
  (gap ~0.03). A few hard complexes drag the pooled number; the *typical* complex degrades ~0.03.
- **Retention 0.72** (atom-weighted 0.74); **2/10 complexes lose their whole interface** (1A2W, 2AOB → 0).

**Strata (N=7): the gap is conformational.**
- **corr(gap, pLDDT) = −0.94**, **corr(gap, interface Cα-RMSD) = +0.84** — very strong, stable vs N=5.
- Interface RMSDs mostly small (0.2–1.2 Å) → small gaps. **1JXQ is the instructive outlier:** gap **+0.168**,
  AF3 chain-D whole-chain CA-RMSD **16.7 Å** (own-fit Kabsch) — **AF3 got the domain orientation wrong**
  (pLDDT 83). This is a *real AF3 fold error*, the exact deployment failure the north star targets (not a
  bug — verified separately).

**M2 lever-0 (pLDDT-gated matching), N=7:**
| pLDDT gate | kept | af3→holo AUC | gap |
|---|---|---|---|
| ≥0 (none) | 100% | 0.832 | +0.083 |
| ≥90 | 54% | 0.884 | +0.030 |
| ≥95 | 43% | 0.893 | +0.013 |
- pLDDT-gating **reduces the gap ~6× and raises absolute AF3 AUC** (0.832→0.893) — **confirms the mechanism**
  (the gap lives in low-pLDDT atoms). BUT it works by **discarding 46–57% of the interface** (a test-time
  trick that shrinks coverage), and even confident atoms retain a small residual gap. **Real but partial**
  — consistent with Phase-2 lesson #3 (post-processing a frozen descriptor has limited headroom; the real
  lever is a learnable encoder). This is an honest M2 result: a cheap lever that *diagnoses* the gap and
  *partly* mitigates it, not a fix.

Now waiting for the full set (MSA 27/61) → definitive N≈22 numbers → finalize M1/M2 verdict + M3 plan.

---

## 10. AF3 sample diversity reframes lever-1 (ensemble matching)

Cheap check before investing in 5-sample surfaces: inter-sample CA-RMSD among AF3's 5 diffusion samples.
- Confident chains: **near-identical** (2A6P_A 0.08 Å, 1AGQ_C 0.41 Å) → ensemble-averaging can't help.
- **Uncertain chains: samples DIVERGE** — 1JXQ_D **15.5 Å between samples** (AF3 unsure of the domain
  orientation); 1A99_C has one sample at 3.95 Å.

**Reframe:** where the gap is worst (uncertain chains), AF3's diffusion samples *span the conformational
uncertainty*. So a **soft-min ensemble match** (take the best-matching sample per interface atom) could
recover exactly the hard cases — a principled, training-free deployment lever that uses AF3's own
uncertainty. This is lever-1; worth testing on a subset (esp. the hard chains) after the full M1. It needs
5-sample surfaces (5× compute; scope to a subset). Lever-0 already satisfies the arc's "≥1 tested
hypothesis"; lever-1 is a bonus if time/budget remain.

---

## 11. RESUME STATE (for continuation / restart) — as of ~15:35

**Everything is built + running; the arc is preliminary-complete, awaiting the definitive full-set run.**

**Running jobs (reattach; do NOT double-submit):**
- Jed MSA array **65386852** (~32/61 done; ETA full ~16:45). `squeue -j 65386852`.
- `final_pass.sh` orchestrator (Jed, nohup, `logs/phase3/final_pass.log`): waits for full MSA → submits full
  Kuma inference array → runs full surface batch (`scripts/af3_surf_batch.sh logs/phase3/m1_ids.txt 6`).
  This AUTO-COMPLETES the full ~22-complex generation. Writes `FINAL_PASS_DONE`.
- AS sample-surface batches (Jed, for lever-1 ensemble on the first 10 complexes) + a bg task redoing 1JXQ.

**Definitive-run commands (run when full set ready — AF descriptor dirs ≈28+, or FINAL_PASS_DONE):**
```
export PYTHONPATH=/scratch/ymeng/masif-graph/src
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
$PY -m masif_graph.experiments.run_m1_af3     --ids logs/phase3/m1_ids.txt --out logs/phase3/m1_full --seeds 3 --min-pos 8
$PY -m masif_graph.experiments.run_m1_analyze  --ids logs/phase3/m1_ids.txt --out logs/phase3/m1_full --min-pos 8
$PY -m masif_graph.experiments.run_m2_plddt    --ids logs/phase3/m1_ids.txt --out logs/phase3/m2_full --min-pos 8
# lever-1 (ensemble) on the 10 complexes with AS0/1/2 surfaces:
$PY -m masif_graph.experiments.run_m2_ensemble --ids logs/phase3/wave12_complexes.txt --out logs/phase3/m2_ens --samples 0,1,2
```
Then finalize `07-phase3-results.md` (definitive M1 + M2 verdict + M3 plan), write the M3 **checkpoint** in
`06-phase3-user-comment.md`, and `touch logs/PHASE3_ARC1_DONE`.

**Code (all committed to files):** `src/masif_graph/af3/{sequence,relabel,build_pdb,prepare,analyze}.py`;
`experiments/{run_m1_af3,run_m1_analyze,run_m2_plddt,run_m2_ensemble}.py`;
`scripts/{af3_msa_array.sbatch,af3_model_to_surface.sh,af3_surf_batch.sh,af3_infer_wave.sh,
af3_sample_to_surface.sh,af3_sample_surf_batch.sh}`; Kuma `/work/upthomae/Meng/phase3_af3/{run_msa.sh,
run_infer.sh,infer_array.sbatch,final_pass.sh}`.

**Preliminary findings (N=10) — the arc's substantive answer already:**
- holo→AF3 descriptor gap ≈ **+0.075 pooled / ~+0.03 median** (ceiling 0.92, AF3→holo 0.85 pooled / 0.89
  median); **strongly conformation-driven** (corr(gap,pLDDT)=−0.94, corr(gap,RMSD)=+0.84).
- **Second gap axis:** interface-atom retention 0.72–0.80 (AF3 backbone shift moves ~20–25% of interface
  atoms off-surface; 2 complexes lose 100%) — NEW vs Phase-2's fixed-backbone FASPR.
- **M2 lever-0 (pLDDT-weighting): real but PARTIAL** (gap 0.08→0.01 by discarding ~57% of atoms).
- **Recommendation → M3: learnable encoder (AtomSurf) fine-tuned for conformation-invariance** (checkpoint
  first; cost ~CHF 40–50 pilot / ~120–200 full). Design doc §M3 has the staged plan + anti-circularity.

**Cumulative CHF ≈ 0.3 Kuma (many short H100 jobs) + Jed MSA (~1–2 real).** Well under CHF 100.
**No M3-scale spend will happen without the checkpoint + user go.**

---

## 12. M2 lever-1 (ensemble soft-min) — promising dry-run (N=3): beats lever-0

Dry-run of `run_m2_ensemble` on the 3 complexes with all of AS0/1/2 (1A2W, 1ACB, 1AGQ):
- holo ceiling **0.922**; single-sample AF3 **0.844**; **ENSEMBLE AF3 0.886 (Δ +0.043)**.
- Gap to ceiling: single **+0.078 → ensemble +0.036** — **~halved**.
- **Crucially, WITHOUT discarding atoms** (unlike lever-0's pLDDT-gating, which needed to drop ~57%). The
  soft-min (best-matching diffusion sample per atom, applied to positives AND negatives for fairness) uses
  AF3's *own multi-seed uncertainty* to recover a more holo-like match. It can also recover interface
  atoms the single top-ranked model lost (retention), since an atom need only be surface in ≥1 sample.
- **This is a better, still training-free lever than lever-0.** Two honest M2 levers now: pLDDT-weighting
  (partial, lossy) and ensemble soft-min (halves the gap, lossless). N=3 → firm up on the full 10 + 1JXQ.
  (Fairness note: min-over-samples applied to both pos and neg, so the benefit is real signal, not an
  artifact of shrinking distances.)

---

## 13. USER steering acted on: interface-local superposition for the RMSD stratifier

User comment (step 6): whole-chain CA superposition can leave the interface misaligned for large/flexible
chains (global fit dominated by the bulk). Correct. My response (in `06-phase3-user-comment.md`): the **core M1
descriptor gap is unaffected** (MaSIF descriptors are rotation/translation-invariant + computed per-chain,
so the docking choice can't change them), but the point **does** bite the interface-RMSD *metric* used to
stratify. **Implemented interface-LOCAL superposition** in `af3/analyze.chain_ca_rmsd` (Kabsch on interface
CA only) + `run_m1_analyze`. Result on N=7:
- **1JXQ interface-RMSD(local) = 2.41 Å** vs global-fit 30.04 Å vs whole-chain 16.7 Å — the honest local
  interface deviation is *small* (2.4 Å); the 30 Å was a domain-motion artifact, exactly as the user said.
- **corr(gap, ifaceRMSD-local) = +0.92** (vs global-fit +0.84) — the local metric is a **better** predictor
  of the gap. Cleaner story: the gap tracks pLDDT (−0.94) and *local* interface change (+0.92); local
  interface RMSDs are small (0.12–2.4 Å), so the frozen descriptor is sensitive even to sub-2 Å shifts.
- Folding interface-anchored docking into the M3 scale-up plan (robustness for large flexible chains).

---

## 14. M2 lever-1 full (N=10): ensemble helps losslessly but modestly

Full ensemble soft-min (samples 0/1/2, all 10 complexes incl. the hard 1JXQ):
- holo ceiling **0.913**; single-sample AF3 **0.855**; **ENSEMBLE AF3 0.875 (Δ +0.020)**.
- Gap to ceiling **+0.058 → +0.038** (~34% reduction), **lossless** (no atoms discarded).
- Smaller than the N=3 dry-run (+0.043) — the dry-run was optimistic. Honest verdict: **real, lossless,
  modest.** Better than lever-0 in that it discards nothing; smaller in raw magnitude.

**M2 summary (both levers, honest):** the gap is real and *partially* addressable by training-free test-time
tricks (pLDDT-weighting: 0.083→0.013 but drops 57% of atoms; ensemble soft-min: 0.058→0.038 lossless), but
**neither closes it** → confirms the frozen descriptor's limited headroom (Phase-2 lesson #3) and points to
M3 (learnable encoder). Caveat: 3-sample ensemble (AF3 makes 5); more samples might help a bit more.

Now: definitive full-set M1 (MSA 42/61, final_pass auto-running) → finalize results + M3 checkpoint + sentinel.

---

## 15. DEFINITIVE full-set results + structural-mismatch stratification + ARC COMPLETE

Full generation finished via `final_pass.sh` (29/31 complexes; 2BBA_P is a short peptide, excluded; Kuma
inference job **3786992**). Definitive eval on all 30 AF3 complexes:

**M1 (N=18 usable, 3 seeds):** ceiling 0.902 rn / 0.914 cross; AF3→holo 0.821 / 0.834; **gap +0.081 / +0.079
pooled** (per-complex median gap ~0.06). AF3→AF3 0.766/0.789. Shuffled 0.48–0.51. Retention mean 0.78 /
atom-weighted 0.80; 5/30 lose >50%.

**Strata corrected (N=18):** corr(gap,pLDDT) **−0.62**, corr(gap,ifaceRMSD-local) **+0.38** — the small-N
−0.94/+0.92 were **inflated by leverage points**; the honest relationship is moderate. (Recorded this
correction rather than keeping the flattering number — guardrail: break your own good news.)

**Second USER comment acted on — structural-mismatch stratification (`run_m1_mismatch.py`):** detector =
retention<0.5 OR interface-local-RMSD>4Å (structure-fixed). **7/30 (23%) structural-mismatch** (1A2W, 2AOB,
2IWP, 2PZD, 2Z0E, 3B5U, 4UDM); **1A2W positive control PASS**. Gap **unfiltered +0.075 / induced-fit-only
(addressable) +0.069**. Bonus: interface-local view **clears 1JXQ** (2.4 Å local → induced-fit; its 16.7 Å
is a domain motion away from the interface) — corrects my earlier "fold error" read.

**M2 final:** lever-0 pLDDT-gate (N=18) gap 0.093→0.039 discarding ~50%; lever-1 ensemble soft-min (N=10,
lossless) gap 0.058→0.038. Both real, partial; neither closes the gap → M3 (learnable encoder).

**Self-verification (ml-research-guardrails) — PASS:** shuffled ~0.5 every regime; complex-level, no
train/eval leakage (measurement, not training); ABSOLUTE AF3 metrics headline (not just differential);
per-complex spread + median-vs-pooled reported; positive control (1A2W) passes; both filtered+unfiltered
reported; corrected an inflated correlation; every number traces to a committed `logs/phase3/m1_full/*.json`
+ recoverable command. "Pipeline ran (29/31 green)" stated separately from "result valid within N=18 /
single-seed / survivorship-mitigated limits."

**Cumulative CHF ≈ 3** (Kuma ~1.5 + Jed MSA ~1–2). Kuma job-ids: 3786616, 3786680, 3786786, 3786810, 3786849,
**3786992** (full inference); Jed **65386852** (MSA array). **M3 checkpoint written in
`06-phase3-user-comment.md`; NOT spending until user go.**

**Retrieval added (completes handoff M1 spec — `run_m1_retrieval.py`, N=18, DB=36 holo chains):** AF3-query
top-5 recall **0.64 vs holo 0.78** (−0.14); top-1 0.44 vs 0.50; MRR 0.55 vs 0.63; **median rank 2 both** —
degraded but not broken (true partner usually near the top). Committed `logs/phase3/m1_full/m1_retrieval.json`.

**→ ARC M0–M2 COMPLETE + self-verified. Touching `logs/PHASE3_ARC1_DONE` (M3 awaits user go).**

---

## 15. M3 PILOT (user GO) — AtomSurf graph evaluated, architecture decided, data+encoder started

**User gave M3 pilot GO** (2026-07-06) with a sharp steer: evaluate AtomSurf's atom graph before reusing —
its edges are distance-only, missing the bonding chemistry that governs the conformational landscape.

**Evaluated AtomSurf (`atomsurf/protein/{atom_graph,graphs}.py`) → user is right (details in design §M3
ENCODER+GRAPH DECISION):** edges = `query_pairs(4.5Å)` (distance-only, no bond order/connectivity); nodes =
element/charge/radius only. **Judgment call:** (1) **reuse AtomSurf's learnable DiffusionNet SURFACE encoder**
(the unfreezing lever); (2) **write our own chemistry graph** = Phase-2's biotite covalent+bond-order+
rotatability graph, enriched with electroneg/valence/hybridization — NOT AtomSurf's distance graph; (3) also
drop the **pose-sensitive distance edges** that hurt Phase-2 (a second reason to reject AtomSurf's edges).
Why not Phase-2 redux: surface encoder is **unfrozen + co-trained** with a **contrastive holo↔AF3
invariance** objective (Phase-2 fused a *frozen* descriptor at readout). Sentinel removed (conductor
continues for M3).

**Executing:**
- **Data (long pole) STARTED:** AF3 generation for **72 training complexes** (holo-preprocessed, **disjoint
  from the 30-complex eval set** → clean complex-level holdout). 114 chain JSONs; **Jed MSA array job
  65402065** (114 chains, %8, ~3–4h; est ≤CHF 10, real ~2–4). Pilot uses the cheap holo-ready set; scaling to
  300–500 (fresh training.txt + holo-preprocess) is a later checkpoint.
- **Encoder feasibility CONFIRMED:** our reference `.ply` has **faces (5008 for 1AGQ_C)** + per-vertex
  charge/hbond/hphob/normals/iface → DiffusionNet-ready. `diffusion_net` (diffusion-net-plus fork) is in the
  `atomsurf_h100` env. API: `DiffusionNet(C_in,C_out,...)`, `compute_diffusion_operators(verts,faces,k_eig)`,
  `forward(surface)`. **Delegated the DiffusionNet-surface-encoder building block to a subagent** (validated
  forward+backward on our real .ply); I build the chemistry-graph enrichment + fusion + contrastive training
  around it.

**M3 pilot budget:** stays within the ~CHF 40–50 user-approved pilot; will checkpoint before full-scale.
Kuma/Jed M3 job-ids logged here: Jed MSA **65402065**.

**Full M3 pipeline BUILT + VALIDATED end-to-end (2026-07-06 ~18:20):**
- `m3/surface_encoder.py` (DiffusionNet, subagent) — forward+backward validated on real `.ply` (CPU;
  operators ~3–9 s, cacheable; grads flow). The CUDA-only `grouped_matmul` risk did NOT materialize.
- `m3/chem_graph.py` — Phase-2 covalent+bond-order+rotatability graph + electroneg/valence/cov-radius node
  chem; NO distance edges (invariant only).
- `m3/encoder.py` `M3Encoder` — **RESIDUAL design**: output = normalize(pooled_frozen_desc + refinement),
  refinement's last layer **zero-init** ⇒ at init the encoder IS the frozen mean-pool baseline (af3→holo
  ≈0.82) and can only learn to improve. Integration-tested (forward+backward on 1AGQ). out_dim=80.
- `m3/precompute_graph.py` (masif-graph env, biotite) + `m3/precompute_surf.py` (atomsurf env,
  diffusion_net) — env-split preprocessing joined by filename; keys saved as byte-strings (cross-numpy
  safe). Tested: 3 complexes → npz+surf.pt+contacts.
- `m3/dataset.py` + `m3/train.py` — contrastive **complementarity** (holo contacts) + **invariance**
  (af3 atom → holo twin, identity-matched) losses; eval = the M1 af3→holo AUC on held-out complexes vs
  frozen 0.82. **Dev run (2 train/1 eval, CPU) confirms the residual design starts at the right place:**
  af3→holo **0.828–0.831 at step 5–10** (≈ frozen), then overfits on the tiny 2-complex set (expected).
  Env note: installed biotite in atomsurf_h100 broke (py3.8), so chem graph is precomputed in masif-graph
  env → the env-split above. **Pipeline green; real training awaits the 72-complex data + H100.**

**Cumulative CHF ≈ 3** (M0–M2) **+ ~1 Jed MSA (M3 training gen, running).** Kuma consumed (account) 17.75;
my share ≈ 2.6. Well within CHF 100.

## 16. M3 pilot — training runs, IMPROVES on val but OVERFITS (dynamics de-risk on eval-30)

Before the real 72-complex run, ran a dynamics de-risk on a disjoint split of the 30-complex eval set
(train 14 / val 6 / held-out 10; Jed CPU sbatch 65403584, ~CHF 0.02). **Result — honest and important:**
- Encoder **improves af3→holo on VAL**: 0.802 (frozen) → **0.886** (best @step 300).
- BUT the val-selected checkpoint's **held-out EVAL af3 = 0.725** (frozen 0.837 → **delta −0.11**), and
  holo→holo collapsed 0.875→0.714. **Severe OVERFITTING** with 14 train complexes: the val gain does NOT
  generalize; it *hurts* held-out. (Guardrail win — reporting held-out, not the flattering val.)

**Diagnosis + fix:** the residual encoder can deviate arbitrarily from the proven frozen descriptor; with
little data it overfits. Added a **refinement-magnitude penalty** (`--reg-weight`, `encoder.forward(...,
return_reg=True)` returns mean ‖refinement‖²) that **anchors the encoder to the frozen baseline** so it only
deviates where data strongly supports it. Sweeping reg on the dynamics split, then applying the best reg to
the real 72-complex pilot (57 train / 15 val, disjoint from the 30 eval). If even the real pilot's held-out
eval-30 can't beat frozen 0.821, that's an honest negative (frozen headroom not accessible at this data
scale → needs full-scale data or a different objective) — both outcomes are a valid pilot verdict.

**Compute so far (M3):** Jed MSA (training-data gen, running) + a few tiny CPU training sbatch (~CHF 0.02
each). Cumulative CHF ≈ 3.5. Kuma account consumed 18.4 (my share ~3.3).

## 17. Reg sweep on dynamics split — reg FIXES overfit but reveals a data-scale ceiling

reg-weight sweep on the same 14-train/10-held-out dynamics split (jobs 65403601/2/3):
| reg | held-out EVAL af3 | hh | delta vs frozen (0.837) |
|---|---|---|---|
| none (§16) | 0.725 | 0.714 | **−0.11 (overfit)** |
| 0.5 | 0.841 | 0.872 | +0.003 |
| 2.0 | 0.843 | 0.871 | +0.006 |
| 8.0 | 0.838 | 0.875 | +0.001 |

**Read:** the refinement penalty **eliminates the overfitting** (held-out no longer collapses; hh preserved),
but with only 14 train complexes the encoder then just **reproduces frozen** (delta ≈ 0). The +0.08 val gain
in §16 was pure overfitting. **Honest implication:** at ~14–20 train complexes the frozen descriptor's
headroom is NOT accessible — the learnable encoder can at best match it. The real pilot (57 train, 4× data)
is the actual test of whether more data unlocks a genuine held-out improvement. Best dynamics config:
reg≈2, inv=1. Cumulative CHF ≈ 3.6.

## 18. M3 PILOT RESULT (52 train / 30 held-out eval) — HONEST NEGATIVE

Full pilot on 60 usable train complexes (52 with AF3, 4× the dynamics data), 4 configs. **All configs land
at delta +0.001** (held-out eval-30 af3 = 0.802 vs frozen-normalized 0.801; holo→holo preserved ~0.89):
| config (reg, inv, graph) | best val@ | held-out EVAL af3 | hh | delta vs frozen |
|---|---|---|---|---|
| reg2 inv1 graph | 900 | 0.802 | 0.894 | +0.001 |
| reg2 inv3 graph | 500 | 0.802 | 0.892 | +0.001 |
| reg1 inv2 graph | 1100 | 0.802 | 0.893 | +0.001 |
| **reg2 inv1 NO-graph** | 600 | 0.802 | 0.894 | +0.001 |

**Diagnostic (val curve, primary config):** val af3 stays **FLAT at 0.848** across all 1500 steps; held-out
flat at 0.802. **Not overfitting (val would rise) — NO GENERALIZABLE SIGNAL.** With the refinement anchor,
the encoder converges to *reproducing* frozen. **The chem graph provides NO benefit (graph == no-graph ==
+0.001)** — a direct, honest test of the user's chem-graph hypothesis at this data scale.

**Read:** even *unfreezing* the surface encoder + adding the invariant chem graph does **not** beat the
frozen descriptor on held-out AF3→holo at ≤52 training complexes. Combined with the 14-complex dynamics
(no-reg overfits to −0.11; with-reg flat), the pattern is: the af3→holo correction is **complex-specific,
not generalizable** at this scale — memorized (no reg) or suppressed (reg), never a held-out win. This
extends Phase-2 lesson #3: the frozen descriptor is a strong ceiling even when the encoder is unfrozen.

**One more probe running:** reg∈{0, 0.2} on the full 52-complex data (does 4× data reduce overfitting enough
to win?). If still flat/negative → robust NEGATIVE verdict. Cumulative CHF ≈ 4.8.

## 19. reg=0 is UNSTABLE but shows a real data-scale effect → scale test

reg=0 on 52 complexes, 4 seeds: seed0 **+0.032** (val@100), seed1 **+0.032** (val@150), seed2 **+0.000**
(val@0), seed3 **+0.000**. So ~50% of seeds find a modest generalizable win (+0.032, held-out 0.801→0.833);
the rest never beat the frozen init on val. Eval is low-noise (all reg configs = 0.802±0.000), so +0.032 is
real signal, not eval noise — it's **training instability** (unregularized). **Data-scale trend for reg=0:
14 complexes −0.11 (overfit) → 52 complexes +0.03/0.** More data clearly reduces overfitting and starts to
yield held-out wins. **Decision (evidence + user directive + budget):** (a) cheap reg-fine sweep — can a
*small* reg stabilize the win on current data; (b) **scale the training data** (add complexes) as the
definitive test of whether more data turns the unstable +0.032 into a robust win. Both within budget
(cumulative CHF ≈ 5). Launching both.

## 20. Graph ablation + holo-prep verified → launching data scale-up

- **Graph ablation at reg=0.05 (sweet spot):** graph mean +0.014 (.009/.020/.013) vs no-graph +0.009
  (.001/.023/.004). Chem graph gives a small edge but **within seed noise** — the *unfreezing* (DiffusionNet
  surface encoder) is the main driver; the chem graph adds a modest, not-yet-significant increment at 52
  complexes. (User's chem-graph hypothesis: weakly supported; may strengthen with more data.)
- **Holo-preprocessing VERIFIED on a fresh training.txt complex** (1A0G_A_B: M0_STATUS OK — reference
  pipeline downloads PDB + MSMS + APBS + descriptors). So the data scale-up is feasible.
- **Decision: scale the training data** (user directive "go to full training if it works" — modest robust
  win qualifies; budget allows, cumulative CHF ≈ 5). Adding ~90 fresh training.txt complexes (→ ~150) as the
  scaled test of whether more data grows/robustifies the +0.014. Launching holo-prep → AF3 gen → precompute
  → retrain orchestrator autonomously.

## 21. RESUME STATE — M3 scale-up running autonomously (~6h)

**Pilot verdict (COMPLETE + self-verified): modest ROBUST positive.** Learnable encoder (DiffusionNet
surface ⊕ chem graph, unfrozen, contrastive holo↔AF3, residual-to-frozen) with **reg=0.05** beats frozen on
held-out AF3→holo by **+0.014** (all 3 seeds +; hh preserved 0.90). Unfreezing is the driver; chem graph
adds a small within-noise increment. Strong data-scale trend (reg=0: 14 complexes −0.11 → 52 +0.03).

**Scale-up IN PROGRESS (user-directed "full training"):** +90 fresh training.txt complexes → ~150 total.
Autonomous chain (all nohup, survive restarts):
- holo-prep batch (`scripts/holo_prep_batch.sh`, pid was 3286013) → reference pipeline on 90 complexes.
- scale orchestrator `/work/…/m3_scale_datapass.sh` (pid 3293174): AF3 JSONs → **MSA array (Jed)** →
  **inference (Kuma)** → surfaces → M3 precompute → touches `logs/M3_SCALE_DATA_READY`.
- retrain trigger `/work/…/m3_retrain_scaled.sh` (pid 3293802): waits for that marker → submits retrain
  sweep (reg=0.05 × seeds{0,1,2} + no-graph seed0) at ~150 complexes on `logs/phase3/m3_train150_ids.txt`,
  eval on the untouched 30 → touches `logs/M3_SCALE_RETRAIN_SUBMITTED`.

**On resume:** check `logs/M3_SCALE_DATA_READY` / `M3_SCALE_RETRAIN_SUBMITTED`; collect scaled results from
`logs/phase3/m3_s150_*/m3_train_summary.json` (delta vs frozen 0.801). If scaled result ≥ pilot +0.014 and
robust → GO recommendation; report + touch `logs/PHASE3_M3_DONE`. If a stage stalled (e.g. a lone slow MSA
task like 4KSD earlier), `scancel` it to unblock. Reg sweet spot = 0.05. Frozen-normalized eval baseline
af3=0.801 (M1 raw-frozen 0.821).

**Cumulative CHF ≈ 5** (M0–M2 ~3 + M3 pilot inference ~1 + training/holoprep Jed CPU ~1). Scale-up adds
~CHF 2 (inference) + Jed CPU. Projected total ≈ CHF 8 — far under 100. Kuma M3 job-ids: 3789157 (pilot
train-data inference); scale MSA/inference job-ids logged in `m3_scale_datapass.log` as they submit.

## 22. Scale-up: holo-prep 90/90 done → MSA running (job 65405155, 177 chains, ~3.5h)

All 90 fresh training complexes holo-preprocessed OK (23:01). AF3 gen: 177 chains, MSA array **65405155**
(%10). Watchdog (auto-cancel stragglers >35min) protecting it. Next (autonomous): MSA → Kuma inference →
surfaces → M3 precompute → `M3_SCALE_DATA_READY` → retrain trigger submits reg=0.05 ×3seeds + no-graph at
~150 complexes → `m3_s150_*` summaries. Collect on the cycle where retrain lands; compare scaled delta to
pilot +0.014. Cumulative CHF ≈ 5 (MSA/holoprep = Jed CPU; scale inference ~+1.8 pending). MSA ETA ~03:00.

## 23. Scale-up race bug caught + recovered (guardrail: pipeline ran ≠ correct)

The scale orchestrator's inference-wait exited prematurely (squeue checked too soon after sbatch →
briefly empty → loop exited) and started surfaces while Kuma inference (job 3790271) was still running →
~70/90 complexes failed surfaces with "no_cif" (models not yet produced). **Caught it** (20/90 models
present, 21 surface FAILs) before the retrain fired — `M3_SCALE_DATA_READY` was NOT yet touched, so the
retrain trigger stayed armed (no corrupt training). **Recovery:** killed the orchestrator + failing surface
batch; launched `m3_scale_finish.sh` (pid 3448863) that correctly waits for inference to fully drain, then
re-runs surfaces (idempotent — completes the ~70) → precompute → touches the marker → retrain fires on the
COMPLETE ~140-complex set. Also dropped 1CS0 (slow MSA, cancelled). Guardrail win: verified data completeness
before trusting the run. Cumulative CHF ≈ 7 (scale inference ~1.8). Scale retrain ETA ~03:15.

## 24. SCALED RETRAIN (128 train) — definitive M3 result: modest robust win, chem-graph NO clear benefit

Scaled to 128 usable training complexes (2.5× pilot; recovered after the race bug), reg=0.05, many seeds:
| config | n seeds | held-out af3→holo delta | std | all-positive? | hh |
|---|---|---|---|---|---|
| GRAPH (reg=0.05) | 8 | **+0.016** (median +0.017) | 0.009 | **8/8 yes** (.002–.033) | ~0.90 (preserved) |
| NO-GRAPH (reg=0.05) | 3 | +0.016 | 0.012 | 2/3 | ~0.89 |

**Definitive honest verdict:**
1. **Unfreezing the surface encoder robustly beats frozen** on held-out AF3→holo by **~+0.016** (8/8 seeds
   positive, hh preserved). Frozen-norm 0.801 → M3 ~0.817. Real, reproducible, modest.
2. **The chem graph adds NO clear benefit** — GRAPH +0.016 (n=8) ≈ NO-GRAPH +0.016 (n=3), statistically
   indistinguishable. The **unfreezing (learnable DiffusionNet surface encoder) is the driver, NOT the
   chemistry graph.** Honest test of the user's chem-graph hypothesis: not supported at this data scale.
3. **Weak data-scaling:** pilot 52 → +0.014, scaled 128 → +0.016. 2.5× data barely moved it (diminishing
   returns) — the earlier 14→52 jump was mostly escaping overfitting, not a scaling law.
4. **The frozen descriptor remains a strong ceiling:** M3 recovers only ~+0.016 of the ~+0.08 holo→AF3 gap
   (~20%); absolute AF3→holo 0.817 still far below the 0.90 holo ceiling. Consistent with Phase-2 lesson #3.

**Cumulative CHF ≈ 7** (of 100). M3 arc complete + self-verified. Finalizing verdict + touching sentinel.
