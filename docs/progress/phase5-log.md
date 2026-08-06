# Phase 5 — running log (watch me think)

Append-only. This is my memory across restarts. Design = `docs/13-phase5-design.md`.
Steering channel = `docs/14-phase5-user-comment.md` (re-read at every step boundary).
Autonomy: headless-style, no human in loop; budget **CHF 100** (compute) for the whole phase;
decide + document + keep going until the Phase-5 gate is met or options exhausted.

---

## RESUME STATE (keep current — read this first on any restart)
- **Phase:** M0 (build reusable machinery + de-risk; all CPU, ~free).
- **Running job-ids:** none yet.
- **Cumulative spend:** CHF 0.
- **Next concrete action:** M0c — install mmseqs2, build sequence-cluster-clean split.
- **Do-not-double-submit markers:** (none yet).

---

## 0. Recon (M0a) — assets found, gaps to fill, plan

**Reusable assets (verified this session):**
- **Surface+descriptor gen (.sif):** `/scratch/ymeng/Neosurf_Neosurf/masif-neosurf_v0.1.sif`
  (+ `/work/upthomae/Meng/MaSIF/IMAGE/...`, `/work/.../MaSIF-Neosurf-af2/...`). Driven by
  `scripts/phase4_preproc.sbatch` → `scripts/holo_prep_batch.sh` (data_prepare_one +
  compute_descriptors) → descriptors under
  `masif-neosurf-af2/masif/data/masif_ppi_search/descriptors/sc05/all_feat/<id>/`. CPU-only, ~free, resumable.
- **AF3 gen:** `alphafold3.sif` at `/work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif`;
  weights `/work/upthomae/Meng/AF3_weights/af3.bin.zst`; scripts `scripts/af3_{msa_array.sbatch,
  infer_wave.sh,model_to_surface.sh,sample_to_surface.sh}`; envs `protenix`, `chai`. Phase-3 ran ~31.
- **Holo npz store:** `/work/upthomae/Meng/phase4/stageA_full_npz` — **4872 complexes** with full
  hetero graph + frozen 80-D descriptors + contacts. Coverage vs the MaSIF-search split:
  **train 4812/4943, test only 60/959** → must preprocess ~899 missing test complexes (free CPU).
- **NPZ schema** (per `{id}__{holo|af3}__p{1,2}.npz`): atom_feat(N,14), aa_edge/aa_order/aa_rot,
  vert_feat(V,4), vv_edge/vv_dist/vv_cos, va_v/va_a/va_dist/va_cos, surf_node_idx, desc_straight(S,80),
  desc_flipped(S,80), coord(S,3), keys. Plus `{id}__contacts.npz`: `pos`(dense), `pos_sc`(sc-gated).
- **p4 code (reuse, don't rebuild):** `src/masif_graph/p4/{dataset,encoder,eval_af3,retrieval_af3,
  precompute,objective,train_retrieval}.py`.
- **Learned encoder (D-P5.1 default):** `/work/upthomae/Meng/phase4/ret_full_ctr_best.pt`
  (centered full-set; MUST eval with DC-offset centering, `--center`).
- **Env python:** `/work/upthomae/Meng/conda_envs/masif-graph/bin/python`.
- **m2_npz:** `/work/upthomae/Meng/phase4/m2_npz` — ~51 complexes with holo+af3 graphs (Phase-4 M2 set).

**Gaps to fill:**
1. `mmseqs2` missing → install on Jed (internet) for the ~30%-id cluster split.
2. Holo graphs for ~899 missing **test** complexes → run `phase4_preproc.sbatch` (free CPU) + `p4.precompute`.
3. AF3 apo models + graphs for the cluster-clean test subset → Kuma/local AF3 (budgeted, M1).

**Refined plan (cheapest-first, each gated):**
- **M0b** living docs (this file + `14-phase5-user-comment.md` + `PHASE5_HANDOFF.md`).
- **M0c** install mmseqs2; extract chain sequences for all 5902 complexes; cluster @30% id; derive
  the cluster-clean test split (test clusters disjoint from train). Report surviving eval count.
- **M0d** preproc holo for a ~50-cplx cluster-clean test subset (free); build npz; stand up the
  retrieval harness (extend `retrieval_af3.py`) and **reproduce MaSIF-search HH (frozen)** = sanity gate.
- **M0e** run AF3 on 1–2 test complexes; measure per-model wall-clock/cost → M1 estimate.
- **M1** AF3 apo at cluster-clean test scale (budgeted; log cost).
- **M2** THE GATE: 4-cell query×DB (HH/AH/HA/AA) retrieval, learned vs frozen, robustness Δ + controls.
- **M3** atom-graph ablation on AF3 cells + write-up + verdict.

**Guardrails (skill dir empty, ethos from CLAUDE.md/slurm skill):** shuffled-label ≈0.5 control every
eval; frozen MaSIF on identical pairs = ceiling; per-complex spread not just pooled; sequence-cluster
holdout (no homolog leak); "pipeline ran" ≠ "result valid"; try to break good news before believing it;
mandatory `--center` for the learned encoder (else collapse to chance, doc 10 §24).

## M0c. Sequence-cluster-clean split — DONE
- Sequences: RCSB entry FASTA for 5618 unique PDBs (batched/cached `tools/rcsb_fasta/`) →
  `logs/phase5/all_chains.fasta` = 12,211 chain records (192 missing = nucleic/ligand, ~1.5%).
- mmseqs2 easy-cluster `--min-seq-id 0.3 -c 0.5 --cov-mode 1` → 4205 clusters
  (`logs/phase5/mmseqs/clu_cluster.tsv`).
- Split logic: test complex is **clean** iff every chain-cluster is absent from ALL train complexes;
  then within-test dedup (greedy, no cluster reused).
- **RESULT: 959 test → 353 train-clean → 304 deduped** = the cluster-clean eval set
  (`logs/phase5/eval_clusterclean.txt`). **62% of nominal test are train homologs** — the leak the
  design flagged is real and large; honest denominator is 304, not 959.
- Holo npz already present for only 16/304 → preprocess ~288 (free CPU .sif) + AF3 for the 304.

## RESUME STATE (updated)
- Phase: M0d — inspect preproc pipeline, submit holo preproc for 304 clean-test (free CPU), build harness.
- Running jobs: none. Spend: CHF 0.
- Key files: eval set `logs/phase5/eval_clusterclean.txt` (304); harness TBD.

## M0d-BLOCKER. Preprocessing modules lost to scratch-cleanup (NOT in git)
Discovered while wiring holo preproc: `src/masif_graph/{io,surface,pairs,align,metrics,score,train,
perturb}` lost their `.py` source; nothing is committed to git (0 commits) so no git restore.
**Impact:** the *eval* path is intact (all `p4/*.py`, `graph/hetero.py`), but `p4/precompute.py`
(needed to build npz for new holo+AF3 surfaces) imports:
- `io.reference` (load_complex, complex_is_available, PDB_DIR) — **.pyc present (cp38) → decompile**
- `surface.atoms` (build_surface_atoms) — **.pyc present → decompile**
- `graph.hetero` (build_hetero_graph) — **.py present ✓**
- `pairs.construct` (vertex_contacts, atom_positives_from_vertex_contacts) — **NO .pyc → rewrite**
Also lost (not needed for Phase 5 path): align/metrics/score/train/perturb (no pyc).
**Recovery plan:** (1) decompile io/surface/__init__ from .pyc; (2) rewrite pairs.construct from
precompute+hetero usage; (3) VALIDATE by regenerating an existing complex's npz and diffing vs the
stored `stageA_full_npz` ground truth (4872 available). The .sif pipeline itself is intact.

## RESUME STATE (updated)
- Phase: M0d recovery — restore preprocessing modules, validate against stored npz, then preproc 304.
- Running jobs: none. Spend: CHF 0.

## M0d-RECOVERY. Lost modules RESTORED from /work backup (not decompilation)
Found a full persistent source backup at `/work/upthomae/Meng/phase4/src/masif_graph/` (io/reference.py
8782 B, dated 2026-07-02 19:00 — byte-size + timestamp match the cp38 .pyc header, i.e. the exact code
that built the stored npz). Restored 25 missing .py (io, surface, pairs, align, metrics, score, train,
perturb, graph/build+model). All `p4.precompute` imports now resolve. Decompilation (decompyle3/uncompyle6
failed on bodies; pycdc gave structure w/ <NODE> artifacts) was abandoned in favour of the exact backup.
Mitigations: symlinked `.sif` (`/work/.../MaSIF/IMAGE/masif-neosurf_v0.1.sif`) into REFROOT; copied the
restored src to `/work/upthomae/Meng/phase5_src_backup/` so scratch-cleanup can't wipe it again.
Next: reconstruct holo `m0_run_one.sh`, smoke-test one clean-test complex that HAS stored npz, diff to validate.

## M0d. Pipeline reconstructed + fixed (download bug) — VALIDATING
- Bug: reference `00-pdb_download` uses dead `ftp.wwpdb.org` -> 126-byte empty stub -> extractPDB
  IndexError. FIX in `scripts/m0_run_one.sh`: fetch `https://files.rcsb.org/download/{PDB}.pdb` then
  protonate via .sif (01 re-protonates anyway). Pipeline then runs clean through 01/04 (surfaces,
  masif_site, ppi_search, shape-complementarity all rc=0). Final TF descriptor step OOMs on the LOGIN
  node (rc=137) — expected; needs a compute node.
- Wrote 4-cell retrieval harness `src/masif_graph/p5/retrieval_bench.py` (HH/AH/HA/AA x learned/frozen,
  DC-offset centering, shuffled control, per-complex ranks, robustness Δ). Syntax OK.

## RESUME STATE (updated)
- Phase: M0d validation. **Running job 65960109** = phase4_preproc on 4 val complexes (1BO4_A_B,
  1B3T_A_B, 1C8N_A_C, 1CQ3_A_B) on a compute node; when done -> precompute npz -> diff vs stored
  stageA_full_npz to validate; then submit full 304-set preproc.
- Spend: ~CHF 0.05 so far (validation job walltime-capped at 4.22 but exits early).
- Harness smoke pending on m2_npz.

## M0d2. Retrieval harness VALIDATED (smoke on old m2_npz, 30 cplx, DB=60)
Harness runs end-to-end. z_std=0.176 (centered, no collapse). Shuffled control medRank=32≈N/2 (chance ✓).
Preview (NOT the gate — leaky/small m2 set): learned ≫ frozen (HH/AA learned top5=0.65 vs frozen 0.28;
medRank 1 vs 14). Notable: learned same-state cells (HH,AA) strong; mixed-state (AH,HA) drop more
(query one state, DB other) — AA (both predicted, = the AF2-DB-queried-by-AF2 use case) is the strongest
learned cell, holo->AA drop ~0. Consistent with §23. Real gate awaits the clean 304 + AF3.

## M0d. Pipeline VALIDATED against stored npz — PASS
Rebuilt 4 complexes' npz, diffed vs stored stageA_full_npz: 1B3T & 1CQ3 byte-IDENTICAL (all arrays
maxΔ=0). 1C8N p2: coords/atoms/graph-topology identical, only surface feats wobble ~0.4 (MSMS/APBS
non-determinism — benign). 1BO4: current RCSB file has 3 more atoms than the old download (PDB-version
diff — benign). => pipeline logic correct; I build holo+AF3 fresh & internally consistent for the 304.
Preproc must run on a COMPUTE NODE (login OOMs the TF desc net, rc=137).

## RESUME STATE (updated)
- Phase: M0e/M1 kickoff. Submitting 304-set holo preproc (free CPU). Then AF3 gen.
- Validated: split(304), harness, holo pipeline. Spend ~CHF 0.1.

## M0e/M1. AF3 costed + launched — CHEAP; MSA running
- Holo npz build DONE: **301/301** clean-test -> /work/upthomae/Meng/phase5/npz (holo).
- 3 holo failures dropped: 2AFF_A_B, 2W0C_C_B, 3KTM_C_F. Eval set effectively 301.
- AF3 inputs: **591 chain JSONs** (obs seq) for the sc304 sides -> /work/upthomae/Meng/phase5_af3/inputs.
- **MSA (Jed): ~13 min/side**, ~CHF 13 for all (free-ish CPU). Array **65963839** (591 tasks, %60), ~2h ETA.
  Output: msa/{name}/{name}/{name}_data.json (2-level nest).
- **Kuma inference: 64 s/side @ NSAMP=1**, est CHF 0.17/job capped but ACTUAL ~CHF 0.01-0.02/side ->
  ~CHF 6-12 for all 591. GPU path verified (h100, qos debug=5 jobs/1h; mem<=5900MB/cpu -> use 16cpu/90G).
- **INCIDENT (fixed):** double-submitted the MSA array (65963785 + 65963839) because my jobid parse
  grepped the CHF number from sbatch output, not the real id. Cancelled 65963785; kept 65963839.
  LESSON: parse `Submitted batch job <id>` explicitly, never `grep -oE [0-9]+` on cost-bearing output.

## RESUME STATE (updated)
- Phase: M1 AF3. **Running: MSA array 65963839** (591 sides, ~2h). Holo npz done (301).
- After MSA -> Kuma inference (chunked, NSAMP=1) -> af3 surfaces (af3_model_to_surface) -> af3 npz -> M2 gate.
- Spend so far: ~CHF 15 (holo preproc + val + MSA-in-progress). Inference ~CHF 10 more. Well under 100.
- Files: eval set logs/phase5/eval_sc304.txt (301 usable); holo npz /work/upthomae/Meng/phase5/npz;
  af3 chains logs/phase5/af3_chains_all.txt (591); inference sbatch /work/upthomae/Meng/phase5_af3/p5_af3_infer.sbatch.

## M2 harness vectorized + validated
Rewrote src/masif_graph/p5/retrieval_bench.py scoring with matmul + torch.scatter_reduce (segment
max/min over DB atoms) -> fast. Verified byte-identical results vs the non-vectorized version on the
old m2 set (DB=60): HH_frozen 0.28/med14, HH_learned 0.65/med1, shuffled med33=chance. Gate ready.

## RESUME STATE (10:28)
- Phase: M1 AF3 gen. MSA array **65963839** progressing: 75/591 data.json done.
- Next: when MSA ~complete -> submit Kuma inference (INF_CHAINS=logs/phase5/af3_chains_all.txt,
  /work/upthomae/Meng/phase5_af3/p5_af3_infer_chunk.sbatch, qos=normal, K=45, array=1-14) ->
  p5_af3_surf.sbatch (301 complexes, af3 surfaces) -> p4.precompute --state af3 -> M2 gate on
  /work/upthomae/Meng/phase5 npz (holo+af3), ids logs/phase5/eval_sc304.txt, --center --pos-key pos.
- Holo npz DONE (301). Harness validated. Spend ~CHF 15.

## RESUME STATE (waiting on compute; 2 tracked monitors)
- **MSA array 65963839** running (~104/591 at last check, ~2h total). Monitor task b3gom3nmy exits when done.
- **AF3-path validation** (task bllvzs69p): waits for 8 cost-test models -> runs p5_af3_surf_batch on 4
  complexes -> p4.precompute --state af3 -> validates af3 npz path. Kuma infer job 3998529 (8 sides).
- **NEXT when MSA done**: submit full inference: on Kuma, `INF_CHAINS=logs/phase5/af3_chains_all.txt
  INF_K=45 INF_NSAMP=1 sbatch --qos=normal --array=1-14 /work/upthomae/Meng/phase5_af3/p5_af3_infer_chunk.sbatch`
  (chunk driver; skips done). Then `sbatch scripts/p5_af3_surf.sbatch logs/phase5/eval_sc304.txt 16`
  (compute node, af3 surfaces). Then `p4.precompute --ids eval_sc304 --out /work/upthomae/Meng/phase5/npz
  --state af3`. Then M2 gate:
  `python -m masif_graph.p5.retrieval_bench --data /work/upthomae/Meng/phase5/npz --ids logs/phase5/eval_sc304.txt
   --ckpt /work/upthomae/Meng/phase4/ret_full_ctr_best.pt --center --pos-key pos --out logs/phase5/gate_pos.json`
   (+ --pos-key pos_sc). Run on compute node / background (loads 301x4 chains ~10min).
- Results doc: docs/15-phase5-results.md (skeleton, fill from gate JSON). Spend ~CHF 16.

## M1. Full pipeline VALIDATED end-to-end (holo+af3 npz)
af3 surface must run on COMPUTE NODE (login OOMs desc net). 4 cost-test complexes: af3 surfaces built
(compute job 65964354), af3 npz produced, Rec loads holo+af3 ok (1CQ3 inter=90, retention=0.91 -> apo
keeps 91% of interface). Harness aborts <5 complexes (guard), not a bug. Pipeline fully de-risked.
Pipelining: firing full inference now (chunk driver skips no-MSA sides cheaply); re-fire after MSA.

## RESUME STATE (M1 running)
- MSA array **65963839** (Jed): ~266/591 done, ~60min ETA. Monitor task b3gom3nmy notifies on completion.
- Inference array **3998633** (Kuma, normal, %8): processing MSA-ready sides -> /work/.../phase5_af3/models.
  Chain list on SHARED /work (Kuma can't read Jed /scratch). NSAMP=1.
- **When MSA monitor fires**: (1) re-fire inference `ssh kuma "INF_CHAINS=/work/upthomae/Meng/phase5_af3/af3_chains_all.txt
  INF_K=45 INF_NSAMP=1 sbatch --qos=normal --array=1-14%8 /work/upthomae/Meng/phase5_af3/p5_af3_infer_chunk.sbatch"`
  (idempotent, skips done). (2) when all ~591 models exist -> `sbatch scripts/p5_af3_surf.sbatch logs/phase5/eval_sc304.txt 16`.
  (3) `p4.precompute --ids eval_sc304 --out /work/upthomae/Meng/phase5/npz --state af3`. (4) M2 gate (pos & pos_sc).
- Spend ~CHF 18.

## M1 timing correction
Actual AF3 inference = ~176s/side (cost-test single 64s was an anomaly). K=45 chunks would exceed the
1.5h walltime -> resubmitted K=24, 25 chunks (job 3998650), throttle raised to 16. Inference is now the
bottleneck (~591x176s/16 ~ 1.8h). Est total inference cost ~CHF 15 (h100 ~CHF0.5/GPU-hr subsidised).
Awaiting MSA monitor b3gom3nmy -> then re-fire inference (catch stragglers) -> af3 surface -> npz -> gate.

## M1c/M2. AF3 surfaces running -> gate chained
- AF3 models: 600 dirs; **284/304 complexes have BOTH models** (logs/phase5/_af3_both.txt).
- af3 surface job **65965106** (compute node, 284 complexes, ~90min).
- Monitor **bn94pqg96** chains: wait surfaces -> build af3 npz (p4.precompute --state af3) ->
  M2 gate x2 (pos -> gate_pos.json, pos_sc -> gate_possc.json). Notifies with gate numbers.
- Spend ~CHF 20.

## RESUME STATE
- Phase: M2 gate pending (monitor bn94pqg96). Holo npz done (301). AF3 ~284 complexes.
- After gate: interpret -> M3 graph ablation (needs no-graph retrieval encoder, see below) -> write-up.

## M2 GATE RESULT (pos_sc) — STRONG PASS
DB=294 (n=147 with pos_sc>=8 intersection holo+af3), center=True, z_std healthy. shuffled=chance (medRank158).
| cell | frozen t5/med | learned t5/med |
|---|---|---|
| HH holo->holo | 0.48/6 | 0.57/2 |
| AH af3->holo  | 0.37/14 | 0.50/5 |
| HA holo->af3  | 0.35/12 | 0.48/8 |
| AA af3->af3   | 0.33/18 | 0.56/2 |
Robustness (HH->AA top5 drop): frozen +0.150 vs **learned +0.017** (~9x more robust). Learned beats
frozen on HH (do-no-harm floor) AND every AF3 cell. GATE MET on pos_sc. (dense pos gate re-running.)

## M3 launched + dense pos gate running
- Added MASIF_NO_AA env flag to load_chain_graph (both /scratch + /work src): ablates atom-atom
  covalent (chem) edges. Verified aa_edge -> (2,0). Mirrors Phase-4 §18 ablation protocol.
- No-aa retrieval training submitted: Kuma job **3999274** (same recipe as ret_full_ctr + MASIF_NO_AA=1,
  init vicreg_sc, --center, 60ep). Saves p5_ret_noaa_best.pt. ETA ~2-4h.
- Dense **pos** gate still running (larger patches -> slower); pos_sc gate already PASSED.

## M2 GATE = PASS (pos_sc); dense pos OOM fixed
- **pos_sc gate PASSED decisively**: learned beats frozen on HH floor (0.57 vs 0.48 top5, med 2 vs 6)
  AND on every AF3 cell; holo->AA robustness drop learned +0.017 vs frozen +0.150 (~9x). Shuffled=chance.
- Dense pos gate kept getting SIGKILL/OOM (exit137): dense patches -> huge (nq x total_DB_atoms) matmul
  + cdist. FIX: added --max-patch (default 128) subsampling interface atoms/chain. 20-cplx dense preview
  (learned AA 0.68/med1 vs frozen 0.41/med6) confirms harness correct. Full dense re-running (bgkotfy7h).
- LESSON: bare `&` background procs get reaped when the tool call returns; and `| tail` masks a killed
  python as exit0. Use run_in_background + direct-log (no pipe) for long jobs.
- M3: no-aa training Kuma 3999274 running; monitor bizhb1dc0 (sentinel-wait) -> no-aa gate.
- Results doc 15 updated (§3-6) with the pos_sc PASS + preliminary verdict GATE MET. Spend ~CHF 25.

## M2 GATE = DECISIVE PASS (both patches, compute-node sbatch 65966197)
Login node SIGKILLs the big dense run -> ran gate via sbatch (scripts/p5_gate.sbatch, compute node). Both:
- **pos_sc** (DB=294,n=147): learned HH 0.57/med2 vs frozen 0.48/med6; AA 0.56/med2 vs 0.33/med18;
  robustness HH->AA drop learned +0.017 vs frozen +0.150.
- **pos dense** (DB=568,n=284): frozen COLLAPSES (HH 0.09/med114, AA 0.06/med134 = ~random) while
  **learned HH 0.63/med1, AA 0.64/med1**; learned holo->AA drop -0.005 (fully robust). Shuffled=chance both.
=> GATE MET decisively. Learned invariant encoder is the better AND robust deployment retriever on
AI-predicted structures; frozen needs the semi-oracular sc patch and collapses on realistic dense interfaces.
Results doc 15 §3-6 updated. Spend ~CHF 27.
- M3 no-aa: training 3999274 running (~20min in); monitor blxd8smpg -> submits no-aa gate sbatch on completion.

## LEAKAGE FOUND + FIXED (user-prompted, guardrail catch)
The encoder under test (ret_full_ctr_best.pt) was trained on retrieval_train_ids.txt (4872), which is NOT a
clean subset of training.txt — it **includes 60 TEST-list complexes** (Phase-4 stageA split sloppiness).
My original cluster-clean filter used training.txt as reference, so **16 of the 304 eval complexes were
exact-id members of the encoder's training set** (+1 seq-homolog). FIX: re-filter eval set to be clean
(exact + 30% cluster) vs the ACTUAL encoder training set -> **287 clean (269 with AF3)**
(logs/phase5/eval_sc304_clean_vs_enc.txt). Re-running the gate: job 65966870 -> gate_fullclean_{pos,pos_sc}.json.
Impact expected small (17/304=5.6%, leaks favoured learned) but the qualitative verdict (learned>>frozen) is
robust to it. Notebook + docs updated to the leak-free set; provenance cell added.

## PHASE 5 COMPLETE — GATE MET (leak-free)
Leak-free gates (287 clean, 269 w/ AF3): dense learned HH 0.63/med1, AA 0.64/med1; frozen 0.08/0.06 (collapse).
Robustness holo->AA drop learned -0.009 vs frozen +0.16. No-aa ablation: atom graph NOT the source (dense AA
0.62 vs 0.64, equally robust) — 4th null; graph helps only mixed-state cells. Results doc 15 finalized (§3-7 +
Phase-6 rec). Notebook notebooks/phase5_results.ipynb (inline code + provenance cell). Memory saved. Sentinel
logs/phase5/PHASE5_DONE touched. Spend ~CHF 38 of 100.
