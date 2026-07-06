# Phase 2 — Progress Log (append-only)

Conductor agent (Claude Fable 5, headless SLURM job **65042258** on Jed node jst017).
Mission: `PHASE2_HANDOFF.md`. Spec: `docs/03-phase2-design.md`. Deliverable:
`docs/04-phase2-results.md`. Budget: **CHF 100** (Jed job + Kuma jobs). Ethos:
`ml-research-guardrails` — *a crash is cheap, a confident wrong result is expensive.*

All times absolute (Europe/Zurich). CHF spend tracked cumulatively.

---

## 2026-07-03 ~16:05 — Iteration 1 start: orientation

Read in full: `PHASE2_HANDOFF.md`, `docs/03-phase2-design.md`, `docs/02-phase1-results.md`,
`docs/00-context-and-goals.md`, and all Phase-1 reuse modules
(`io/reference.py`, `surface/atoms.py`, `pairs/construct.py`, `metrics/separation.py`,
`experiments/probe_core.py`).

**Runtime context (from supervisor preflight):** internet OK (anthropic 405, rcsb 200),
apptainer 1.4.0, claude 2.1.196, Kuma SSH reachable (kuma1), `model_data -> model_data_paper`
symlink present. Deadline 2026-07-04 15:28. 16 CPU cores, 503 GB RAM on Jed node.

**State of the world (nothing Phase-2 done yet):** no `docs/04-phase2-results.md`, no
`logs/PHASE2_DONE`, no Kuma jobs submitted. Fresh start. **CHF spent so far: 0** (the Jed CPU
job's cost is the serial-QOS allocation, tracked below).

**Reusable assets confirmed:**
- **47 complexes** with complete holo surfaces+descriptors already computed by Phase-1 (list in
  scratchpad `complete_complexes.txt`). Holo side is reusable — I only need to generate the
  *repacked* (apo-like) surfaces + build atom graphs.
- Reference pipeline entry: `cd masif-neosurf-af2/masif/data/masif_ppi_search && ./data_prepare_one.sh <id> && ./compute_descriptors.sh <id>` (per `scripts/m0_run_one.sh`).
- Phase-1 baseline (do-no-harm floor): atom **mean-pool** descriptor-separation AUC —
  pooled neg_mix **0.916**, pooled random-neg **0.889**, per-complex median **0.939**
  (vertex baseline 0.944 / 0.946 / 0.972).

**Environment:** `/work/upthomae/Meng/conda_envs/masif-graph` (py3.11, numpy 2.4.6, scipy,
sklearn, plyfile). **Missing: torch, rdkit, biotite, PyG, FASPR.** g++/gcc/make/cmake present.

### Key design decisions (locked here; rationale in the doc)

- **D-P2.1 — Graph library:** implement the relational GNN in **pure PyTorch** (scatter/index_add),
  *not* PyG, for the CPU floor. Graphs are small (≤ few-k nodes/chain); pure-torch is far more
  robust to install than PyG+torch_cluster+torch_scatter version-matching, and matches the repo's
  self-contained-parser ethos. PyG install attempted opportunistically for the Kuma scale-up but
  the floor does **not** block on it. *(Divergence from handoff M0 "install PyG"; documented.)*
- **D-P2.2 — Bond perception:** use **biotite** residue-template connectivity as the *primary*
  for proteins (robust, gives bond orders for standard + many nonstandard residues), with RDKit
  available for cross-checks. Rationale: RDKit-from-PDB is fragile on protonated benchmark PDBs;
  biotite templates are deterministic. *(Handoff lists RDKit primary / biotite fallback; I invert
  for reliability on the protein-only Phase-2 scope. Documented.)*
- **D-P2.3 — Descriptor stays out of message passing.** The graph message-passes over *invariant*
  chemistry+geometry only (element, charge, degree, aromaticity, rotatable topology, RBF spatial
  distances). The frozen 80-D descriptor is fused only at the **readout head**:
  `fused1 = Proj(surf_straight ⊕ g)`, `fused2 = Proj(surf_flipped ⊕ g)` where `g` = graph readout
  (role-independent). Keeps complementarity (straight/flipped) entirely in the surface channel;
  avoids the graph entangling straight vs flipped (T3). Robustness mechanism: under repack the
  surface channel shifts but the covalent-anchored `g` does not, so the fused pair distance moves
  less. *(Deliberate reading of design "descriptor as node feature"; documented.)*
- **D-P2.4 — Torch build:** install **CUDA (cu121) torch** into the shared `/work` env so the same
  env is reusable on Kuma H100; it runs on CPU on Jed for the floor.
- **D-P2.5 — Repack mapping:** holo↔repacked atoms mapped by identity `(chain,resseq,atom_name)`
  (backbone fixed → only sidechain coords move). Repacked eval restricted to atoms that are surface
  atoms in *both* states; coverage/drop-rate reported. Contacts/positives stay defined on the holo
  backbone (unchanged).

**Next:** install env (background), study `data_prepare_one.sh` for repack surface regen,
build the atom-graph module, smoke ONE complex holo+repack.

## 2026-07-03 ~16:10 — M0-env done; FASPR validated

**Env installs (into `/work/upthomae/Meng/conda_envs/masif-graph`):**
- `torch 2.4.1+cu121` (CUDA build; CPU-runs on Jed, GPU-ready for Kuma H100). CUDA not avail on
  Jed node (expected). scatter/index_add verified.
- `rdkit 2026.3.3`, `biotite 1.6.0`, `networkx 3.6.1`.
- **FASPR** cloned + built at `/work/upthomae/Meng/tools/FASPR/FASPR` (+ `dun2010bbdep.bin`).
- **PyG deferred** (D-P2.1): relational GNN will be pure-torch. Will install for Kuma if needed.

**FASPR smoke (3TDM_A, isolated monomer):** runs in 0.04 s; **backbone bitwise-identical**
(CA/N/C/O unchanged), **sidechains repacked** (GLN2 CB/CG moved, LYS11 NZ moved 0.5 Å);
chain ID + residue numbering preserved; heavy-atom-only output (916→916). Confirms the
controlled fixed-backbone monomer repack the design specifies.

**Repacked-surface regen plan (reuse reference pipeline under a parallel id):** FASPR-repack each
extracted chain `01-benchmark_pdbs/{PDBID}_{Ck}.pdb` in isolation → assemble both into
`00-raw_pdbs/{PDBID}RP.pdb` (holo backbone frame) → run reference `01-...triangulate` (per chain),
`04-masif_precompute` (site+ppi_search), `compute_descriptors` under id `{PDBID}RP_{C1}_{C2}`.
Skip `00-pdb_download` (no RCSB entry). Only sidechain rotamers differ holo vs RP; everything
else identical (same reduce+MSMS+APBS+descriptor-net path) → clean controlled perturbation.

**CHF spent: 0** (Jed serial-QOS CPU job only so far).

## 2026-07-03 ~16:28 — M0 pipeline end-to-end GREEN; M1 architecture validated

**Repack surface regen works end-to-end.** `REPACK_STATUS 3TDM_A_B OK` in ~2.5 min (triangulate
×2, precompute site+ppi_search, descriptors — all rc=0). Holo 2562 vs repack 2666 surface
vertices → the surface genuinely moved under the sidechain repack (the perturbation is real).
Fixed two bugs in `repack_one.sh`: `set -u` PYTHONPATH guard; `timeout` can't call a bash
function (inlined the `singularity exec` calls). Added per-job `TMPDIR` isolation for parallel runs.

**Graph builder** (`graph/build.py`): biotite residue-template connectivity, 100% atom overlap
with io.reference table, bond orders + sidechain-rotatable flags + flex-depth + RBF spatial edges.

**Model + harness** (`graph/model.py`, `graph/dataset.py`, `train/harness.py`): relational GNN
(pure torch) + fusion head; contrastive trainer with per-chain rotamer augmentation; differential
holo→repack eval on the **intersection positive set** (same contacts both states).

**M1 validation on 3TDM (single complex):**
- Record build 1.5 s; 23 holo positives, 21 survive to the holo↔repack intersection (2 atoms
  buried after repack). Repack p1 surface 530→555 atoms (repack builds complete sidechains).
- **Rotation invariance: max|Δembed| = 1.3e-7** (machine precision) — model is rotation-invariant
  by construction (only distances/bond-types/flags feed it). **M1 T2 check PASSES.**
- Overfit single complex: loss 0.54→0.14 in 15 steps; eval AUC ~1.0 (as expected when trained on
  it). Pipeline can fit. Shuffled control noisy on n~120 (0.38–0.44); will verify ≈0.5 on full set.

**Holo sc-positive census (47 complete complexes, single-char chains):** only **14** have ≥10
positives, **16** ≥8, **25** ≥5 (matches Phase-1: sc-complementary contacts are sparse). Repacking
the 25 (≥5 pos) now (6-way parallel) as the floor pool; test set will use the ≥10-pos subset.

**COMPUTE-STRATEGY FINDING (important):** the scale bottleneck is **CPU surface preprocessing**
(MSMS/APBS/descriptor-net in the TF1 `.sif`, CPU-only), NOT GPU training. The Phase-2 GNN is tiny
(~1k-node graphs, ~25 complexes) and trains in seconds on CPU. So **Kuma GPU gives little benefit**;
the real scale-up is parallel Jed CPU preprocessing to enlarge N. I will prioritize the CPU
deliverable and keep Kuma spend minimal (budget is a ceiling, not a quota). **CHF spent: 0.**

## 2026-07-03 ~16:33 — M0-probe: the natural surface collapse under repack (premise holds)

Raw mean-pooled descriptors, **no training**, 6 repacked complexes (intersection positives):
| state | randneg pooled | randneg median | negmix pooled | shuffled |
|---|---|---|---|---|
| holo   | 0.901 | 0.890 | 0.938 | 0.454 |
| repack | 0.865 | 0.839 | 0.896 | 0.480 |

**Natural collapse (holo−repack): +0.036 randneg / +0.042 negmix.** Heterogeneous per complex:
1ERN +0.163, 2AOB +0.116, 1JXQ +0.051 collapse clearly; 3TDM +0.012 mild; 2A6P −0.004, 2BBA
−0.031 slightly reverse. **The apo-like perturbation does degrade the surface descriptor** (the
effect the graph must reduce) — modest (~0.04) but real. Holo raw AUC 0.901 reproduces Phase-1's
0.889 baseline → pipeline is consistent. Shuffled ~0.45–0.48 → ~0.5 (control sane). Will recompute
on the full repacked set. **CHF spent: 0.**

## 2026-07-03 ~16:50 — Iteration 2 (supervisor resume): reattach + launch floor ablation

**Reattach (no double-submit):** re-read log + handoff. State recovered:
- **Repacks 25/25 complete** (all RP surfaces present in reference `04a/04b-precomputation`).
- **Holo enlargement still running** (`holo_enlarge_batch.sh` PID 2553219, niced-19): 64 holo
  complexes now (up from 47), growing toward the 898-id candidate list (max 90 this batch).
- **Smoke driver was killed at the iter-1→iter-2 boundary (16:46)** *after* caching `records.pkl`
  (18 records, 16:41) but *before* writing `phase2_results.json` — training never finished. No
  results lost; the expensive record-build is cached.
- **No Kuma jobs submitted. CHF spent: 0.**

**Verified driver + harness against guardrails** (`run_phase2.py`, `train/harness.py`):
- Split **by complex**; holo+repack twins stay together; test complexes fully held out (only
  train complexes' repack state is seen, via augmentation) → **no twin leakage**. ✓
- Rotamer augmentation (p_aug=0.5) applied to **all** cells incl. surface_only → graph must beat a
  fair augmentation-trained surface baseline. ✓
- Eval on **intersection positives** (identical contacts both states); shuffled control per state;
  differential = pooled_randneg(holo) − pooled_randneg(repack). ✓

**Launched floor ablation** (PID 2571459, `logs/p2_floor/`): all 25 repacked ids, 5 cells
(surface_only/+covalent/+rotatability/spatial/full) × 3 seeds × 200 steps, min-pos 5,
test-min-pos 8, split-frac 0.6. Rebuilds records fresh from all 25 (18-record cache was stale).
Validating early log lines (raw baseline + shuffled control) before trusting the full run.
**CHF spent: 0** (Jed CPU only).

## 2026-07-03 ~17:15 — FLOOR ablation complete (p_aug=0.5, N=23; 14 train / 9 test)

Full pipeline GREEN end-to-end. Controls: rotation-invariance 1.6e-07 (pass); shuffled ~0.46–0.49
both states (pass); RAW holo randneg 0.876 reproduces Phase-1 (pass). Split by complex, no twin
leakage. **All numbers from `logs/p2_floor/phase2_results.json`.**

| cell | holo_rn | repack_rn | **DEG_rn** | holo_nm | repack_nm | **DEG_nm** |
|---|---|---|---|---|---|---|
| surface_only | 0.869±.005 | 0.853 | +0.015±.006 | 0.866 | 0.829 | **+0.037±.005** |
| covalent | 0.868 | 0.854 | +0.014 | 0.843 | 0.821 | +0.022 |
| rotatability | 0.867 | 0.848 | +0.020 | 0.849 | 0.826 | +0.023 |
| spatial | 0.858 | 0.849 | **+0.009** | 0.853 | 0.834 | **+0.019** |
| full | 0.863 | 0.839 | +0.024 | 0.839 | 0.813 | +0.026 |

**Reads (honest, mixed):**
- **randneg** metric (easy negatives): differences within seed noise (±.005); graph shows *no*
  robustness gain. Metric is insensitive (raw collapse only +0.036).
- **negmix** metric (hard cross-complex negatives, discriminative): surface_only degrades most
  (+0.037); **every graph cell degrades less**, spatial lowest (+0.019). Spatial's *absolute* apo
  perf (repack_nm 0.834) even exceeds surface_only (0.829) despite ~0.013 lower holo.
- **Attribution / T4:** spatial edges carry the robustness; chemistry (covalent/rotatability) helps
  less than spatial and does **not** stack — full (+0.026) is worse than spatial alone (+0.019).
  → chemistry partially *redundant* with spatial geometry, not orthogonal, at this scale.
- **Do-no-harm:** graph cells cost ~0.01 holo (spatial 0.858 vs 0.869). Borderline.
- **Per-complex spread caveat:** pooled DEG dominated by ONE near-random complex **3B5U**
  (holo negmix ~0.45, collapses 0.45→0.24 = +0.21 surface_only; spatial softens to +0.15). At N=9
  a single hard complex swings the pooled number. Median ≪ pooled. **Underpowered.**

**Experiment matrix launched (all CPU, CHF 0):**
- ✅ floor p_aug=0.5 N=23 → `logs/p2_floor`
- 🔄 noaug p_aug=0.0 N=23 → `logs/p2_noaug` (pure holo→apo transfer; isolates graph's intrinsic
  robustness when surface can't learn the perturbation)
- 🔄 scaled p_aug=0.5 N=29 → `logs/p2_scaled_aug` (6 new repacks 1A2W/1A99/1ACB/1AGQ/1AK4/1AN1 all
  qualified ≥8 inter-pos; 29 usable, ~12 test/~17 train)
- 🔄 scaled p_aug=0.0 N=29 → `logs/p2_scaled_noaug`

Added `--p-aug` arg to `run_phase2.py`. **CHF spent: 0.**

## 2026-07-03 ~17:34 — CPU relief; robust-median analysis; draft results doc

**Analysis of floor via `analyze_phase2.py` (per-complex MEDIAN degradation, robust to 3B5U):**
- negmix DEG(median): surface_only **+0.027**, spatial **+0.008**, full **+0.005** → the graph
  reduces degradation on the *typical* complex, not only the 3B5U outlier. Signal more real than the
  pooled number alone suggested.
- **Do-no-harm FAILS vs the correct anchor:** every *trained* cell (incl. surface_only) is below
  **raw mean-pool** on holo negmix (0.84–0.87 < 0.897) AND repack (best 0.834 < raw 0.854). The
  contrastive fusion head underperforms the frozen descriptors at N=14 train. Graph = "less-bad
  training", not "beats the descriptor". Key finding for the verdict.

**Ops:** stopped `holo_enlarge_batch` — node was 5× oversubscribed (load 81/16) with 3 ablation runs
+ enlarge thrashing; enlarge had diminishing returns (only 6 of 31 new complexes qualified, already
repacked & in the N=29 scaled set). Killed its xargs + in-flight `.sif` workers by PID; all 3
training runs verified ALIVE. (Lesson: `pkill -f reduce` matched my own shell cmdline → self-TERM;
switched to PID-targeted kills.)

**Wrote draft `docs/04-phase2-results.md`** (methods, controls, natural collapse, floor table +
reads; §4/§5/§7 pending noaug + scaled). Added `analyze_phase2.py`. **CHF spent: 0.**

## 2026-07-03 ~17:40 — DECISIVE noaug result: graph robustness is augmentation-dependent, not intrinsic

**noaug (p_aug=0, N=23) complete** — the true holo→apo transfer (train holo-only, test repack):
| cell | DEGnm(pool) | DEGnm(med) | holo_nm | repk_nm |
|---|---|---|---|---|
| surface_only | +0.034 | +0.029 | 0.862 | **0.827** |
| spatial | +0.040 | +0.026 | 0.840 | 0.800 |
| full | +0.039 | +0.029 | 0.811 | 0.772 |
| raw mean-pool | +0.043 | — | 0.897 | **0.854** |

**Without augmentation the graph gives NO robustness benefit** — degrades same/more than surface_only;
absolute apo *worse* with graph (repk_nm 0.800/0.772 < surface 0.827 < raw 0.854).

**Paired per-complex sign test (negmix DEG, graph vs surface_only):**
- **augmented (floor):** all graph cells help **7/9** complexes, mean +0.017 (sign p=0.18 — consistent
  but n.s. at N=9).
- **noaug:** spatial/full help only **3/9**, mean −0.004/−0.008 (p≈0.5) → graph HURTS in transfer.

**Interpretation:** the floor's graph benefit was the model *learning the FASPR perturbation from
augmentation*, not a conformation-invariant signal (D-P2.3 mechanism NOT realized). Removing
augmentation removes the benefit. **Do-no-harm FAILS throughout** (trained head < raw mean-pool on
holo; graph makes it worse). Added paired sign-test to `analyze_phase2.py`. Filled results §4.
**Verdict trending NO-GO** (confirm at N=29). **CHF spent: 0.**

## 2026-07-03 ~18:10 — N=29 CONFIRMATION + FINAL VERDICT: NO-GO (Phase 2 complete)

Both scaled runs finished (17 train / 12 test). Full 4-run analysis in `logs/p2_combined_analysis.txt`.

**Key reframing (correction to the design's §5 metric):** "differential degradation" is *confounded* —
a trained head lowers holo toward repack, shrinking the gap without improving apo (a constant scores 0
gap). The honest metric is **absolute apo AUC (repk_nm)**. On it, the ranking is **invariant across all
4 runs**: `raw mean-pool > surface-only > graph`.

| run | raw repk_nm | surf repk_nm | spatial | full |
|---|---|---|---|---|
| aug N=23 | 0.854 | 0.829 | 0.834 | 0.813 |
| noaug N=23 | 0.854 | 0.827 | 0.800 | 0.772 |
| aug N=29 | 0.886 | 0.865 | 0.832 | 0.831 |
| noaug N=29 | 0.886 | 0.847 | 0.799 | 0.766 |

**Findings:** (1) frozen descriptor + a *trivial trained head* already absorbs FASPR repack (noaug N=29
surface collapse **+0.002** vs raw +0.056); (2) the **graph injects pose-sensitivity** (noaug N=29 full
+0.041; per-complex Δ skewed positive) because its spatial edges move with the rotamers; (3) augmented
"benefit" is n.s. (best sign-p=0.15) and reverses without augmentation → an augmentation-memorisation
artifact; (4) do-no-harm FAILS everywhere (trained < Phase-1 baseline 0.889/0.916 on holo).

**VERDICT: NO-GO for Phase 3.** The graph does not deliver conformational robustness; it is worst on
absolute apo. **D1-B NOT triggered** — the ceiling is the read-out/fusion + the *too-mild FASPR proxy*,
not the frozen descriptor (a head already robustifies it). **Recommended next step:** re-scope to a
*harder* perturbation (backbone motion / true apo / AF2) before any graph rework or descriptor retrain;
hold D1-B as contingency. Guardrails all pass (shuffled ~0.5 both states; no twin leakage; per-complex
spread + paired sign tests reported; every number traces to a committed JSON). **Cumulative CHF: 0**
(CPU-only; no Kuma jobs ever submitted). Deliverable `docs/04-phase2-results.md` complete + self-verified
(handoff §9 checklist all ticked). Touching `logs/PHASE2_DONE`.
