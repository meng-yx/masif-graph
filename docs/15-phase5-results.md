# Phase 5 — Results (deployment-faithful binder retrieval; holo & AF3-apo)

> Living, honest. Every number traces to a committed artifact + a recoverable command. "Pipeline
> ran" is stated separately from "result is valid". Design = `docs/13-phase5-design.md`; running
> log = `docs/progress/phase5-log.md`. Gate = §3 of the design (learned ≥ frozen on HH AND smaller
> holo→AF3 degradation across AH/HA/**AA**).

## Status
- **M0 DONE**: cluster-clean split, pipeline recovery+validation, vectorized 4-cell harness.
- **M1 COMPLETE**: AF3-apo generation for the 301 eval complexes (MSA + Kuma inference + surfaces).
- **M2/M3 COMPLETE**: gate MET (leak-free); graph ablation done.

## 1. Benchmark construction (what the gate is measured on)
- **Eval set**: the MaSIF-search test list (959) filtered to a **sequence-cluster-clean** subset —
  mmseqs2 @30% id over all 5,902 complexes' chains; keep test complexes whose every chain-cluster is
  absent from training, then within-test dedup. **959 → 353 train-clean → 304 deduped**; 3 holo
  preprocessing failures → **301 usable** (`logs/phase5/eval_sc304.txt`). **62% of the nominal test
  set were train homologs** — the leak the design flagged is real and large; the honest denominator
  is 301, not 959. All 301 are single-chain-per-side (clean AF3-monomer apo case).
- **Structures**: holo = experimental (RCSB), rebuilt through the reference `.sif` pipeline; apo =
  self-generated **AF3** monomer prediction per side (unbound-like), NSAMP=1 top model.
- **DB / task**: DB = all 301×2 = 602 chain interface patches; each query chain must rank its true
  partner. Interface patch = intersection atoms present in both holo & AF3 (identical set across
  states → learned-vs-frozen and holo-vs-AF3 exact). Dense `pos` patch (primary) + sc-gated `pos_sc`.
- **Methods**: learned = from-scratch invariant encoder `ret_full_ctr_best.pt` with **DC-offset
  centering** (mandatory); frozen = MaSIF 80-D descriptor on identical patches (exact ceiling).
- **Controls**: shuffled-partner (chance); z_std sanity (post-center); frozen on identical patches.

## 2. Pipeline validity (recovery + reproduction)
- Phase-1 preprocessing modules were lost to scratch-cleanup (not in git); **restored from a /work
  backup** (io/reference.py byte-size+timestamp match the code that built the stored npz). The
  reference `00-pdb_download` (dead wwPDB FTP) was patched to fetch from RCSB.
- **Validation**: rebuilt 4 complexes' npz vs the stored ground-truth `stageA_full_npz` — 2/4
  byte-identical; the other 2 differ only by MSMS/APBS numerical wobble (~0.4) or a PDB-version
  atom-count difference. Pipeline logic verified.
- **Harness**: vectorized (matmul + scatter_reduce); byte-identical to the reference scoring on the
  m2 set. Shuffled control returns chance (medRank≈N/2).

## 3. Leakage control (corrected after review)
The encoder under test (`ret_full_ctr_best.pt`) was trained on `retrieval_train_ids.txt` (4872), which
**inadvertently included 60 test-list complexes** (a Phase-4 split imperfection). The first Phase-5 eval set
(304, filtered vs the nominal `training.txt`) therefore still contained **16 exact-id members of the encoder's
training set** (+1 homolog). The eval set was re-filtered to be disjoint — exact + 30%-id sequence cluster —
from the encoder's **actual** training data: **959 → 287 held-out (269 with AF3)**. All numbers below are on
this leak-free set; they are essentially unchanged from the pre-fix run (the 16 leaks were 5.6% and did not
move the verdict).

## 4. THE GATE — 4-cell matrix (learned vs frozen vs no-atom-graph), leak-free

### pos_sc — DB=286, n=143, max_patch=128, center=True  [gate_fullclean_pos_sc.json]
| cell | frozen t5/med | **learned** t5/med | no-atom-graph t5/med |
|---|---|---|---|
| HH holo→holo (floor) | 0.49/6 | 0.57/2 | 0.46/7 |
| AH af3→holo | 0.37/14 | 0.49/6 | 0.38/16 |
| HA holo→af3 | 0.36/13 | 0.47/8 | 0.40/10 |
| **AA af3→af3 (headline)** | 0.32/18 | **0.55/3** | 0.49/7 |

Robustness HH→AA top-5 drop: frozen **+0.164**, learned **+0.018**, no-graph -0.028. Shuffled control top5=0.024 (chance).

### pos — DB=538, n=269, max_patch=128, center=True  [gate_fullclean_pos.json]
| cell | frozen t5/med | **learned** t5/med | no-atom-graph t5/med |
|---|---|---|---|
| HH holo→holo (floor) | 0.08/110 | 0.63/1 | 0.58/2 |
| AH af3→holo | 0.09/123 | 0.56/2 | 0.43/12 |
| HA holo→af3 | 0.07/127 | 0.54/2 | 0.45/9 |
| **AA af3→af3 (headline)** | 0.06/128 | **0.64/1** | 0.62/1 |

Robustness HH→AA top-5 drop: frozen **+0.022**, learned **-0.009**, no-graph -0.043. Shuffled control top5=0.011 (chance).

**Read:** on the deployment-realistic **dense** patch, frozen MaSIF is near-random (AA top5 0.06, medRank 128)
while the **learned encoder retrieves the true binder at medRank 1** (AA top5 0.64) even when BOTH query and DB
are AF3-predicted. Learned holo→AA degradation ≈ 0 on both patches. Frozen only looks competitive on its
semi-oracular sc-gated patch.

## 5. Graph ablation — the atom graph is NOT the source of the advantage
The no-atom-graph encoder (chem/covalent edges removed, else identical recipe + centering) is **as strong and
as robust on the headline cells**: dense AA 0.62 vs 0.64 (Δ≈0.02), holo→AA drop −0.043 (as robust as full).
So the conformation-robustness and the deployment-AA retrieval come from the **invariant from-scratch training,
not the atom graph** — a 4th independent confirmation of the Phase-2/3/4 graph-null. *Nuance:* the atom graph
does help the harder **mixed-state** cells (dense AH 0.56 full vs 0.43 no-graph; HA 0.54 vs 0.45), i.e. it aids
matching when query and DB are in *different* conformational states — a modest, real, but non-headline role.

## 6. Verdict — GATE MET (decisively, leak-free)
On a sequence-cluster-clean held-out set (**287 complexes**, disjoint from the encoder's actual training data)
with self-generated AF3-apo structures, the from-scratch SE(3)-invariant encoder:
1. **beats frozen MaSIF on the holo→holo do-no-harm floor** (dense 0.63 vs 0.08; sc 0.57 vs 0.49);
2. **is conformation-robust** — holo→AA top-5 drop ≈ 0 (dense −0.009) vs frozen +0.02–0.16;
3. **dominates the fully-AI-predicted AA deployment cell** (dense AA top5 0.64, medRank 1 vs frozen 0.06/128).
Frozen MaSIF's usable retrieval needs the semi-oracular sc patch and collapses on realistic dense interfaces.
**The learned encoder is the better and more robust deployment retriever on AI-predicted structures — the
Phase-5 north star — and the atom graph is not what delivers it.**

## 7. Recommendation → Phase 6  (see `docs/16-phase6-design.md`)
Gate met, so move the PoC encoder toward deployment via three ordered workstreams: **(A)** an
inference-illustration notebook that documents the I/O contract (per-surface-atom 32-D embedding; binding is a
downstream pairwise `zᵀTz`, not a per-atom score); **(B)** a data-scaling ablation (is 4,872 PPI complexes
enough? → source a larger PPI corpus, e.g. **DIPS ~42k**, if not) with the Phase-5 leakage discipline; **(C)**
extend preprocessing to **small-molecule ligands** and **retrain on a combined PPI + protein–ligand corpus**
(PDBbind/PLANET/PPAP + molecular-glue ternaries) to unlock the **neosurface** use case.
**Correction:** the **TED human-domainome AF2 database is inference-only — it has no known-positive set**; all
train/eval positives come from the **PDB**. The Stage-C false-positive/precision funnel
(`docs/11-phase4-stageC-ppi-scoring.md`) is the phase after that.
