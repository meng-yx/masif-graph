# Phase 6 — Workstream C — results: a ligand-capable unified retrieval encoder

> Status: **COMPLETE** (2026-08-07). Every number below is traceable to a committed artefact under
> `logs/phase6C/results/` (force-added to git despite the `logs/` ignore rule, and mirrored to
> `/work/upthomae/Meng/phase6C/results/` so a /scratch cleanup cannot take them) and a command in §7. Total compute: **~CHF 7.5** of a CHF 100 budget
> (Jed 292 core-hours ≈ CHF 1.46; Kuma 11.6 GPU-hours ≈ CHF 6.05).

## 0. What was asked, and what "done" means
Retrain the from-scratch invariant encoder on a **combined corpus** — PPI complexes **+**
protein–ligand complexes — in a **unified 26-D atom feature space**, and report three axes:

1. **Do-no-harm PPI gate** — the Phase-5 retrieval gate on the frozen 287-complex clean eval set.
   The combined model must not meaningfully regress PPI retrieval.
2. **Mixed held-out** — cluster-clean protein–ligand (and PPI) held-out retrieval.
3. **Neosurface benchmark** — inference-only, on ligand-induced ternary complexes.

A negative result, honestly verified, is a valid finish. Throughout, **"the pipeline ran" is stated
separately from "the result is valid"**.

---

## 1. The representation (what actually got built)

**Unified 26-D atom features** (`src/masif_graph/p6/atoms.py`) describe *any* heavy atom — element,
is-ligand, backbone, aromatic, degree, is-surface, in-ring, hybridisation, H-bond donor/acceptor,
formal charge, sidechain flex depth, element chemistry — so protein and ligand atoms live in one
space and PPI complementarity can transfer to protein–ligand.

**Path B for ligands** (locked at HEAD `410238d`, after the `.sif score_binder` ligand-surface path
proved unfixable at scale): the ligand is **heavy atoms as graph nodes**, not a computed surface.
A protein–ligand complex is therefore emitted in exactly the artefact shape of a PPI complex —

| file | contents |
|---|---|
| `{cid}__holo__p1.npz` | protein: atom nodes + surface-vertex nodes + 3 invariant edge types |
| `{cid}__holo__p2.npz` | ligand: atom nodes + covalent edges, `n_vert = 0`, every atom a readout |
| `{cid}__contacts.npz` | positives as (protein surface row, ligand row); `pos` ≤5 Å, `pos_sc` ≤4 Å |

— so `p4.dataset.ComplexP4`, the chain-level retrieval loss and the whole Phase-4 training loop
consume the mixture with **no new code path**: the mixture is just a longer id list.

**The two sides are encoded independently.** This is the load-bearing detail. Injecting the ligand
into the protein graph — the naive reading of "ligand atoms as graph nodes" — would let protein
surface vertices message into the ligand, so a ligand embedding would already encode its own
protein and retrieval would be free. The metric would look excellent and mean nothing.

*What Path B gives up:* ligand-derived surface **vertices**, i.e. the ligand contributes chemistry
and connectivity but no MSMS shape channel. The protein side of every pair still carries full
surface information.

---

## 2. Corpus and the leakage discipline

| corpus | source | built | notes |
|---|---|---|---|
| PPI | Phase-4 `stageA_full_npz`, re-featurised to 26-D | **4,711 / 4,871** | 160 dropped by the verification below |
| protein–ligand | PDBbind refined 2020 (`INDEX_refined_data.2020`) | *(see §2.3)* | `.sif` surface + Path-B ligand graph |
| PPI eval (frozen) | Phase-5 287-clean list | 301 holo + 284 AF3 | never touched during training |
| neosurface | `masif-neosurf-af2/computational_benchmark` | **14 / 14 systems, 28 cases** | inference only |

### 2.1 The PPI corpus had to be recovered, not rebuilt
The /scratch 30-day cleanup had destroyed the reference surface tree: only **16 of 4,872** training
complexes still had their `.ply` + precompute inputs, so `p4.precompute` could not be re-run at 26-D.
The npz themselves survived on `/work`, and the 14→26-D change touches **only `atom_feat`**, so the
features were patched in place: 23 of 26 dims come from the stored vector or are exactly derivable
from it, and the 3 that need per-atom name + residue name (H-bond donor, acceptor, formal charge)
were recovered by re-extracting the chain from a fresh RCSB download.

This is only safe because it is **checked, not assumed**. Each chain must reproduce the stored 14-D
vector *bit-for-bit* and match the stored surface `keys` at every surface row; failures are dropped,
never patched. 40/40 pilot chains reproduced the reference chain PDB exactly; 301/301 Phase-5 eval
complexes passed; 160/4,871 training complexes failed and were dropped — traced to trailing residues
that RCSB has since added by entry remediation (e.g. `1PFF_A`: 2,519 re-derived vs 2,511 stored),
not to a systematic per-residue-type bias. Cost avoided: ~500 core-hours of surface rebuilding.

### 2.2 Split (`src/masif_graph/p6/split.py`)
Two leakage axes: protein **sequence cluster** (mmseqs2, 30% id, `--cov-mode 1 -c 0.5`) and ligand
**Bemis–Murcko scaffold**. Rules:
* the frozen eval set's protein clusters are **forbidden in both corpora** — a PDBbind target
  homologous to an eval chain leaks into the do-no-harm gate exactly as surely as a PPI one;
* held-out sets are carved as **whole connected components** of the "shares a protein cluster or a
  ligand scaffold" graph, computed over the **union** of both corpora.

That last word is load-bearing and was got wrong first: with per-corpus components, **203 of 300**
protein–ligand holdouts shared a sequence cluster with a PPI training complex, because a per-corpus
component graph cannot see a cross-corpus edge. The verify step caught it; after the fix all three
leak counters are **0** (eval-into-train, val-PL-into-train, PPI-holdout-into-train).

The scaffold graph is **degenerate** — scaffold edges chain components together until one swallows
>50% of the corpus — so the split falls back to protein clusters and scaffold overlap is *reported*
rather than pretended away: the `val_pl_scaffold_unseen` subset (clean on **both** axes) is
evaluated and reported separately alongside the full holdout.

### 2.3 Sizes

Built: PDBbind refined **5,240 / 5,316** (72 skipped by the >8,000-heavy-atom cap or having no chain
within 6 Å of the ligand, 3 `.sif` failures → **99.7%** of non-skipped); PPI **4,711 / 4,871**
re-featurised; Phase-5 eval 301 holo + 284 AF3; neosurface 14/14 systems → 28 cases.

| split | n | note |
|---|---|---|
| `train_ppi` | 4,418 | |
| `train_pl` | 4,546 | |
| `val_pl` | 300 | of which **198 scaffold-unseen** (clean on both axes) |
| `val_ppi_stageA` / `val_ppi_stageB` | 80 / 197 | checkpoint-selection monitors only |
| `eval_ppi` (frozen) | 287 | never used for selection |

**394 PDBbind complexes were dropped for being homologous to the PPI eval set.** Without the
cross-corpus filter those would have gone straight into the do-no-harm gate's training data.

### 2.4 Two harness validations worth stating before any result
1. **The re-featurised eval set is the Phase-5 eval set.** Frozen MaSIF on the 26-D npz reproduces
   the Phase-5 published gate *exactly* — HH top5 0.084 / medRank 110 and AA top5 0.061 / medRank 128
   on n=269, DB=538, the same three decimals and the same denominators. So the `atom_feat` patch left
   descriptors, interface definitions and the AF3 join untouched, and the learned numbers below are
   directly comparable to the Phase-5 bar (**HH 0.630 / AA 0.639**).
2. **All three axes read chance for an untrained encoder** (`randinit`): gate learned top5 0.009–0.015
   against a shuffled control of 0.011; mixed held-out top5 0.014–0.023 against chance 0.017–0.025;
   neosurface top5 0.000, median rank 262 of 596, ligand effect 12 better / 16 worse. The harness is
   not leaking.

---

## 3. Training

Phase-4 anti-collapse recipe, unchanged: VICReg pretrain (var 2.0 / cov 0.04) → chain-level
retrieval fine-tune; freeze-τ @ 0.1, T weight-decay 1e-3, lr 5e-4 cosine, d=64 / d_out=32 / L=4,
grad-clip 1.0, **DC-offset centering mandatory** at train and eval.

Divergences, logged as required:
* batches are built to contain **both** corpus types (`--pl-frac`), because the chain-level loss
  uses the in-batch chains as its hard decoy pool — an all-PPI or all-ligand batch never asks the
  model to separate a true partner from a plausible wrong one of the other type;
* `--max-patch 128` caps interface atoms per chain (Phase 4 was uncapped); the chain score matrix
  is O(N²·n_a·n_b) and PPI dense patches run to hundreds of atoms against a ligand's ~25;
* model selection on **mixed held-out MRR**, since protein–ligand complexes have no AF3 state.

**Controls trained alongside the combined model**, so the do-no-harm claim can attribute a change
to the right cause:
* `ppionly` — the identical recipe and identical PPI subset, no ligand data. Separates "26-D
  features changed something" from "adding ligands changed something".
* `randinit` — an untrained 26-D encoder, run through all three axes. This is the chance line.
* the Phase-5 14-D encoder (`ret_full_ctr_best.pt`) on the 14-D eval npz — the historical anchor.

---

## 4. Axis 1 — do-no-harm PPI gate

### Axis 1 — do-no-harm PPI gate (Phase-5 287-clean, dense `pos` patches)

| model | HH top5 | HH medR | AA top5 | AA medR | holo->AA drop | shuffled top5 |
|---|---|---|---|---|---|---|
| random-init (chance) | 0.015 | 258 | 0.009 | 253 | +0.006 | 0.011 |
| Phase-5 14-D encoder | 0.630 | 1 | 0.639 | 1 | -0.009 | 0.011 |
| 26-D PPI-only (control) | 0.651 | 1 | 0.660 | 1 | -0.009 | 0.011 |
| 26-D ligand-only (control) | 0.011 | 262 | 0.013 | 258 | -0.002 | 0.011 |
| 26-D COMBINED (deliverable) | 0.610 | 1 | 0.623 | 1 | -0.013 | 0.011 |
| *frozen MaSIF (same patches)* | 0.084 | 110 | 0.061 | 128 | +0.022 | 0.011 |

n = 269 complexes, DB = 538 chains (chance top5 ~ 0.0093).

**Reading.** The 26-D feature space is itself a small *gain*: the PPI-only 26-D control (0.651 / 0.660)
sits **above** the Phase-5 14-D encoder (0.630 / 0.639). Against that, the combined model gives back
some of it — **0.610 / 0.623**, i.e. **−0.020 / −0.016 vs the Phase-5 anchor** and **−0.041 / −0.037 vs
the matched 26-D PPI-only control**.

Is that "meaningful"? The Workstream-B seed spread on this metric was **±0.04** (1500-complex pairs:
0.613 vs 0.569), and each condition here is **one seed** — so the gap versus the Phase-5 anchor is
comfortably inside seed noise, and the gap versus the matched control is about one seed spread. Read
conservatively: **a small but probably real cost, not a collapse.** Median rank stays **1**, the model
remains ~7× frozen MaSIF (0.084), and conformational robustness is untouched — the holo→AF3-apo drop
is −0.013 (negative: AF3 slightly *better* than holo), matching both controls.

The ligand-only control lands at chance (0.011 / 0.013), which is the sanity check that this gate
measures PPI ability and nothing else.


## 5. Axis 2 — mixed held-out retrieval

### Axis 2 — mixed held-out retrieval (same-type decoy pool, centered)

| model | PPI top5 | PPI MRR | PPI medR | P-L top5 | P-L MRR | P-L medR | P-L pocket->lig top5 | P-L lig->pocket top5 | P-L shuffled top5 |
|---|---|---|---|---|---|---|---|---|---|
| random-init (chance) | 0.028 | 0.032 | 98 | 0.012 | 0.019 | 138 | 0.014 | 0.010 | 0.014 |
| 26-D PPI-only (control) | 0.602 | 0.590 | 1 | 0.021 | 0.025 | 148 | 0.021 | 0.021 | 0.014 |
| 26-D ligand-only (control) | 0.028 | 0.036 | 100 | 0.036 | 0.036 | 116 | 0.045 | 0.027 | 0.014 |
| 26-D COMBINED (deliverable) | 0.576 | 0.542 | 2 | 0.040 | 0.042 | 76 | 0.041 | 0.038 | 0.014 |
- PPI: DB 198, chance top5 0.0254, chance median rank ~99, chance MRR ~0.027
- P-L: DB 292, chance top5 0.0172, chance median rank ~146, chance MRR ~0.019

Scaffold-unseen subset (clean on protein cluster AND ligand scaffold):

| model | P-L top5 | P-L top1 | P-L medR | n | chance top5 |
|---|---|---|---|---|---|
| random-init (chance) | 0.008 | 0.000 | 98 | 382 | 0.0262 |
| 26-D PPI-only (control) | 0.024 | 0.008 | 94 | 382 | 0.0262 |
| 26-D ligand-only (control) | 0.042 | 0.005 | 77 | 382 | 0.0262 |
| 26-D COMBINED (deliverable) | 0.063 | 0.010 | 54 | 382 | 0.0262 |

Scaffold-deduplicated holdout (one complex per scaffold; removes the congeneric-decoy ambiguity that depresses top-1):

| model | P-L top5 | P-L top1 | P-L medR | n | chance top5 |
|---|---|---|---|---|---|
| random-init (chance) | 0.009 | 0.000 | 108 | 422 | 0.0237 |
| 26-D PPI-only (control) | 0.021 | 0.005 | 107 | 422 | 0.0237 |
| 26-D ligand-only (control) | 0.045 | 0.014 | 83 | 422 | 0.0237 |
| 26-D COMBINED (deliverable) | 0.066 | 0.009 | 56 | 422 | 0.0237 |

Train-set vs held-out retrieval (identical set sizes) — separates "cannot learn" from "cannot generalise":

| model | PPI train top5 | PPI held-out top5 | P-L train top5 | P-L held-out top5 |
|---|---|---|---|---|
| 26-D PPI-only (control) | 0.464 | 0.602 | 0.029 | 0.021 |
| 26-D ligand-only (control) | 0.028 | 0.028 | 0.041 | 0.036 |
| 26-D COMBINED (deliverable) | 0.429 | 0.576 | 0.095 | 0.040 |



**Reading — this is where the workstream's thesis is decided.**

*Protein–ligand retrieval is real but weak, and it is driven by the PPI data.* Across all three
holdout variants the ordering is identical and the margins are not small:

| holdout variant | chance medR | ppionly | plonly | **combined** |
|---|---|---|---|---|
| full (n=292 DB) | ~146 | 148 | 116 | **76** |
| scaffold-unseen (clean on both axes) | ~98 | 94 | 77 | **54** |
| scaffold-deduplicated | ~108 | 107 | 83 | **56** |

The **transfer hypothesis is supported**: the combined model roughly **doubles** the ligand-retrieval
signal of the ligand-only model (full holdout: 146→76 rank improvement vs 146→116), and PPI-only
training gives exactly nothing on ligands (148 ≈ chance 146). Two confounds were checked and both run
*against* this conclusion, so it is if anything understated:
* Stage-B protein–ligand exposure is **equal** by construction (§3), and the combined run's in-batch
  same-type decoy pool is *smaller* (16 ligands vs 32) — i.e. **easier** negatives than `plonly` got.
* Stage-A pretraining exposure is matched in total complex-visits (15 × 8,964 ≈ 30 × 4,546).

*But the absolute level is low.* Top-5 is 0.04–0.07 against a 0.02–0.03 chance line. This is a real
signal, **not a deployable virtual-screening retriever** — a median rank of 54 out of a 192-chain pool is far from
"find the binder".

**The train-vs-held-out diagnostic explains why, and rules out the boring explanation.** `plonly`
scores 0.041 on *training* complexes vs 0.036 held out — it never learned the task even on data it
saw, so its weakness is not overfitting. `combined` scores 0.095 train vs 0.040 held out: PPI data
more than doubles even the **training-set** fit, so it is teaching genuinely better complementarity
features rather than acting as a regulariser. The remaining train→held-out gap is the generalisation
limit at this corpus size.

*One oddity, explained:* PPI train-set retrieval (0.429–0.464) is **lower** than PPI held-out
(0.576–0.602). That is a property of the split, not a bug — holdouts were carved as *small* cluster
components (≤25), while the training pool keeps the giant components (largest 3,269), so training
complexes have far more confusable homologs in their own decoy pool.

**Selection caveat (stated, not hidden):** the checkpoint was selected on this same held-out set, so
these are the best of 8 evaluated epochs. The selection-free **final-epoch** numbers from the training
history are essentially identical — combined ligand medR 74 (vs 76 selected), PPI top5 0.571 (vs
0.576); `ppionly` PPI MRR 0.589 final vs 0.590 selected — because every curve had plateaued. Axis 1
and axis 3 are unaffected: neither was ever used for selection.

## 6. Axis 3 — neosurface benchmark

### Axis 3 — neosurface benchmark (28 ligand-induced cases)

| model | with-ligand top5 | medR | no-ligand top5 | medR | ligand helps/hurts/ties |
|---|---|---|---|---|---|
| random-init (chance) | 0.000 | 262 | 0.000 | 257 | 12/16/0 |
| 26-D PPI-only (control) | 0.000 | 236 | 0.071 | 294 | 14/14/0 |
| 26-D ligand-only (control) | 0.000 | 253 | 0.036 | 258 | 10/18/0 |
| 26-D COMBINED (deliverable) | 0.036 | 267 | 0.036 | 297 | 17/11/0 |
| *frozen MaSIF (ligand-blind by construction)* | n/a | n/a | 0.000 | 308 | n/a |

DB = 596 chains (568 held-out decoy chains), n = 28 cases, chance top5 = 0.0084.



**Reading — this axis is a NEGATIVE result, and the negative includes the published method.**

Nothing here beats chance. With DB = 596 whole chain surfaces, chance median rank is ~298; the models
land at 236–297 and **frozen MaSIF lands at 308**, i.e. no better than random either. Top-5 is 0.000–0.071
against a chance line of 0.0084 — on n=28 that is one or two lucky cases, not a capability.

The ligand-present vs ligand-absent contrast — the test of whether any *neosurface* signal exists — is
also inconclusive. The combined model improves on 17 of 28 cases and worsens on 11 (median rank 267
with the drug vs 297 without); under a coin-flip null that is p ≈ 0.17. Suggestive at best.

**Honest attribution of the null.** Three reasons this benchmark cannot presently decide the question,
in decreasing order of how much I trust them:
1. **n = 28 against a 596-chain database.** The benchmark is under-powered by construction; the
   published protocol uses 28 + 200 decoys, and even there per-system reporting is the norm.
2. **The query is deliberately oracle-free** — the patch is defined by the drug alone and DB entries are
   whole surfaces, because that is what deployment has. This is a much harder setting than the axis-1/2
   protocols, which use interface patches on both sides.
3. **Path B gives the drug no shape channel.** The ligand contributes chemistry and connectivity but no
   MSMS surface, which is precisely the geometric complementarity a *neosurface* is made of.

That frozen MaSIF also fails here is worth stating plainly: it means this is a hard benchmark under
this protocol, not a specific failure of the learned encoder. It does **not** license any claim that
the learned encoder is better — both are at chance, and "equally at chance" is not a win.

## 7. Reproduction

Artefacts (all committed paths are in this repo; large binaries live on `/work`, which is shared
Jed↔Kuma and is not on the /scratch cleanup timer):

| what | where |
|---|---|
| 26-D PPI npz / protein–ligand npz / eval npz / neosurface npz | `/work/upthomae/Meng/phase6C/npz_{ppi,pl,eval,neosurf}` |
| split + `split_report.json` | `logs/phase6C/split/` |
| checkpoints | `/work/upthomae/Meng/phase6C/ret_{combined,ppionly,plonly}_best.pt` |
| training curves | `/work/upthomae/Meng/phase6C/ret_*_result.json`, `logs/train_*.out` |
| all evaluation JSONs | `logs/phase6C/results/` |
| running log (decisions, spend, job ids) | `docs/progress/phase6C-log.md` |

```bash
# --- data (Jed) --------------------------------------------------------------------------------
sbatch --array=0-265%220 scripts/p6C_pdbbind_array.sbatch \
       logs/phase6C/pl_chunks /work/upthomae/Meng/phase6C/npz_pl     # PDBbind refined -> Path-B npz
sbatch --array=0-7%8       scripts/p6C_refeat_array.sbatch            # PPI 14-D -> 26-D, verified
sbatch --array=0-13        scripts/p6C_neosurf_array.sbatch /work/upthomae/Meng/phase6C/npz_neosurf
python -m masif_graph.p6.split --ppi-ids logs/phase6C/final_ppi.txt \
       --pl-ids logs/phase6C/final_pl.txt --out logs/phase6C/split
bash scripts/p6C_stage_split.sh logs/phase6C/split && bash scripts/p6C_make_npz_all.sh

# --- training (Kuma H100; /work is shared, so no data transfer is needed) -----------------------
ssh ymeng@kuma.hpc.epfl.ch 'cd /work/upthomae/Meng/phase6C && \
  sbatch p6C_kuma_pipeline.sbatch combined combined 15 32 0 && \
  sbatch p6C_kuma_pipeline.sbatch ppionly  ppionly  30 32 0 && \
  sbatch p6C_kuma_pipeline.sbatch plonly   plonly   30 32 0'

# --- evaluation: all three axes for one checkpoint (compute node, never the login node) ---------
sbatch scripts/p6C_gate.sbatch combined /work/upthomae/Meng/phase6C/ret_combined_best.pt \
       /work/upthomae/Meng/phase6C/npz_eval
python scripts/p6C_collect_results.py          # regenerates the tables above from the JSONs
```

## 8. Verdict

**The deliverable exists and is verified: a single 26-D encoder trained on 4,418 PPI + 4,546
protein–ligand complexes, cluster-clean on both leakage axes, evaluated on three axes with four
controls (random-init, PPI-only, ligand-only, and the Phase-5 14-D encoder).** Per axis:

| axis | verdict |
|---|---|
| **1. Do-no-harm PPI gate** | **PASS, with a small measured cost.** 0.610 / 0.623 vs the Phase-5 anchor 0.630 / 0.639 (inside the ±0.04 seed spread) and vs the matched 26-D PPI-only control 0.651 / 0.660 (≈ one seed spread). Median rank 1; conformational robustness preserved; ~7× frozen MaSIF. |
| **2. Mixed held-out** | **NEW CAPABILITY DEMONSTRATED, WEAK. Transfer hypothesis SUPPORTED.** Ligand retrieval is well above chance (scaffold-unseen median rank 54 of 192 vs chance 96 (measured 98)) and the combined model roughly doubles the ligand-only model's signal, while PPI-only training gives exactly chance. Absolute level is far from deployable. |
| **3. Neosurface benchmark** | **NEGATIVE / INCONCLUSIVE.** No model beats chance at DB=596, **including frozen MaSIF** (median rank 308 vs chance ~298). The ligand-present contrast is 17/11 (p≈0.17). n=28 is under-powered; this benchmark cannot presently decide the question. |

### What this does and does not license

**It licenses:** the claim that a *single* encoder in a *shared atom feature space* can carry both
protein–protein and protein–ligand complementarity, and that the protein–protein data measurably
improves the protein–ligand side. That is the mechanism Workstream C was built to test, and it is the
first positive result for the shared-chemistry axis (the four earlier nulls were about the atom
*graph* adding conformational robustness — a different claim, so there is no contradiction).

**It does not license:** any deployment claim for neosurface or molecular-glue binder search. Axis 3
is a null, axis 2's absolute numbers are far below usable, and one seed per condition is thin.

### Honest statement of scope
*The pipeline ran* — 5,240 PDBbind complexes preprocessed, 3 GPU runs, 3 axes, 4 controls, zero
leakage on all three counters, controls at chance, harness reproducing Phase-5 exactly. *The result is
valid* for axes 1 and 2 within the stated seed caveat. *The result is not established* for axis 3.

### If this is continued, in priority order
1. **Give the ligand a shape channel.** Path B's missing MSMS surface is the most plausible cause of
   the axis-3 null and the low axis-2 ceiling — a from-scratch ligand-surface tier reusing
   `computeMSMS` (which works) plus RDKit Gasteiger charges is the stretch goal already scoped.
2. **≥2 seeds per condition** before treating the −0.04 do-no-harm gap as real.
3. **A bigger neosurface benchmark** (MolGlueDB's 114 ternaries) — n=28 cannot resolve this effect.
4. **Ligand-axis robustness** (AF3-apo protein + experimental ligand), with the rule already fixed in §9.

## 9. Explicitly NOT in scope here (and one decision recorded for later)

* **No apo/predicted ligand conformations, anywhere.** Training is holo-only for both corpus types;
  the positive pair is always the two sides of *one* complex, never a holo-vs-apo pair, so the
  workflow never needs a per-entry conformational pair. AF3/apo states appear only in the axis-1
  Phase-5 gate, which is protein–protein. Axis-3 ligand geometry is the **experimental bound pose**
  (RCSB ModelServer instance endpoint) — matching the deployment assumption that a ligand-bearing
  structure always comes from experiment.
* **Ligand-axis robustness (AF3-apo protein + experimental ligand)** is the natural next test of the
  project's north star applied to protein–ligand. **Correction (2026-08-07, user):** an earlier draft
  of this bullet called it "out of scope for C, the handoff excludes new AF3/MSA generation". That
  was wrong. The handoff's "AF3/MSA are NOT needed here" was written when C's corpus was assumed to
  be the *existing* PPI set; C then added ~5,000 new protein–ligand complexes, and new training data
  implies generating its matching apo state. It is taken up in **Phase 7** (`docs/20-phase7-design.md`
  D7-7), scoped to the ~300 held-out `val_pl` proteins first. The rule is fixed by user decision:
  **the protein varies, the ligand stays at its experimental pose** — predicting an apo ligand
  conformation is meaningless, and if a future workflow needs a nominal pair, the ligand is
  duplicated unchanged.
* **Ligand-derived surface vertices.** Path B gives the ligand chemistry and connectivity but no
  MSMS shape channel (see §1). A from-scratch ligand-surface tier remains the stretch option.
* MolGlueDB's 114 ternary PDBs were **not** used: the masif-neosurf `computational_benchmark` is the
  better-specified asset (explicit subunit split + drug assignment + a decoy protocol), and it makes
  the number directly comparable to the published method. MolGlueDB stays available for a later,
  larger neosurface benchmark.
