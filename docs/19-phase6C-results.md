# Phase 6 — Workstream C — results: a ligand-capable unified retrieval encoder

> Status: **IN PROGRESS** (this document is filled as each axis lands; every number below is
> traceable to a committed artefact under `logs/phase6C/results/` and a command in §7).

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
*(filled when the PDBbind array completes)*

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
*(pending)*

## 5. Axis 2 — mixed held-out retrieval
*(pending)*

## 6. Axis 3 — neosurface benchmark
*(pending)*

## 7. Reproduction
*(pending — commands + artefact paths)*

## 8. Verdict
*(pending)*
