# Phase 6 — From proof-of-concept to a deployable, ligand-capable neosurface binder-search model

> Design-ahead-of-code (write the doc, then build). Continues the house convention (numbered sections,
> explicit gates, honest NOT-in-scope). Phase 5 met its gate (`docs/15`): the from-scratch SE(3)-invariant
> encoder is a better and more conformation-robust binder retriever than frozen MaSIF on AI-predicted
> structures. Phase 6 turns that **proof-of-concept encoder** into something **deployable for the real goal
> — neosurface / molecular-glue binder search over an AF2 database.**

## 0. One-paragraph summary
Three workstreams, run in dependency order **A → B → C**: **(A)** an *inference-illustration* that nails the
encoder's I/O contract (what artefact it consumes and generates — a per-surface-atom embedding field, not a
per-atom score); **(B)** a *dataset-sufficiency* study — is the Phase-4 4,872-complex PPI training set enough,
and if not, where do we get a larger PPI corpus (DIPS etc.); **(C)** the payoff — extend preprocessing to
**small-molecule ligands** and **retrain on a combined PPI + protein–ligand corpus**, unlocking the
neosurface/molecular-glue use case. A is days; B is ~a week (a cheap go/no-go ablation + data acquisition);
C is the multi-week phase. **Correction to record:** the TED human-domainome AF2 database is **inference-only**
— it has *no* known-positive set; all train/eval positives come from the **PDB**.

## 1. Why Phase 6, and what is still only PoC
Phase 5 proved the *representation* works, but three things block deployment:
1. **The I/O contract is implicit.** Nobody outside the code knows exactly what the encoder eats and emits, or
   that "binding score" is a *downstream pairwise* `zᵀTz`, not a network output. This must be explicit before
   we change preprocessing (C) or hand the model to anyone.
2. **The training corpus may be small.** The encoder learned from 4,872 PPI complexes (complex-level, not even
   internally dedup'd). We do not know if that saturates; a bigger PPI set might raise the ceiling.
3. **It is protein–protein only.** The actual goal (per the north star + `masif-neosurf`) is **neosurface**
   search — surfaces *created by a bound small molecule* (e.g. CRBN+pomalidomide → ZF degrons). The current
   preprocessing and atom graph have **no ligand representation**. This is the real deliverable.

## 2. Recommended order & why: A → B → C
The order is forced by dependency and cheap-gate-first:
- **A first** — the I/O contract is a prerequisite for changing preprocessing in C, and it's ~free.
- **B second** — a *cheap go/no-go* (does more PPI data even help?) that sizes the ambition of C's corpus,
  and its data-sourcing runs in parallel with C's code changes.
- **C last** — the biggest effort and the strategic payoff; consumes both A (contract) and B (corpus).

The two data efforts converge: B's larger PPI set + C's protein–ligand set become **one combined training
corpus**, built with the *same sequence-cluster leakage control we hardened in Phase 5* (the 60-test-in-train
leak we caught is the cautionary tale — cluster-clean vs the *actual* train set, always).

## 3. Workstream A — inference-illustration & the I/O contract
**Goal:** a notebook that starts from one example `.pdb` and walks the full path to a prediction, with a
visualization at each stage, so the artefact shape is unambiguous.

**The contract (what the notebook makes explicit):**
- **Consumes:** a per-chain **HeteroSurfaceGraph** — surface heavy-atom nodes (14-D invariant chem features) +
  surface-vertex nodes (4-D MaSIF channels: shape-index, hbond, charge, hydrophobicity) + 3 invariant edge
  types (atom–atom covalent, vertex–vertex mesh, vertex–atom).
- **Emits:** a **per-surface-atom 32-D embedding** `z` (L2-normalized) — an *embedding field over the surface*,
  **not** a scalar score.
- **Score:** binding is a **downstream, pairwise** computation — `s = median_i max_j (z^q_i)ᵀ T z^d_j` over
  interface atoms (learned bilinear `T`), one number per chain-pair; retrieval ranks a DB by this.

**Cells (each with a viz):** example PDB → MSMS surface + the 4 vertex channels (colored mesh) → atom graph
(bonds) → the hetero-graph npz → `encoder(graph) → (N_surf_atoms, 32)` (e.g. PCA-to-RGB of the embedding on
the surface) → the pairwise `zᵀTz` heatmap for a true partner vs a decoy. **Deliverable:** the reader can state
the artefact in one sentence. *(Standalone, ~1–2 days, no training.)*

## 4. Workstream B — is 4,872 enough, and where to get more PPI structure
Two parts; do (a) before investing in (b).
- **(a) Data-scaling ablation (the go/no-go).** Retrain the encoder on random subsets (e.g. 1k / 2k / 4.8k
  complexes, ≥2 seeds) and plot the **Phase-5 gate metric** (dense-AA retrieval + holo→AA robustness) vs
  training-set size. *If the curve is still climbing at 4.8k → more data helps (fund b). If flat → saturated;
  C should invest in the ligand axis, not scale.* Reuses the Phase-4/5 pipeline; ~a few GPU runs.
- **(b) Source a larger PPI corpus (only if (a) says so).** Candidates, roughly in order of yield/ease:
  **DIPS** (~42k binary complexes mined from the PDB — the standard large PPI set), PDB biological-assembly
  queries via **PISA**, **Dockground**, **ProtCID**. All are PDB-derived → same `.sif` surface pipeline
  applies. **Leakage discipline is mandatory:** sequence-cluster (≤30% id) split, and the eval set must be
  clean vs the *actual* training complexes (Phase-5 lesson).

## 5. Workstream C — ligand-capable preprocessing + combined retrain (the payoff)
**Goal:** a model that scores **neosurfaces** — surfaces shaped by a bound small molecule — so it can find
which proteome domains bind a ligand-bearing target (molecular-glue substrate discovery).
Three parts:
- **(a) Ligand-capable preprocessing.** The neosurf `.sif` (`masif-neosurf-af2 -b score_binder`) already builds
  **ligand-modified surfaces** (it processes a model with a small-molecule ligand and restricts the query patch
  to atoms/vertices around the ligand). The new work is on the **atom-graph side**: represent **ligand heavy
  atoms** in the graph (element/bond/aromaticity features beyond the 20 amino acids; ligand bonds from the SDF/
  mol2) and define the **neosurface patch** (surface around the ligand). Reclone note: the ligand-aware repo is
  `git@github.com:meng-yx/masif-neosurf-af2.git -b score_binder`.
- **(b) Combined training corpus.** PPI (Phase-5 set, or B's larger one) **+** protein–ligand interfaces —
  **PDBbind / PLANET / PPAP** (already on `/work/upthomae/Meng/PDBBindplus/`) **+** neosurface pairs
  (protein·ligand → partner). Cluster-clean split across the union. Decide the training objective mixture
  (PPI contrastive + neosurface contrastive; possibly a shared encoder with a ligand-aware node type).
- **(c) Retrain + re-evaluate.** Retrain the from-scratch encoder on the combined corpus; evaluate on **(i)**
  the Phase-5 PPI gate (do-no-harm — must not regress) and **(ii)** a **new neosurface benchmark** (e.g. known
  molecular-glue ternary systems: CRBN+IMiD → ZF degrons, DCAF15, etc.) — retrieval of the true recruited
  domain, with a ligand-present vs ligand-absent contrast to show the neosurface signal is real.

## 6. Deployment framing (record the correction)
The **TED human-domainome AF2 database** (`/work/upthomae/Meng/TED_human_domainome_MaSIF/`, ~25.7k
intracellular domains, frozen descriptors precomputed) is used **only for inference** — running the trained
encoder over it to produce a **probabilistic ranking** of candidate binders for a target. It contains **no
known-positive set** and is **not** a validation set. All **training and evaluation positives come from the
PDB** (PPI complexes + protein–ligand + ternary molecular-glue structures). Any "does it work" claim is made on
PDB-derived held-out benchmarks, never on the domainome.

## 7. Gates / success criteria
- **A:** the notebook runs end-to-end from a PDB to a pairwise score, with the artefact shape documented. (No
  metric — a communication deliverable.)
- **B:** a data-scaling curve with a clear verdict (saturated vs still-climbing at 4,872) that decides whether
  to acquire a larger PPI corpus.
- **C:** **(do-no-harm)** combined-retrain does not regress the Phase-5 PPI gate; **(new capability)** on a
  held-out neosurface/molecular-glue benchmark it retrieves the true recruited domain above chance and above a
  ligand-absent control — i.e. the neosurface signal is real and ligand-dependent.

## 8. Risks & traps
1. **Leakage (again).** The union corpus must be cluster-clean vs the *actual* train set; ternary systems share
   proteins (many CRBN structures) → dedup by protein cluster, not PDB id. (Phase-5 caught a 60-complex leak.)
2. **Neosurface ground truth is thin.** Curated molecular-glue ternary sets are small; the benchmark may be
   low-n → report per-system, not just pooled, and lean on the ligand-present/absent contrast as the control.
3. **Ligand atom features / bond perception** (protonation, tautomers, metals) are finicky — smoke-test on a
   few known ternaries before scaling.
4. **AF2/AF3 rarely model the ligand.** The apo/predicted side of a neosurface query may lack the ligand → the
   deployment query likely uses an *experimental or docked* ligand pose; state this assumption explicitly.
5. **Data-scaling ablation must be honest** — same eval, ≥2 seeds, report spread; don't over-read a noisy curve.

## 9. Explicitly NOT in Phase 6
- Validating on the TED domainome as if it had positives (it doesn't — inference only).
- The Stage-C protein-level **false-positive / precision funnel** (OT correspondence scorer + restraint
  co-folding to turn a retrieval shortlist into calibrated yes/no binding calls) — that is the **next** phase,
  built on Phase-6's ligand-capable retriever (`docs/11-phase4-stageC-ppi-scoring.md`).
- A learned pose scorer / aligner hardening.

## 10. Compute & cost sketch
- **A:** CPU, ~free (one example through the existing pipeline + encoder).
- **B(a):** a few GPU retrain runs (Kuma H100, ~CHF 2–3 each) + CPU preprocessing of subsets (free-ish).
- **B(b)/C:** the real cost — surface preprocessing of a larger corpus (CPU, `.sif`, hours–days of wall) +
  ligand preprocessing + one or more combined-retrain runs (GPU). Budget-gate each; keep the per-session
  spend guardrail (`CLAUDE.md`). Cheapest-sufficient first: the B(a) go/no-go before any large data buy-in.
