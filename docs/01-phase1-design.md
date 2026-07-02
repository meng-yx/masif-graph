# Phase 1 — Per-atom reframing (no graph features yet)

> **Status:** Design. Prerequisite read: [00-context-and-goals.md](00-context-and-goals.md).
> **Purpose of Phase 1:** answer the make-or-break question *before* building the graph embeddings or retraining anything:
> **Does representing the surface as per-atom pooled fingerprints (instead of per-vertex) preserve the ability to tell true contacting atom pairs from decoys — and can we align proteins globally at atom granularity?**
> Graph embeddings (A/B), descriptor retraining, ensembles, and ligands are explicitly **out of scope** here (Phases 2–4).

## Operating constraints (from project kickoff)

- **Preprocess from raw PDBs** — no dependence on pre-existing preprocessed dumps. For Phase 1 the *probe* runs the **reference `masif-neosurf-af2` pipeline as an executable tool** to turn PDBs → surfaces + per-vertex 80-D descriptors; a clean reimplementation of preprocessing is a **parallel** Phase-1 track that does **not** block the go/no-go signal.
- **CPU cluster now; GPU later.** Everything in the critical path of the probe is CPU-feasible (MSMS/APBS are CPU; reference descriptor net runs CPU inference, slow but fine for tens of complexes). Any GPU-dependent step is flagged **[GPU-gated]** and is **not** started until GPU access + explicit go-ahead.
- **No autonomous heavy jobs** until the user says so. This doc plans the work; execution is gated.
- **Conda + pip**, two-environment reality (see §8).

---

## 1. Decisions resolved for Phase 1 (provisional defaults)

These lock the four Phase-1-gating decisions from `00-context-and-goals.md §5` to concrete defaults. Marked provisional — revisit if the probe says so.

| # | Decision | Phase-1 default | Rationale |
|---|---|---|---|
| **D1** | Per-atom surface embedding | **Pool pretrained per-vertex descriptors** (mean + max), reference net frozen | Cheapest path; isolates the granularity change; no training needed → CPU-only friendly. Escalate to atom-centric readout (D1-B) only if the probe fails. |
| **D4** | "Contacting atom pair" / positive definition | **Reuse the reference vertex-contact positives, mapped to owner atoms** (see §4) | Keeps the probe apples-to-apples with the reference's own descriptor-separation metric. |
| **D8** | Interface localization | **Not needed for the probe** (pairs are known); for the alignment prototype, reuse reference MaSIF-site iface propagated to atoms | Defer building an atom-level interface head; don't let it block Phase 1. |
| **D10** | Data/preprocessing boundary | **Re-run from PDB via reference pipeline; keep APBS electrostatics as a feature** (it is baked into the reference descriptors we pool) | Matches kickoff choice; electrostatics survives because we pool existing descriptors. Revisit APBS cost for v2 ensembles. |

Everything else (D2, D3, D5, D6, D7, D9) stays open — none is on the Phase-1 critical path.

---

## 2. The per-atom representation

### 2.1 Surface heavy atom (the unit)
For a preprocessed chain we have: the regularized surface mesh vertices `V` (coords + normals), and the chain's heavy atoms `A` (from the PDB, excluding H).

1. Build a KDTree over heavy-atom coordinates.
2. For each surface vertex `v`, find its **nearest heavy atom** `a(v)` → this is the **persisted vertex→atom index** the reference throws away.
3. **Surface heavy atom** = any heavy atom that is `a(v)` for at least one vertex. Atoms owning zero vertices (fully buried) are excluded from the surface-atom set (but remain available as bonded neighbors for Phase-2 graph A).

Persist per chain: `vertex_atom_idx` (len = n_vertices), and the surface-atom table (`atom_id`, element, residue, coord, `n_owned_vertices`, owned-vertex list).

### 2.2 Per-atom fields
For each surface atom `a`:
- **coord** = heavy-atom coordinate.
- **normal** = area/count-weighted mean of owned-vertex normals (unit-normalized). Carries an outward orientation for point-to-plane ICP later.
- **surface embedding (straight)** `e_str(a)` = pool over owned vertices of the reference `desc_straight` (80-D).
- **surface embedding (flipped)** `e_flip(a)` = pool over owned vertices of the reference `desc_flipped` (80-D).
  - **Pool the precomputed flipped descriptors directly** — do *not* flip the pooled straight vector. The flip is a non-linear transform through the reference net, so only per-vertex `desc_flipped` is valid to pool.
- **pooling operator**: start with **mean** and **max** (evaluate both in the probe). Attention/learned pooling deferred (needs training → [GPU-gated]).
- **exposure** `n_owned_vertices` (diagnostic; drives an optional min-exposure filter).
- (carried for later, not used in the probe metric) pooled raw 5 features, mean owned-vertex iface score.

Complementarity convention (matches reference search): compare **one partner's flipped** embedding to **the other partner's straight** embedding.

---

## 3. Milestones

### Milestone 0 — Probe input generation *(reference-as-tool, CPU)*
Pick **N ≈ 30–50** complexes from `data/lists/testing.txt` (held-out; small enough for CPU). For each, run the reference `masif-neosurf-af2` preprocessing + `compute_descriptors` from the raw PDB to produce, per chain: regularized `.ply` (verts, normals, iface), and per-vertex `desc_straight` / `desc_flipped` (80-D). Store paths; **no new training**.

### Milestone 1 — The pooling feasibility probe *(the gate, CPU)*
Build the per-atom representation (§2) for each probe complex, then measure **descriptor-separation ROC-AUC** at two granularities on the **same complexes**:

- **Per-vertex baseline** (reproduces the reference metric on the probe set): positives = contacting vertex pairs (§4.1); negatives = non-contacting pairs; score = `||desc_flip(v_A) − desc_straight(v_B)||`. AUC over positives(small distance) vs negatives(large).
- **Per-atom** (the thing under test): positives = contacting **atom** pairs (§4.2); negatives = non-contacting atom pairs; score = `||e_flip(a_A) − e_straight(a_B)||`. AUC.

Report, per pooling operator (mean, max):
1. per-atom AUC vs per-vertex AUC (headline).
2. AUC **stratified by atom exposure** (`n_owned_vertices` bins) — does a min-exposure filter help?
3. distance-distribution overlap plots (pos vs neg), atom vs vertex.

**Go / no-go:**
- **Greenlight** full Phase-1 build if per-atom AUC ≥ per-vertex AUC − ~0.02 (negligible loss), ideally ≈ baseline (~0.98 on this metric).
- **Escalate** to D1-B (atom-centric surface readout) or reconsider the representation if per-atom AUC drops materially (≳ 0.05), and check whether a min-exposure filter recovers it first.

### Milestone 2 — Global alignment prototype *(after M1 greenlight, CPU)*
Replace the reference **per-patch** RANSAC loop with **global, atom-level** registration on a few native complexes (start from a randomized binder pose):

1. **Global correspondences**: over all target×binder surface atoms, candidate pairs with fused-embedding (here surface-only) distance below a threshold.
2. **Robust fit**: `registration_ransac_based_on_correspondence` (Open3D — the *correspondence* variant, not feature-matching) over that global set → weighted **Kabsch/Umeyama** on inliers.
3. **Refine**: point-to-plane **ICP** using atom coords + per-atom normals.
4. **Sanity metric**: interface-RMSD of the recovered pose vs native; fraction of native contacts recovered. No learned scorer yet.

This validates that "minimize distance over low-embedding-distance pairs, globally" recovers native poses at atom granularity before we invest in the learned scorer (Phase-2/D7).

### Deliverable of Phase 1
A short results note (`docs/02-phase1-results.md`) with the two AUCs, the exposure stratification, the alignment-prototype RMSDs, and an explicit go/no-go recommendation for Phase 2.

---

## 4. Pair construction (atom level)

### 4.1 Reference vertex-contact positives (for the baseline)
Reproduce the reference positive definition on the probe complexes: p1 surface vertex whose nearest p2 vertex is < **1.0 Å** (`pos_interface_cutoff`), optionally gated by the shape-complementarity band (`0.5 < sc < 1.0`). Negatives: p2 vertices farther than the cutoff (within-complex) and vertices from other complexes (cross), mirroring `neg_mix`.

### 4.2 Contacting atom pairs (for the per-atom metric)
Map the §4.1 contacting vertex pairs to their **owner atoms** via `vertex_atom_idx`: `(a(v_p1), a(v_p2))` becomes a positive atom pair. Deduplicate. Negatives: non-contacting surface-atom pairs, sampled with the same cross/within/hard split as `neg_mix` (hard = atoms just outside the contact cutoff). This keeps the per-atom metric directly comparable to the per-vertex baseline (same underlying contacts).

> Open sub-choice (record result, don't block): an alternative positive definition is direct heavy-atom proximity (inter-atom distance < ~4 Å). The probe can report both to see which correlates better with embedding separability; §4.2 (vertex-derived) is the default because it inherits the surface-contact semantics the descriptors were trained on.

---

## 5. Explicitly NOT in Phase 1
- Graph embedding A (chemistry/bond) and B (geometry) — Phase 2.
- Retraining / reimplementing the surface descriptor net — Phase 2 (D3/D6). Phase 1 pools the **frozen reference** net.
- Unified contrastive complementarity objective — Phase 2 (D3).
- Learned pose scorer — Phase 2/D7 (M2 uses geometry-only sanity metrics).
- Relaxed ensembles, standalone ligands, liganded training data — Phase 4.
- Full-dataset (4,943-pair) preprocessing/training — [GPU-gated], after Phase-1 gate + user go-ahead.

---

## 6. Proposed module layout (new repo)

```
masif-graph/
  data/lists/                    # reused PDB lists (training/testing/…) — the only reused artifact
  docs/                          # 00 context, 01 phase-1 (this), 02 results, …
  src/masif_graph/
    io/                          # PDB & surface (.ply) readers, reference-output adapters
    surface/                     # vertex→atom mapping, per-atom pooling, surface-atom table
    pairs/                       # contacting-pair + neg_mix construction (atom & vertex)
    align/                       # global correspondence + RANSAC(correspondence)+Kabsch+ICP
    metrics/                     # descriptor-separation AUC, interface-RMSD, contact recovery
    experiments/                 # Milestone-1 probe, Milestone-2 alignment prototype
  scripts/                       # CLI entry points; SLURM submit stubs [GPU-gated]
  tests/
```
Phase 1 touches `io/`, `surface/`, `pairs/`, `align/`, `metrics/`, `experiments/`. `graph/`, `score/` arrive in Phase 2.

---

## 7. Phase-1-specific risks

- **T1 (the whole point):** pooling may blur sub-Å complementarity. → M1 measures it directly; exposure stratification + min-exposure filter are the first mitigations before escalating to D1-B.
- **Reference reproducibility:** Milestone 0 depends on the `masif-neosurf-af2` env building on CPU (TF1.13 + MSMS + APBS + PyMesh). Standing this up is the first practical hurdle; treat as its own task.
- **Metric leakage:** ensure the per-vertex baseline and per-atom metric use identical complexes and identical pos/neg construction, or the comparison is meaningless.
- **Small-N noise:** N≈30–50 complexes may give noisy AUC; report per-complex spread, not just the pooled number.

---

## 8. Environment & dependencies

Two conda environments, because the reference stack and the new stack conflict (TF1.13/py3.6 vs modern PyTorch/py3.10):

1. **`masif-neosurf-ref`** — the reference `masif-neosurf-af2` environment (Docker/py37 recipe already in that repo), used only to run Milestone 0 (preprocessing + descriptors). We do not recreate it here; we invoke the existing reference tooling.
2. **`masif-graph`** — the new env (`environment.yml` in this repo): Python 3.10, numpy/scipy/scikit-learn/pandas/matplotlib/jupyter, biopython, RDKit, Open3D, networkx; PyTorch (CPU wheels for now), PyG, e3nn via pip. All Phase-1 *new* code (mapping, pooling, metrics, alignment) runs here on CPU.

Native preprocessing tools (MSMS, APBS, PDB2PQR, reduce, PyMesh) live with the reference env; the new env does **not** need them for the probe (it consumes the reference's `.ply` + descriptor `.npy`).

---

## 9. Immediate next actions (all CPU, non-autonomous, gated on your go-ahead)

1. Stand up the reference `masif-neosurf-af2` env on the CPU cluster and confirm it preprocesses one training-list PDB end-to-end (surface + descriptors).
2. Implement `io/` + `surface/` (vertex→atom mapping, surface-atom table, pooling) against that one complex; visually sanity-check the mapping.
3. Implement `pairs/` + `metrics/` and run Milestone 1 on N≈30–50 complexes.
4. Write `docs/02-phase1-results.md` with the go/no-go.

Step 1 is the first thing to do and the main unknown (does the legacy stack build on this CPU cluster?). Everything else is straightforward Python once inputs exist.
