# Phase 4 — A from-scratch heterogeneous GNN for conformation-invariant interface matching

> **Status: design (2026-07-07).** Supersedes the frozen-descriptor line of Phases 1–3. Reopens **D1**
> (surface-embedding architecture), commits **D3-A** (unified learned complementarity), and flips **D6**
> from *reuse-frozen* to *retrain-from-scratch*. Read `07-phase3-results.md` first — this plan exists
> because of what Phase 3 measured and ruled out.

## 0. One-paragraph summary

Phases 1–3 kept the pretrained MaSIF descriptor **frozen** and layered structure on top (pooling → atom
graph → learnable head). That line is exhausted: the frozen 80-D descriptor is a **strong ceiling**
(~0.90 holo), the chemistry graph added **nothing** when late-concatenated, and unfreezing only a head on
top of the frozen bottleneck recovered just **+0.016 of the +0.069 induced-fit gap**. Phase 4 unfreezes the
*whole representation*: a **single heterogeneous GNN** — surface-vertex nodes (carrying MaSIF's hand-crafted
input channels) **and** atom nodes (carrying chemistry), with message passing along covalent bonds and
along vertex↔atom edges — trained **from scratch** for two things at once: (a) an atom's embedding is
**invariant across conformers** of the same protein (holo ≈ any AF3 sample), and (b) **interface atom-atom
correspondence** — contacting atoms across two partners match under a learned complementarity operator. The
target is sharpened and well-posed: **close the addressable induced-fit gap so querying with any one of the
5 AF3 samples retrieves like the holo model.** The unaddressable ~23% structural-mismatch cases are
explicitly out of scope (§6).

## 1. Why Phase 4, and why now (what Phases 1–3 established)

Three findings, each load-bearing (all from `07-phase3-results.md`):

1. **The gap is real, moderate, and mostly induced-fit.** Holo→AF3 descriptor-separation gap ≈ **+0.08
   pooled / +0.06 median**; after removing the ~23% structural-mismatch monomers the **addressable
   induced-fit gap is +0.069**. That +0.069 is the Phase-4 objective.
2. **The frozen descriptor is a strong ceiling that head-only learning can't breach.** M3 (learnable
   DiffusionNet + chem-graph *head* on the frozen 80-D, residual-anchored) beat frozen by only **+0.016**
   (8/8 seeds, robust but modest), and **complex-count scaling flatlined** (52→128: +0.014→+0.016). More
   data on the *same frozen-bottleneck architecture* will not close the gap.
3. **The chemistry graph, as tested, was inert — but it was tested in its weakest form.** M3 fused the graph
   by **late concatenation**; the frozen descriptor **never message-passed along a single bond**. Phase-2's
   NO-GO likewise tested the graph carrying signal on its own. The coupling the north star actually
   implies — *descriptor propagating through the connectivity graph* — was never cleanly tested. Phase 4
   tests exactly that, from scratch.

**Consequence:** the honest next bet is not "iterate the head" but "unfreeze below the descriptor." That is a
larger build and a real fork (D1/D6), justified by the evidence above, not by optimism about the graph.

## 2. Sharpened goal (the Phase-4 gate)

> **For induced-fit complexes**, the per-atom embeddings of the {holo + 5 AF3 samples} of the same interface
> atom collapse (in the matching-relevant directions) to nearly one point, so that **any one of those 6
> conformations used as the query retrieves the true holo partner ≈ as well as the holo query would**, and
> identifies the correct contacting atoms — **without harming holo→holo** (the do-no-harm floor).

Measured on the Phase-3 M1 benchmark (§7). Success = AF3-query AF3→holo AUC approaching the ~0.90 holo
ceiling **and** the per-sample spread across the 5 AF3 samples shrinking toward holo. The frozen-M3 result
(+0.016) is the bar to beat; the induced-fit gap (+0.069) is the target to close.

## 3. Design decisions this phase resolves

| Decision | Phase 1–3 | Phase 4 | Rationale |
|---|---|---|---|
| **D1** (surface embedding) | A: pool a frozen per-vertex net | **Reopened → GNN-native.** Surface features are *vertex node inputs*; the descriptor is *learned by the GNN itself* (vertex↔vertex mesh MP = learnable geodesic conv). No separate CNN. | The frozen bottleneck is the ceiling (§1.2). Unfreeze the representation, not a head. |
| **D6** (retrain vs reuse) | reuse frozen MaSIF | **Retrain from scratch** (PyTorch, one framework) | Head-on-frozen exhausted; must access sub-descriptor headroom. |
| **D3** (complementarity) | frozen flip trick | **D3-A: unified learned complementarity** — a learned symmetric bilinear operator `T` subsumes the flip. | Flip is a hardcoded `T = diag(±1)`; learn it. Keeps fast inner-product retrieval. |
| **D2** (rotation invariance) | n/a (frozen inv.) | **Invariant scalars** on all geometric edge features (distances, angles-to-normal). No raw xyz into the descriptor. | Non-negotiable: descriptor-distance matching breaks otherwise. Equivariant channels (e3nn/GVP) deferred to a possible scorer, not the descriptor. |
| **D4** (positives) | vertex contacts | **Sc-filter interface vertices → map to interface atoms**; contact = shared/adjacent interface vertices. Hard-negative mining (§5.2). | Reuses MaSIF's validated interface definition at atom granularity. |

**This reopens D1 explicitly.** Phase-1's D1-A leaning ("pool a frozen net first, revisit B only if it
underperforms") has now been tested to exhaustion; the evidence says go past even D1-B to a GNN-native,
from-scratch descriptor. State this divergence loudly for anyone reading the D-decision log.

## 4. Architecture — one heterogeneous GNN

**Nodes.**
- **Atom nodes** — *all* heavy atoms (surface *and* buried). Buried atoms carry no surface signal but
  provide covalent context, so a surface atom's embedding is informed by its full sidechain connectivity
  (the connectivity/rotatability prior the project rests on). Node features: element, formal charge,
  hybridization, degree, aromaticity, H-count (reuse the Phase-2 graph builder).
- **Surface-vertex nodes** — MSMS surface vertices, carrying **MaSIF's hand-crafted input channels**
  (shape index, distance-dependent curvature, Poisson–Boltzmann electrostatics, hydropathy, H-bond
  potential). We adopt MaSIF's *inputs* and its *geodesic-locality prior* — not its CNN.

**Edges (all geometric features expressed as SE(3)-invariant scalars — D2).**
- **atom–atom, covalent only** (bond order, rotatable flag, element-pair as edge features). *No through-space
  atom edges* — Phase-2 found spatial edges inject pose-sensitivity; we want invariance.
- **vertex–vertex, mesh adjacency** — lets surface signal diffuse over the surface. This is the learnable
  replacement for MaSIF's geodesic convolution; **without it the GNN can't build a real surface descriptor.**
- **vertex–atom** — attaches surface to chemistry. *Definition is an ablation (M1):* start (i) top-k /
  distance-cutoff (cheap, gets the pipeline running); test (ii) the **geodesic-patch** variant (nearest
  vertex → geodesic circle → edges to all vertices within, with distance + angle-to-normal features) — the
  most MaSIF-faithful and likely best, but needs geodesic computation. Buried atoms have no vertex edges.

**Message passing.** A few layers of heterogeneous MP (atom↔atom, vertex↔vertex, vertex↔atom). Output = a
per-**surface-atom** embedding `z` (buried atoms contribute only as context). Everything is trainable;
nothing is frozen.

**Scale note.** MaSIF surfaces are 10⁴–10⁵ vertices; heterogeneous MP over that × thousands of complexes is
heavy. Plan to **subsample/coarsen the mesh** (AtomSurf and MaSIF both do — read AtomSurf's vertex handling
before committing) and cap vertices per protein in M0.

## 5. Training objective — invariance + correspondence from one contrastive loss

One shared encoder `E`; `z = E(atom | protein, conformer)`.

### 5.1 Correspondence via a learned complementarity operator (replaces the flip trick — D3-A)

Contacting atoms X (on A) and Y (on B) sit in *complementary*, not identical, environments. Score through a
**learned symmetric bilinear form**:

```
s(X, Y) = zₓᵀ T z_Y ,    T = Tᵀ (learned)
```

The flip trick is the special case `T = diag(±1)` on shape channels — we learn the general `T`. Symmetric `T`
makes scoring order-independent (either protein can be query). **Deployment stays a fast retrieval
primitive:** transform every database atom once (`T z_d`), then max-inner-product search — a per-pair MLP
scorer would destroy this and is disallowed for the matcher.

### 5.2 Contrastive loss with hard negatives (not margin + random)

MaSIF/M3 used pairwise margin with *random* negatives (easy → modest retrieval, top-5 0.64–0.78). Use
InfoNCE with mined hard negatives:

```
L_corr = − log  exp(s(X,Y)/τ) / [ exp(s(X,Y)/τ) + Σ_{Y'∈neg} exp(s(X,Y')/τ) ]
```

Negatives, escalating: other-protein atoms (easy) → the same partner's non-contacting surface atoms
(medium) → atoms from **decoy partners that don't bind** (hard). Hard negatives force real complementarity
rather than "any nearby surface."

### 5.3 Invariance as an emergent property (not a collapse-prone standalone term)

The key move for the sharpened goal. Instead of an explicit "pull the 5 AF3 embeddings to holo" loss (trivial
all-collapse minimum; the M3 anchor headache), **draw the query-side embedding from a random conformer** each
step:

```
zₓ ← E(X | c),   c ~ {holo, AF3 sample 1..k} ;   target z_Y from holo
```

The contrastive loss now demands *recover the true contact regardless of which conformer produced the
query* → `E` must map all conformers of X to embeddings that score identically against Y under `T`, i.e.
**conformer-invariant exactly in the matching-relevant directions**, free elsewhere. Invariance falls out of
the task and **cannot collapse** (hard negatives must still be separated). This optimizes the deployment
metric directly.

- **Two-conformer positives:** sample two different conformers of X, require both to match Y — a stronger
  invariance push.
- **Optional auxiliary consistency** (small weight, ablated): `‖z_i^{AF3} − z_i^{holo}‖²` over
  identity-matched atoms (Phase-3 relabel machinery). Targets the per-atom AF3≈holo number directly; keep it
  auxiliary — over-weighting collapses nuisance directions and hurts discrimination.

### 5.4 Why this yields both goals, without collapse

`T` + hard-negative InfoNCE ⇒ interface **correspondence** (goal b). Conformer-augmented queries ⇒ that
correspondence is **conformer-invariant** (goal a). Contrastive negatives ⇒ no collapse; holo→holo stays the
do-no-harm floor. Bilinear `T` ⇒ deployment retrieval stays inner-product-searchable.

**Protein-level aggregation.** Atom-pair scores give the interface correspondence map directly (docking /
interface prediction). For retrieval, aggregate to a protein score via **optimal-transport / soft-alignment**
over the two interface atom sets (respects the one-to-one nature of contacts) rather than a naive sum.

## 6. Data strategy — decouple descriptor pretraining from invariance

Two objectives with very different data needs; do **not** train them jointly on the small paired set.

- **Stage A — descriptor from scratch, holo-only, FULL MaSIF train set (~4,943 complexes).** The
  correspondence objective (§5.1–5.2) needs *no* conformers — just holo interfaces. This is where `E` + `T`
  earn the ~0.90 ceiling and where data is abundant. Decoupling is about **data abundance, not cost.**
- **Stage B — invariance via predicted-structure conformers.** Now that compute is not the constraint (the
  CHF-100 ceiling is a per-session guardrail, not the project budget), **train on the deployment
  distribution**: AF3 / Chai-1 / Boltz **diffusion samples** (many seeds × samples per protein — that spread
  *is* the invariance signal). ESMFold is fast but single-deterministic (little per-protein diversity) → use
  for breadth, not the ensemble. FASPR sidechain jitter is optional extra regularization only. Scale AF3-family
  conformer generation across the training set (Kuma H100 inference; Jed MSA), reusing the Phase-3 pipeline.
- **Structural-mismatch filtering (training hygiene only).** Remove 1A2W-type non-binding-conformation cases
  from training *positives* using the Sc→interface-atom + retention/local-RMSD detector. **This is not a
  deployment filter** (both thresholds need the holo complex; see `07-phase3-results.md` recommendation) — it
  cleans the training signal and is out-of-scope for the objective at query time.

## 7. Evaluation (reuse Phase-3 M1, add per-sample spread)

- **Primary:** AF3-query descriptor-separation AUC (both directions, randneg + cross-complex) vs the ~0.90
  holo ceiling, on **held-out** complexes (complex-level split; eval set disjoint from all training), on the
  **induced-fit-only** stratum. Report **per-sample AUC for each of the 5 AF3 samples + the spread** — the
  literal sharpened target.
- **Secondary:** top-k retrieval against a holo database (the deployment-shaped test), holo vs AF3 query.
- **Controls (ml-research-guardrails, non-negotiable):** shuffled-label ≈ 0.5; complex-level holdout, no
  holo/AF3 leakage; per-complex spread reported; **holo→holo do-no-harm floor** every eval; the from-scratch
  descriptor must first **match MaSIF's ~0.90 on holo** (§8 M1 gate) before invariance claims are credible.

## 8. Milestones (cheapest-first, each gated)

- **M0 — heterogeneous graph builder + invariance-of-representation sanity (days, ~free, CPU).** Assemble
  atom nodes (reuse Phase-2/3 builders), vertex nodes (raw MaSIF channels), vertex↔vertex mesh edges,
  vertex↔atom edges (cutoff/top-k first). **Gate:** rotate a structure → all edge features and the embedding
  are invariant (the make-or-break correctness check); shapes/scale sane on 1–2 complexes; vertex count
  capped.
- **M1 — can a from-scratch GNN re-earn the descriptor? (THE feasibility gate).** Train Stage A
  (correspondence + hard negatives, holo-only), a few hundred complexes → scale toward full. **Gate: held-out
  holo→holo AUC approaches MaSIF's ~0.90.** Run the vertex-feature and vertex↔atom-edge ablations here (raw
  vs mesh-MP-learned; cutoff vs geodesic). *If a from-scratch GNN cannot match MaSIF on holo, that is the
  finding — stop before invariance and diagnose.* Estimate Stage-A GPU cost explicitly here before full
  scale-up.
- **M2 — add invariance; eval on real AF3 (THE objective gate).** Turn on conformer-augmented queries (§5.3)
  with predicted-structure conformers; mismatch-filter training positives. **Gate:** AF3→holo closes a
  meaningful fraction of +0.069 vs the M3 +0.016 bar, per-sample spread shrinks, holo→holo preserved.
- **M3 — scale + geodesic ablation + optional aligner/scorer.** Only if M2 passes: scale conformers/data,
  test geodesic vertex↔atom edges, and (deferred) an equivariant pose scorer over matched contacts (D5/D7).

## 9. Risks & traps (carry forward)

1. **Overfitting a high-capacity from-scratch model** — the dominant risk. Mitigation: Stage-A pretraining on
   the full holo set before invariance; strong hard-negative contrastive signal; complex-level holdout.
2. **Can it even match MaSIF on holo?** — the M1 gate is an honest kill-switch. A from-scratch GNN matching a
   net trained on far more data/engineering is not guaranteed.
3. **Embedding collapse under invariance** — mitigated structurally by baking invariance into the
   contrastive task (§5.3) rather than a standalone penalty; keep any explicit consistency term small.
4. **SE(3)-invariance leaks** — any raw-coordinate feature lets the model learn pose, not chemistry, and
   fights invariance. M0 rotation test is the guard.
5. **Does learned invariance transfer to *real* AF3?** — training on AF3-family conformers (not FASPR) buys
   train/test distribution match; still, evaluate on real AF3 (M2), never only on the training conformer type.
6. **Vertex-count blowup** — coarsen/subsample the mesh; profile in M0.

## 10. Explicitly NOT in Phase 4

- The unaddressable ~23% structural-mismatch monomers (physically absent binding conformation — correct to
  treat as non-binders; no query-side fix exists).
- A deployment-time mismatch filter (impossible without the holo complex).
- Ligands / standalone entities (D9), learned global aligner hardening (D5-B) beyond the optional M3 scorer.
- True apo/AF2 (as opposed to AF3) as the training/eval regime — AF3 is the committed proxy for this phase;
  real-apo validation remains a separate future check.

## 11. Module layout & immediate next actions

- `src/masif_graph/graph/hetero.py` — heterogeneous graph builder (atom + vertex nodes, three edge types;
  reuse Phase-2 atom-graph + Phase-3 surface loaders).
- `src/masif_graph/p4/encoder.py` — the heterogeneous GNN encoder `E`.
- `src/masif_graph/p4/objective.py` — bilinear `T`, InfoNCE + hard-negative sampler, conformer-augmented
  contrastive loss, optional consistency term.
- `src/masif_graph/p4/train.py` — Stage-A / Stage-B trainer; reuse Phase-3 relabel/identity-mapping + M1 eval.
- `scripts/p4_*.sh`, Kuma conductor for conformer generation at scale.

**First actions:** (1) build M0 graph + the rotation-invariance sanity test; (2) wire the Stage-A
correspondence loss and confirm it *trains* on a handful of holo complexes; (3) estimate Stage-A full-set GPU
cost. Do **not** start Stage-B conformer generation at scale until the M1 holo≈0.90 gate is met.

---

**One-line honest framing for the PI:** Phases 1–3 proved the frozen descriptor is a strong ceiling that
head-only tricks and late-fused graphs can't breach; Phase 4 is the earned escalation — unfreeze the whole
representation as a from-scratch heterogeneous GNN and learn conformer-invariance + interface correspondence
jointly — gated first on the honest question *can it even match MaSIF on holo?*
