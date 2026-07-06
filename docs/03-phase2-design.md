# Phase 2 — Conformational robustness via an atom graph (closing the holo→apo gap)

> **Status:** Design (revised 2026-07-03 around the holo→apo objective).
> **Prerequisite reads:** [`00-context-and-goals.md`](00-context-and-goals.md) (decisions D1–D10),
> [`01-phase1-design.md`](01-phase1-design.md), [`02-phase1-results.md`](02-phase1-results.md).
> **Purpose of Phase 2 (the question):** MaSIF's learned surface descriptor is implicitly tuned to
> **bound-state (holo) sidechain rotamers** — crystal sidechains placed *with knowledge of the
> partner*. On **apo / unbound / AF2** structures the surface rotamers are wrong and MaSIF produces
> false positives & negatives, **despite a high holo-benchmark AUC**. Does a graph encoding
> **atom connectivity and bond rotatability** — *how sidechain atoms can move* — fused with the
> frozen surface descriptor, make the representation **robust to sidechain conformation** and close
> the **holo→apo gap**, *without harming holo performance*?

---

## 0. The central reframing (read this first)

Phase 1 measured a small pooling cost on the holo test set. But the **project's real objective is
holo→apo robustness** (see README / CLAUDE.md north star). This has a sharp consequence for how we
evaluate Phase 2:

> **Our current validation set cannot see the benefit we are building.** `testing.txt` is holo PDB
> with perfect, partner-aware sidechains — the same distribution MaSIF was tuned on. A graph that
> adds *conformational robustness* will show ~zero gain there (nothing to fix), and could even look
> slightly worse from added parameters. **Optimizing or gating on holo AUC alone would wrongly
> reject the method.**

So Phase 2 does two things at once: build the graph, **and** build an evaluation that can actually
detect conformational robustness (§4–§5). This pulls the "relaxed/ensemble" robustness idea (a
Phase-4 item in `00 §6`) forward into Phase 2's evaluation core.

---

## 1. Strategy: add orthogonal signal (connectivity + rotatability) before re-engineering the surface

Still **Path B first** (`00 §5` D1): keep the Phase-1 pooled descriptor **frozen** and add a graph,
rather than retraining an atom-centric surface encoder (D1-B, held as contingency). But the *point*
of the graph is now specific: encode the **degrees of freedom** — which bonds are rotatable, hence
which surface atoms are conformationally (un)reliable — so the model can be robust to the rotamer
state that the surface descriptor is over-sensitive to.

**Why not retrain the surface net first (D1-B):** it would only tighten holo descriptors; it does
nothing for conformational robustness and doesn't test the thesis. D1-B stays the contingency if the
graph cannot close the gap.

---

## 2. Decisions resolved for Phase 2 (provisional defaults — revisit if the ablation says so)

| # | Decision | Phase-2 default | Rationale |
|---|---|---|---|
| **D6** | Retrain surface net? | **No — keep frozen Phase-1 pooled descriptors as fixed input.** | Isolates the added-signal effect; on a repacked structure the frozen *net* is simply re-run on the new surface, and the graph learns to correct. |
| **D3** | Fusion semantics | **D3-A: unified learned complementarity** — train the fused projection contrastively so contacting atom pairs are near. | The graph must interact with the surface channels, not sit beside them. |
| **D2** | Rotation invariance | **Invariant graph** (distances via RBF, bond types, rotatable flags — all invariant). | Whatever feeds the matched descriptor must be rotation-invariant or matching breaks. Equivariant channels deferred to the Phase-3 scorer. |
| **D4** | Contact / positive supervision | **Reuse Phase-1 sc-filtered vertex-contact positives mapped to atoms**, defined on the **holo backbone**, held fixed across perturbations (§4). | Fixed-backbone repack keeps contacts well-defined; positives don't move when sidechains do. |
| **Graph** | Two embeddings (A/B) vs one graph | **One heterogeneous (multi-relational) graph** with typed covalent + spatial edges (§3). | The signal — *rotatable bond → atom can move → surface unreliable* — couples topology and geometry; they must message-pass **jointly**. Two independent embeddings fuse too late. Mainstream (GearNet-style; Graphein). |
| **D9** | Entity abstraction | **Molecule-agnostic atom graph**, wire proteins only. | Cheap now, expensive to retrofit; ligands stay Phase 4. |

**PI decisions:** **§8-A superseded** by this reframing (holo AUC → do-no-harm floor; robustness is
the gate). **§8-B representation-only** and **§8-C descriptor-sep + small retrieval** still hold.
**§8-D GPU/budget go-ahead** still pending (needed at M2).

---

## 3. The representation: one heterogeneous atom graph fused with the frozen descriptor

```
  fused(a) = Proj( surface(a) ⊕ RelationalMP(G)[a] )     # unified, rotation-invariant, contrastive
```

**Graph G (per chain, molecule-agnostic):**
- **Nodes** = atoms (surface + sub-surface): element, formal charge, hybridization, aromaticity,
  H-count, degree. Surface nodes additionally carry the **frozen Phase-1 pooled 80-D descriptor**
  (straight/flipped) as a node feature; readout is at surface nodes only.
- **Typed edges (the load-bearing design):**
  - **covalent** — bond order **+ an explicit rotatable/rigid flag** (rotatable single bond vs
    ring / double / conjugated / peptide-ω). **This is the "how atoms can move" signal.**
  - **spatial** — radius-graph / kNN edges with RBF-expanded interatomic distance (the local
    geometry the mesh smooths).
- **Relational message passing** (edge-type-conditioned; e.g. R-GCN/GearNet-style), invariant by
  construction (features are distances, bond types, flags).

**Fusion + objective:** concatenate surface ⊕ graph readout → small MLP head → **contrastive margin
loss** on the Phase-1 atom pairs (D3-A), so complementary contacting atoms are near. Trained
**with rotamer-perturbation augmentation** (§4) so the embedding *learns* conformational robustness,
not just represents flexibility.

Ablatable by **edge type / feature**, not separate embeddings (§5).

---

## 4. Data & milestones

### M0 — atom graphs, enlarged probe, and apo-like structures (CPU)
1. **Atom graphs** from the same PDBs (reuse Phase-1 heavy-atom parse); perceive bonds with **RDKit**
   (primary; in-env, in-lineage) with **biotite** template connectivity as the protein fallback;
   rotatable-bond flags from RDKit. PyG for the graph tensors.
2. **Enlarge the holo probe** to N ≈ 150–300 so ≥50 complexes carry ≥10 sc-positives (kills the
   Phase-1 small-N caveat); re-confirm frozen baselines there.
3. **Apo-like structures via fixed-backbone repack (the controlled perturbation).** *Do **not** use
   `PDB_to_AF2.py` as-is:* AF2 backbones deviate drastically from the PDB, so contacts/positives
   become ill-defined. Instead, generate apo-like surfaces by **repacking sidechains on the fixed
   holo backbone, in the unbound (monomer) context** — split the complex, repack each chain **in
   isolation** (no partner context, so interface rotamers are not partner-tuned), keep the backbone
   fixed. Then re-run the reference surface+descriptor pipeline on the repacked chains. **Because the
   backbone is unchanged, contacts/positives stay defined by the holo geometry** — only the sidechain
   surface moves.
   - **Tool: start with the easiest — FASPR** (single fast binary, fixed-backbone repack). PyRosetta
     `pack`/Rosetta `fixbb` (Rosetta is in the reference lineage) is the more controllable alt.
   - **Start realistic** (one repack pass, apo-like); **stress-test harsher / alternative apo
     generators later** (multiple rotamer draws, aggressive blind repack, eventually true apo/AF2 with
     careful positive handling). The repack-aggressiveness knob is a PI dial (§8-A note).

### M1 — build the graph + fusion; validate the pipeline (CPU small-scale)
Implement `graph/` (heterogeneous graph builder, relational MP, fusion head) + the contrastive
trainer with rotamer augmentation. **Validate on CPU before any GPU:** overfit a few complexes;
**shuffled-label control → ~0.5**; **rotation-invariance unit test** (rotate a chain → embedding
unchanged). No scientific claims yet.

### M2 — the robustness ablation (THE GATE; scale on Kuma GPU)
Train the ablation cells (§5) on the training list with rotamer augmentation; evaluate on the
enlarged held-out probe **at both holo and repacked (apo-like) states**. GPU + conductor-agent +
CHF-100 budget (§6). Report per-complex spread, ≥3 seeds, shuffled control.

### Deliverable
**`docs/04-phase2-results.md`**: the holo do-no-harm numbers, the **robustness (holo→repacked)
differential** per ablation, which edge features carry the robustness, the **go/no-go for Phase 3**,
and the **D1-B trigger decision**.

---

## 5. The gate: do-no-harm on holo + robustness under perturbation

Holo AUC is **necessary but not the objective**. The decisive signal is *differential degradation*
under the apo-like perturbation.

| variant | surface (frozen) | +covalent (bond order) | +rotatability flag | +spatial (RBF dist) | probes |
|---|:---:|:---:|:---:|:---:|---|
| surface-only (atom) | ✓ | | | | holo baseline **and** its collapse under repack |
| +covalent | ✓ | ✓ | | | topology alone |
| +rotatability | ✓ | ✓ | ✓ | | does *flexibility* awareness help robustness? |
| +spatial | ✓ | | | ✓ | geometry alone |
| **full** | ✓ | ✓ | ✓ | ✓ | **the decision** |

Every cell evaluated on **holo** and **repacked** structures, identical complexes/pairs/seeds.

**Gate (revised, supersedes §8-A "recover baseline"):**
- **Do-no-harm floor:** full-model **holo** AUC ≥ the Phase-1 atom mean-pool baseline (graph must not
  break holo).
- **Decisive:** under the apo-like repack, **full-model AUC / native-contact recovery degrades
  significantly less than surface-only**, with an **attributable** contribution from the
  rotatability/spatial edges. That robustness gain = **GO to Phase 3**.
- **Trigger D1-B** if the graph cannot improve robustness and the ablation localizes the ceiling to
  the frozen descriptor.

---

## 6. Compute & budget plan (CPU-validate → Kuma-scale; conductor agent)

- **CPU (Jed, now):** M0 (graphs, repack via FASPR, enlarged probe) + M1 pipeline validation. Small
  GNN/MLP on precomputed descriptors → a first pass is even CPU-feasible.
- **GPU (Kuma, gated):** M2 training + seed sweep — modest (a few GPU-hours/cell). Run the agent as a
  **conductor** (small Jed orchestrator that `ssh`-submits + monitors Kuma jobs), staging
  **Jed → `/work/upthomae/Meng/JED_TO_KUMA` → rsync → Kuma `/scratch`**, spending the **CHF 100**
  budget (cost-check each job, log cumulative, right-size). See the `slurm-claude-agent` and
  `connect-to-kuma` skills.
- **Gate:** no GPU launches without explicit human go-ahead (§8-D).

---

## 7. Explicitly NOT in Phase 2

- Atom-centric surface retrain (**D1-B**) — contingency only.
- Learned **pose scorer** (D7) / learned **aligner** (D5-B) — Phase 3 (per §8-B).
- True **apo/AF2** training+eval with backbone variability, standalone **ligands**, liganded data —
  later / Phase 4 (Phase 2 uses fixed-backbone repack as the controlled apo proxy).
- Attention/learned pooling of the surface descriptor — only if D1-B triggers.

---

## 8. Decisions for the PI

- **8-A — Gate. ✅ REVISED (2026-07-03):** holo AUC = **do-no-harm floor**; the **holo→repacked
  robustness differential** is the decisive gate (supersedes the earlier "recover baseline on holo").
  *Open dial:* how aggressive the repack is — **default = realistic fixed-backbone monomer repack**;
  harsher/alternative apo generators stress-tested later. PI may steer the starting severity.
- **8-B — Scope. ✅ representation-only** (pose scorer D7 → Phase 3).
- **8-C — Eval depth. ✅ descriptor-sep AUC (holo + repacked) + small top-k retrieval**; full
  binder-recovery and true apo/AF2 → Phase 3.
- **8-D — GPU/budget go-ahead. ⏳ PENDING (needed at M2).** M0/M1 are CPU. No GPU jobs until go-ahead.

---

## 9. Module layout, risks, immediate next actions

**New modules:** `src/masif_graph/graph/` (heterogeneous graph builder; relational MP; fusion),
`src/masif_graph/perturb/` (fixed-backbone monomer repack; apo-like surface regeneration),
`src/masif_graph/train/` (contrastive trainer + rotamer augmentation + ablation harness),
`src/masif_graph/score/` (stub; Phase 3). Reuse Phase-1 `io/ surface/ pairs/ metrics/ align/`.

**Risks:** **T4** redundant chemistry (the +covalent/+rotatability ablation is the test — a real
result either way); **T2** broken invariance (D2 + M1 rotation test); repack realism (FASPR fixed-
backbone monomer repack is a *proxy* for apo — validate against true apo later, hence the stress-test
plan); **compute** kept modest by freezing the surface net.

**Immediate next actions (CPU, non-autonomous, gated on §8-D for GPU):**
1. M0: atom graphs (RDKit + biotite) + enlarge probe + **FASPR fixed-backbone monomer repack**;
   re-confirm holo baselines and quantify surface-only's collapse under repack (the effect we aim to
   reduce).
2. M1: heterogeneous graph + fusion + contrastive trainer w/ rotamer augmentation; CPU validation.
3. On go-ahead: M2 robustness ablation on Kuma via the conductor agent → `docs/04-phase2-results.md`.
