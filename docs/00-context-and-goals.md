# MaSIF-graph — Project Context & Goals

> **Status:** Planning / north-star document. Created 2026-07-02.
> **Nature:** Ground-up rewrite. The existing `masif-neosurf` repo is a **reference/template only** — we do not extend its code in place. The **only** artifact reused verbatim is the **list of PDB IDs** used for training/testing.
> **Audience:** Anyone building or reviewing MaSIF-graph. Read this before any phase-specific design doc.

---

## 1. One-paragraph summary

MaSIF-graph is a redesign of MaSIF-PPI-search that makes the **surface heavy atom** — not the surface mesh vertex — the fundamental unit of representation, retrieval, and alignment. Each surface atom carries a fused embedding built from (a) the classical MaSIF learned **surface fingerprint**, (b) a **chemical/bond-graph** embedding, and (c) a **rotation-invariant local-geometry** embedding. Binder discovery moves from **per-patch local docking** to **global protein–protein pose optimization**: find the rigid pose of a candidate binder that best co-locates all target/binder atom pairs whose embeddings indicate complementarity. v1 is benchmarked on PDB structures using the reused training/testing lists; v2 optionally adds relaxed side-chain ensembles and standalone-ligand entities.

---

## 2. Motivation & hypothesis

MaSIF's strength is a **pose-invariant, complementarity-aware surface fingerprint** that lets geometrically/chemically matching surfaces be found by nearest-neighbor search in descriptor space. Its known limits:

- The mesh-vertex representation is **large** (thousands–tens of thousands of vertices/chain) and **geometry-first**; chemistry enters only as three smoothed scalar channels (H-bond, electrostatics, hydrophobicity).
- Alignment is **per-patch**: one target site vertex is matched and RANSAC/ICP-aligned to one candidate patch at a time; the global consistency of the whole interface is only checked afterward (clash filter).
- The representation has no first-class notion of **atoms, bonds, or ligands as molecules**, which matters for neosurfaces (protein–ligand composite surfaces) and for robustness to side-chain movement.

**Central hypothesis:** Representing the surface as a graph of surface atoms, each enriched with explicit bonded-chemistry and local-geometry embeddings, will (1) shrink the problem to hundreds of entities per chain (making *global* alignment tractable and more physically meaningful), (2) add signal orthogonal to the smoothed surface channels, and (3) provide a natural, unified home for ligands and conformational ensembles.

**This hypothesis is not assumed — it must be earned.** Pooling surface geometry onto atoms *loses* sub-Ångström resolution, which is MaSIF's core asset. The graph embeddings must buy that back and then some. Phase 1 exists to measure the cost before we spend on the benefit (see §7, §8).

---

## 3. Reference system (MaSIF-neosurf) — condensed spec to diverge from

We keep this here so the rewrite has a precise target to reproduce-then-improve. All facts below are confirmed from the reference code.

### 3.1 Preprocessing (`preprocess_pdb.py`)
- MSMS solvent-excluded surface (probe 1.5 Å) → `fix_mesh` remesh to ~1.0 Å vertices.
- **5 per-vertex input features**, order `[shape_index, distance-dependent curvature (DDC), H-bond, APBS electrostatics, hydrophobicity]`.
  - `shape_index`, `DDC`: geometric, recomputed at precompute time (not stored in `.ply`).
  - `H-bond`, `electrostatics`, `hydrophobicity`: chemical, stored in `.ply`.
- Geodesic polar coordinates (ρ = Dijkstra geodesic, θ = MDS-flattened angle) per patch.
- **Vertex→atom identity already exists but is transient:** MSMS tags each raw vertex with its owning atom (`read_msms.py` → `names1`), used to assign per-atom chemistry, then *dropped* after remeshing (features re-mapped by KDTree nearest-neighbor). There is no persisted vertex→atom index.
- Ligands: geometry via extra MSMS spheres (`keep_hetatms`); chemistry via an RDKit bond graph (`ligand_utils.py`) + bond-corrected mol2 for APBS.

### 3.2 Descriptor network (`MaSIF_ppi_search.py`)
- MoNet-style **geodesic CNN**; input is a patch (≤200 vertices, 12 Å geodesic radius) of 5 features + (ρ, θ).
- Learnable Gaussian soft-binning into an **80-bin polar grid (16 θ × 5 ρ)**, per feature channel, max-pooled over **16 rotations** → rotation invariance.
- Output: **one 80-D descriptor per vertex**.
- **Complementarity "flip trick":** binder-side features negated (all but hydrophobicity), θ→2π−θ, so *complementary* surfaces (bump↔pocket, +↔−, donor↔acceptor) map to *nearby* descriptors.
- Loss: contrastive margin (positive pair distance → 0; negative pair distance → margin 10) over a 4-chunk batch `[binder, pos, neg, neg_2]`.
- Each protein saved with **two** descriptors per vertex: `desc_straight` and `desc_flipped`.

### 3.3 Matching (`alignment_utils.py::match_descriptors`)
- Pick target site vertex (highest mean MaSIF-site interface score in its patch).
- Threshold DB vertices: descriptor L2 distance < `desc_dist_cutoff` (≈2.0) AND interface score > `iface_cutoff` (≈0.75).

### 3.4 Alignment + scoring (per-patch)
- `registration_ransac_based_on_feature_matching` with 80-D descriptors as the matching feature (`ransac_n=3`, radius 1.5 Å, edge/distance/normal checkers) → **point-to-plane ICP** refine.
- Transform applied to the whole binder PDB → **clash filter** (CA + heavy-atom KDTree).
- **AlignmentEvaluationNN** (PointNet: per-point Conv1D MLP → global average pool → softmax): scores P(correct) from **4 per-point scalars** `[1/dist, 1/desc_dist, normal·normal, 1/vertex-to-atom-dist]`. Label at train time = interface-RMSD < threshold vs decoys.

### 3.5 Training scale / metrics
- Lists: **4,943** train pairs, **959** test pairs (the artifact we reuse).
- Descriptor-net retraining is the *active* work in the reference fork (despite README claiming otherwise): configurable negative sampling (`neg_ratio`, `neg_mix` = cross/within/hard split, `neg_loss_weight`); best = cross-complex 1:1, **test ROC-AUC ≈ 0.988** (descriptor-separation metric). ~27 h on 1 GPU, SLURM-sharded caching.
- Stack is **TensorFlow 1.13** (legacy).

---

## 4. Target architecture — MaSIF-graph

### 4.1 Entities
- **Surface heavy atom** = a heavy atom that is the nearest heavy atom to ≥1 surface vertex. (Definition is a KDTree query from surface points to heavy atoms.)
- Each surface atom is a node. The rewrite operates on **sets of surface atoms**, not meshes, downstream of embedding.

### 4.2 Per-atom embedding = concatenation/fusion of three parts
1. **Surface fingerprint (80-D, from MaSIF surface signal).** Either pooled from per-vertex descriptors of the atom's owned vertices, or produced by an atom-centric surface readout. Encodes shape + smoothed chemistry, complementarity-aware.
2. **Graph embedding A — bonded chemistry.** Message passing over the covalent graph. Node features: element, valence, formal charge, electronegativity, hybridization, aromaticity, H-count. Edge features: bond order. Must include sub-surface bonded neighbors (chemistry of a surface atom depends on buried neighbors).
3. **Graph embedding B — local geometry.** Rotation-invariant encoding of distances (+ invariant angles) to nearby atoms (radial-basis / SchNet–DimeNet style, or GVP/e3nn scalar channels).

### 4.3 Alignment (global, replaces per-patch loop)
- Build **global candidate correspondences**: all target×binder surface-atom pairs whose fused-embedding distance indicates complementarity.
- Fit a rigid pose that co-locates those pairs: robust estimator (correspondence-based RANSAC + weighted Kabsch/Umeyama) → ICP refine. (This literally realizes "minimize distance over low-embedding-distance pairs, globally.")
- Score the pose with a learned **interface scorer** operating on the set of contacting atom pairs (evolution of AlignmentEvaluationNN; keep the permutation-invariant PointNet shape, extend features, optionally add interface message passing).

### 4.4 Framework
- Target **PyTorch** (+ PyG / e3nn for graph & equivariant parts). Do not carry TF1.

---

## 5. KEY DESIGN DECISIONS TO BE MADE LATER

These are the load-bearing forks. Each is stated as an open decision with options, trade-offs, and a **leaning** (not a commitment). Resolve each at the start of the phase that first depends on it.

### D1 — How to produce the per-atom 80-D surface embedding
- **Opt A (pooling):** keep a MaSIF-style per-vertex descriptor net, pool (mean/max/attention) over each atom's owned vertices.
- **Opt B (atom-centric readout):** re-center the geodesic/point patch on each surface atom and train a surface encoder to emit per-atom directly.
- Trade-off: A reuses the proven descriptor net and isolates the granularity change (cheap, testable first); B is more faithful but a larger build and couples with D6.
- **Leaning:** A for Phase 1 baseline; revisit B only if A underperforms per-vertex by more than a small margin.

### D2 — Rotation invariance of geometry embedding B  *(non-negotiable constraint, but the mechanism is a choice)*
- Options: invariant scalars only (distances + angles, SchNet/DimeNet/GemNet) **vs** equivariant features (GVP/e3nn) with invariant readout.
- Constraint: whatever feeds the **matched correspondence descriptor** must be rotation-invariant, or descriptor-distance matching breaks. Equivariant/vector channels, if used, may feed only the **pose scorer** (which sees both partners in a common frame).
- **Leaning:** invariant scalars for the descriptor; consider equivariant channels for the scorer later.

### D3 — Complementarity vs similarity semantics of the fused embedding
- The surface descriptor is trained so *opposite* shapes match (flip trick). Chemistry complementarity is different (donor↔acceptor, +↔−), not identity. Mixing a similarity-style chemistry vector into one L2 "distance" conflates two notions of "close."
- **Opt A (unified learned complementarity):** train the whole fused projection contrastively so true contacting atom pairs are near — subsumes the flip trick into learned complementarity.
- **Opt B (separate distances):** keep surface (flip-complementary) and chemistry (learned-complementary) as distinct distances combined with learned weights.
- **Leaning:** A (more principled); B acceptable as an interim.

### D4 — What "correspondence" and "positive/negative pair" mean at atom level
- Reference positives = shape-complementarity-filtered vertex pairs within 1.0 Å contact. New positives = **contacting surface-atom pairs** — but define contact how? (inter-atom distance, buried-surface overlap, or shared contact vertices?)
- `neg_mix` (cross/within/hard) ports directly; hard negatives (atoms just outside contact) become more meaningful at atom granularity.
- **Decision needed:** contact definition + positive-pair supervision source (reuse shape-complementarity labels? recompute at atom level?).

### D5 — Global aligner: rule-based vs learned
- **Opt A (rule-based):** correspondence-RANSAC + weighted Kabsch + ICP. Deterministic, proven, dimension-agnostic.
- **Opt B (learned):** differentiable registration (soft correspondence + weighted Kabsch, RGM/PREDATOR-style) or one-shot equivariant / diffusion docking (EquiDock, DiffDock-PP).
- **Leaning:** A as the v1 baseline and the bar any learned aligner must beat; B as a v2 research challenger.

### D6 — Do we retrain/replace the surface descriptor net, or reuse pretrained MaSIF descriptors?
- Reuse pretrained (TF1) descriptors as frozen inputs (fast start, but two frameworks) **vs** reimplement + retrain the surface encoder in PyTorch (clean, slower).
- Couples with D1/D3. **Leaning:** reuse frozen pretrained descriptors for Phase 1 pooling; port + retrain when we adopt unified contrastive training (D3-A).

### D7 — Scorer architecture
- Keep PointNet-style permutation-invariant set scorer over contacting atom pairs (extend feature channels) **vs** add message passing across the interface contact graph (context-aware) **vs** full learned pose energy.
- **Leaning:** extend the PointNet first; add interface message passing if it helps.

### D8 — Interface localization without MaSIF-site
- Reference uses MaSIF-site iface scores to pick target vertices and filter candidates. At atom level: reuse a site predictor, learn an atom-level interface propensity head, or drop site-gating and rely on global alignment.
- **Decision needed** (affects both matching efficiency and recall).

### D9 — Ligand & standalone-entity handling (v2, but design now)
- The reference chemistry path is already atom-level and works on a lone molecule; the surface/feature driver assumes a protein chain. Decide whether MaSIF-graph treats "an entity" as chain-agnostic from the start (protein chain, ligand, or complex all as atom sets) — cheaper to bake in than retrofit.
- **Leaning:** design the entity abstraction to be molecule-agnostic from Phase 1, even if only proteins are wired up initially.

### D10 — Data & preprocessing reuse boundary
- Reuse only the PDB **lists**. Re-run preprocessing from PDBs ourselves (surfaces, atoms, graphs) **vs** partially reuse reference `.ply`/precompute outputs to bootstrap. Note the electrostatics/APBS step is the throughput bottleneck and will dominate cost for any ensemble (v2).
- **Decision needed:** whether APBS-derived electrostatics stays a feature, and if so how it's carried onto atoms.

---

## 6. Scope & non-goals

**In scope (v1):**
- Per-atom surface + graph embeddings; global rule-based alignment; retrained matching + scoring; benchmarked on PDB using reused lists.

**In scope (v2, optional):**
- Relaxed / rotamer **ensembles** per candidate (robustness to side-chain movement) — pure data augmentation, gated by preprocessing throughput.
- **Standalone-ligand** entities and liganded-PDB training data (neosurface support).
- Learned aligner as a challenger to the rule-based baseline.

**Explicit non-goals (for now):**
- Sequence/MSA co-modeling, full flexible docking, binding-affinity regression, de-novo binder generation. MaSIF-graph is retrieval + rigid pose + pose scoring.

---

## 7. Phasing (high level — detailed per-phase docs come separately)

- **Phase 0 — Baseline & harness.** Reproduce reference per-vertex descriptor-separation AUC and binder-recovery on the reused lists. Freeze as the number every phase must beat.
- **Phase 1 — Per-atom reframing, no graph features.** Persistent vertex→atom map; surface-atom definition; per-atom surface embedding (D1-A); atom-level matching; **global** correspondence-RANSAC + Kabsch + ICP; retrained scorer. **Gate:** does per-atom + global hold vs per-vertex before adding chemistry?
- **Phase 2 — Add graph embeddings A & B.** Build both GNNs (D2, D3), fuse, retrain matching contrastively, extend scorer. **Ablate** surface-only vs +A vs +B vs full so gains are attributable.
- **Phase 3 — Aligner hardening.** Tune atom-scale RANSAC/ICP; optionally prototype learned aligner (D5-B) as a challenger.
- **Phase 4 — v2 robustness & ligands.** Ensembles; standalone-ligand entities; liganded training data.

Do not build Phases 2–4 until Phase 1's gate is met.

---

## 8. Evaluation strategy

Two-level, both anchored to the reused PDB lists:
1. **Descriptor-separation ROC-AUC** (fast inner loop): can fused-embedding distance separate true contacting atom pairs from decoys? Directly analogous to the reference metric (~0.988 baseline). Use for day-to-day iteration.
2. **Binder recovery / alignment AUC** (slow, decisive): top-k retrieval and pose quality (interface RMSD to native) on held-out complexes; reuse the reference computational-benchmark protocol.

Discipline: keep named ablation presets (as the reference does with `custom_params_*`) so every representation/aligner change is attributable. Expect Phase 1 to land at or slightly below baseline — the intended gain is in Phase 2; a small Phase-1 drop is not failure.

---

## 9. Risks & traps (carry these forward)

- **T1 — Pooling loses geometric resolution.** Sub-Ångström surface complementarity is MaSIF's core asset; averaging vertices onto atoms blurs it. Barely-exposed atoms get noisy fingerprints. *Mitigation:* Phase-1 gate measures this directly.
- **T2 — Broken invariance.** A frame-dependent geometry embedding destroys pose-invariant matching. *Mitigation:* D2 constraint.
- **T3 — Complementarity/similarity conflation.** Mixing similarity-chemistry into a complementarity descriptor corrupts "distance." *Mitigation:* D3.
- **T4 — Redundant chemistry.** Bond-graph features may re-encode what H-bond/electrostatics channels already carry. *Mitigation:* Phase-2 ablations.
- **T5 — Two-framework drag.** Straddling TF1 + PyTorch slows everything. *Mitigation:* D6 leaning toward a clean port.
- **T6 — Ensemble cost (v2).** APBS electrostatics is the preprocessing bottleneck; ×N conformers multiplies it. *Mitigation:* D10; consider cheaper electrostatics or dropping APBS for ensembles.

---

## 10. Reuse inventory (reference repo → new project)

| Reference artifact | Disposition |
|---|---|
| PDB training/testing **lists** (`lists/training.txt`, `testing.txt`, 4,943 / 959) | **Reuse verbatim** |
| MSMS surface + `read_msms` vertex→atom labeling | **Reference** (reimplement the atom-mapping cleanly + persist it) |
| 5 surface feature computations (SI, DDC, hbond, APBS, hphob) | **Reference** (reimplement; decide per-D10 which survive to atoms) |
| MoNet geodesic descriptor net (TF1) | **Reference** (reimplement/port per D1/D6) |
| Flip-trick complementarity | **Reference** (fold into learned complementarity per D3) |
| RANSAC/ICP alignment (Open3D) | **Reference** (reuse Open3D primitives; move per-patch → global correspondence) |
| AlignmentEvaluationNN (PointNet scorer) | **Reference** (reimplement + extend per D7) |
| `neg_mix` negative-sampling framework | **Reference** (port; redefine pairs at atom level per D4) |
| Ligand RDKit bond-graph chemistry | **Reference** (strong basis for graph embedding A + D9) |
| Computational-benchmark protocol | **Reference** (reuse for §8 metric 2) |
| Reference `.ply` / precompute `.npy` outputs | **Bootstrap only if D10 chooses partial reuse** |

---

## 11. Open questions for the team

1. Is the new codebase a fresh repository, or a new top-level package inside this one? (Affects where phase docs and code live.)
2. Compute budget & cluster for retraining + (v2) ensemble preprocessing?
3. Do we keep APBS electrostatics as a feature given its cost, or substitute a cheaper approximation? (D10)
4. Target for "success": beat baseline binder-recovery, or match it with added ligand/robustness capability? (Sets the bar for Phase-1 gate interpretation.)
5. Which resolved decisions (§5) can we lock now vs which stay open until their phase?

---

## 12. Glossary

- **Surface heavy atom** — heavy atom that is the nearest heavy atom to ≥1 surface vertex; the fundamental unit of MaSIF-graph.
- **Fingerprint / surface descriptor** — the 80-D MaSIF-learned surface embedding.
- **Graph embedding A / B** — bonded-chemistry MPNN embedding / rotation-invariant local-geometry embedding.
- **Complementarity** — the property that two partner surfaces (or atoms) match; in MaSIF encoded via the flip trick, in MaSIF-graph to be learned (D3).
- **Global alignment** — rigid pose of the binder optimizing co-location of all complementary atom pairs at once (vs per-patch).
- **Descriptor-separation AUC** — fast metric: how well embedding distance separates true contacting pairs from decoys.
- **neg_mix** — split of each positive's sampled negatives across cross-complex / within-complex / hard sources.
