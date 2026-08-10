# Guideline — PyMOL visualisation of a training pair (all phases)

> **Standing requirement.** Whenever the user asks for a visualisation of the network's input, it
> must follow this document. The reference implementation is Phase 7:
> `scripts/p7_export_pair_viz.py` (exporter) + `scripts/p7_pair_pymol.py` (viewer).
> The rules below were written after getting several of them wrong; §8 lists the specific mistakes.

## 1. Purpose

The visualisation exists to answer one question: **is what the GNN actually eats what we think it
eats?** It is a debugging instrument for the model input, not a figure generator. Everything follows
from that: if a number reaches the network, it must be visible; if it does not reach the network,
the viewer must not imply that it does.

## 2. The contract — every visualisation must satisfy all of these

1. **One `.npz` per training pair.** A training example is a pair of interacting partners, so the
   user loads *one* file and gets *everything*. Never make them assemble a pair from two files.
2. **Both partners, in full.** No cropping, no "pocket only", no subsampling. The GNN consumes the
   entire graph on both sides, so the viewer shows the entire graph on both sides.
3. **Every feature the GNN consumes gets an object.** Node features *and* edge features. If a
   tensor is concatenated into a message or an embedding anywhere in the forward pass, it is
   visible. §4 gives the procedure for enumerating them.
4. **`_left` / `_right` split**, independently toggleable, for every feature.
5. **The training positives are drawn** (`contacts`), so the user can see the supervision signal,
   not just the input.
6. **A companion `.pdb` holding exactly the graph's atom nodes, in graph-node order** — chain A =
   left, chain L (ligand) or B (protein) = right. Atom *i* of that chain *is* node *i*.
7. **Read from the artefact the model reads.** Pull arrays out of the training `.npz`, never
   re-derive them. Anything the training npz genuinely lacks (e.g. vertex coordinates and faces,
   which live in the `.ply`) is fetched separately and *labelled as such*.
8. **Self-contained output folder.** The user downloads one directory and runs it on their own
   machine: exports + the viewer script + a `README.md`. No cluster paths at view time.

## 3. Naming

```
{carrier}_{feature}_{side}
```

* **`carrier`** names *where the value lives in the graph*, never how it is drawn:
  `atom_` (atom-node feature), `vert_` (vertex-node feature), `edges_` (edge feature).
  Reserve `surf_` for the case where a model genuinely consumes a face/surface-level quantity — in
  the current architecture it does not, so nothing is called `surf_*` (see §5.1).
* **`feature`** is the feature's own name, not the carrier's: `edges_aa_bondorder`, not `edges_aa`.
  If one edge type carries several features, each gets its own object.
* **`side`** is `left` or `right`.

Non-feature objects keep bare names: `structure`, `contacts`.

## 4. Enumerating what the GNN consumes (do this every time the model changes)

Read the encoder's `forward` and the dataset builder, and list every tensor that enters. For the
current architecture (`p4/encoder.py`, `p4/dataset.py`):

| carrier | tensor | dim | objects |
|---|---|---|---|
| atom nodes | `atom_feat` | 26 | `atom_element`, `atom_hybridization` (categorical) + one per scalar feature |
| vertex nodes | `vert_feat` | 4 | `vert_si`, `vert_hbond`, `vert_charge`, `vert_hphob` |
| atom–atom edges | `aa_feat` | 5 | `edges_aa_bondorder` (4-way one-hot) + `edges_aa_rotatable` (flag) |
| vertex–vertex edges | `vv_feat` | 9 | `edges_vv_dist` (RBF'd before the MLP) + `edges_vv_cos` |
| vertex–atom edges | `va_feat` | 9 | `edges_va_dist` + `edges_va_cos` |
| supervision | `pos` | — | `contacts` |

**Collapse one-hots into one categorical object** (colour = category) rather than N binary objects.
**Split multi-feature edge types into one object per feature.** RBF expansions are visualised on the
underlying scalar, with the expansion noted in the README.

When the architecture changes, this table is the deliverable that must change with it. A viewer
that silently lags the model is worse than none, because it looks authoritative.

## 5. Rendering conventions

### 5.1 Name by carrier, render for legibility
Vertex features are drawn as a **shaded surface** because that reads far better than a point cloud —
but they are still called `vert_*`, because the value lives on a vertex node. Say so explicitly in
the README and the load-time printout: in this architecture the triangulated surface is **not** a
model input (the training npz contains no faces at all); the mesh reaches the network only as
`vv`-edge connectivity plus the edge scalars and `si` derived from its geometry.

### 5.2 Shared colour scale across sides
Compute each feature's colour range over **both partners together** and apply it to both. Scaling
each side to itself makes two different distributions look identical — which is exactly the thing a
shared-encoder model needs you to be able to see. Ramp: **blue → white → red**, symmetric about zero
for signed features, so white always means zero. Binary features therefore read **blue = 0, red = 1**.
Categorical features get a discrete palette, documented in the README.

### 5.3 Object order: feature-major
Create objects so `<feature>_left` and `<feature>_right` are **adjacent** in the panel:

```
vert_si_left     vert_si_right
vert_hbond_left  vert_hbond_right
...
```

Not all-left-then-all-right — comparing partners is the main use of the viewer.

### 5.4 Build everything; enable a little
Every object is **created** on load so it is present in the panel. Only a readable default subset is
**enabled** (`structure`, `vert_si_left/right`, `contacts`). Do not gate object *creation* behind a
flag: an object the user cannot see in the panel does not exist to them. Measure before assuming a
layer is too expensive to build (§8.3).

### 5.5 Session defaults
Leave the background at PyMOL's default (black); do not override it. Show `structure` as **lines,
coloured by chain** (`util.cbc`, so left and right differ) **and by element** (`util.cnc`). Pick
node/edge colours that survive a black background.

## 6. The export contract

The `.npz` carries, per side, with a `left_` / `right_` prefix:

```
atom_xyz, atom_feat, atom_elem
vert_xyz, vert_normal, vert_feat, faces
aa_edge, aa_order, aa_rot
vv_edge, vv_dist, vv_cos
va_edge, va_dist, va_cos
surf_node_idx
```

plus `contacts_atom` (positives as atom indices) and a `meta` JSON holding the feature-name lists,
the side labels, and per-side counts. **The feature names ship inside the npz**, so the viewer never
hard-codes an ordering that can drift from the model.

**Alignment hazard.** Edge feature arrays are aligned column-for-column with their edge list. Do not
re-`unique` or re-sort an edge list that has feature arrays attached — take the stored canonical
list as-is and assert the lengths match. Where the npz stores *directed* edges with per-directed-edge
features (`aa`), map back to the undirected list explicitly.

## 7. Verification (PyMOL is usually not installed on the cluster)

Assume you cannot launch PyMOL. Before handing over, verify offline and say what you did *not* test:

1. **Syntax** — `ast.parse` both scripts.
2. **Key presence** — every array the viewer reads exists in the npz.
3. **Index bounds** — all face, edge, `surf_node_idx` and contact indices are in range for their
   target arrays.
4. **Feature/edge alignment** — `len(vv_dist) == len(vv_edge)`, etc.
5. **PDB ↔ graph correspondence** — per-side atom counts match, and coordinates agree to <0.01 Å,
   proving PDB order equals graph-node order.
6. **Cost** — time the object construction on the largest example.
7. State plainly that the rendering itself is untested.

## 8. Anti-patterns (all of these actually happened in Phase 7)

1. **Cropping to the interesting region.** The first version showed a 12 Å pocket instead of the
   protein. The GNN sees the whole chain; so must the viewer.
2. **Showing one partner.** A training example is a pair. Both sides, always.
3. **Hiding objects behind a build flag.** The dense `vv`/`va` layers were only created with
   `dense=1`, so 8 of 12 edge objects were invisible. Building all of them costs **0.20 s** for
   ~140k lines — the deferral saved nothing and hid the data. Measure, don't assume.
4. **Rendering a per-edge feature as a filtered subset.** `edges_aa_rot` drew only the rotatable
   bonds, which presents a *feature* as if it were a *filter*. `aa_rot` is a 0/1 value on every
   covalent edge, so draw every edge and colour by the value.
5. **Naming an object after the carrier instead of the feature.** `edges_aa` was ambiguous once the
   edge type carried two different features; it is `edges_aa_bondorder`.
6. **Per-side colour auto-scaling** while claiming a shared scale. Either share the scale or say you
   don't; silently normalising each side hides real differences between partners.
7. **Duplicating one value as two objects** (`surf_si_*` *and* `vert_si_*`). One value, one object,
   named for where it lives.
8. **Documentation drifting from behaviour.** The README claimed binary features read grey/red when
   the code produced blue/red. Re-read the code when writing the README.
9. **Grouping all-left-then-all-right** in the panel, which makes side-by-side comparison tedious.

## 9. Deliverable checklist

- [ ] One `.npz` + one `.pdb` per training pair, both partners complete
- [ ] Every GNN-consumed tensor has an object (§4 table regenerated for the current model)
- [ ] `{carrier}_{feature}_{side}` naming throughout
- [ ] Feature-major object order, left/right adjacent
- [ ] Colour ranges shared across sides; ramp and palettes documented
- [ ] `contacts` (training positives) drawn
- [ ] `.pdb` = exactly the graph's atom nodes, in node order — *verified numerically*
- [ ] All objects built; sensible default subset enabled
- [ ] Default background; `structure` as chain/element-coloured lines
- [ ] Self-contained folder: exports + viewer + README
- [ ] README documents the colour scheme, what is and is not a model input, and what to look for
- [ ] Offline verification done (§7) and the untested parts stated
