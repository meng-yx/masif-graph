# MaSIF-graph training pairs — PyMOL viewer

Five randomly drawn PDBbind protein–ligand training pairs. A training example is always a **pair of
interacting partners**, so both sides are here in full — the entire protein (all atoms, all
vertices, all edges) and the entire ligand. Nothing is cropped.

    left  = protein A          right = ligand B      (for a PPI pair, right would be protein B)

Self-contained: download this folder and run it in your local PyMOL.

## Run

```bash
pymol p7_pair_pymol.py -- pl6ibk.npz
```

or inside PyMOL:

```
run p7_pair_pymol.py
masif_list .                  # what's here
masif_pair pl6ibk.npz         # load a pair
masif_show vert_charge        # enable vert_charge_left + vert_charge_right, hide other layers
masif_show vert_hphob
masif_show atom_hbond_donor
masif_edges                   # build the dense vv / va edge layers (skipped by default)
```

## Files per pair

```
<id>.npz    both sides in full: atom nodes + 26-D features, vertex nodes + 4-D features,
            surface triangles + normals, all three edge types with their edge features,
            and the training contacts
<id>.pdb    EXACTLY the atoms that are graph nodes, in graph-node order
            (chain A = left, chain L = right). Atom i of chain A is left atom i.
```

## Objects

Everything comes in a `_left` / `_right` pair that toggles independently.

| object | what |
|---|---|
| `structure` | the PDB — the actual graph nodes, shown as **lines coloured by chain** (carbons) and by element (heteroatoms), so left and right are distinguishable |
| `vert_{si,hbond,charge,hphob}_{left,right}` | the four per-**vertex-node** features, drawn as a shaded surface (see below) |
| `atom_element_{left,right}` | element identity (categorical, CPK-ish) |
| `atom_hybridization_{left,right}` | sp / sp2 / sp3 (categorical) |
| `atom_<feature>_{left,right}` | one object per remaining atom-node feature — `is_ligand`, `is_backbone`, `aromatic`, `degree`, `is_surface`, `in_ring`, `hbond_donor`, `hbond_acceptor`, `formal_charge`, `flex_depth`, `electronegativity`, `valence`, `covalent_radius` |
| `edges_aa_{left,right}` | atom–atom covalent edges, coloured by **bond order** (white=single, blue=double, orange=aromatic, purple=other) |
| `edges_aa_rot_{left,right}` | the **sidechain-rotatable** subset only (magenta) — the bond-rotatability signal |
| `edges_vv_{dist,cos}_{left,right}` | vertex–vertex mesh edges, coloured by their **edge features** (via `masif_edges`) |
| `edges_va_{dist,cos}_{left,right}` | vertex→atom edges, likewise (via `masif_edges`) |
| `contacts` | the training positives linking left to right |

That is every feature the GNN consumes: the 26-D atom node vector, the 4-D vertex node vector, and
all three edge types **with their edge features**. 51 objects in total.

### Edges are not connectivity-only

Each edge carries a feature vector that is concatenated with the source node state inside the
message MLP — `msg(concat[h_src, edge_feat])` — so these values are as much an input as the node
features are:

| edge type | dim | contents |
|---|---|---|
| `aa` atom–atom (covalent) | 5 | bond-order one-hot ×4 (single / double / aromatic / other) **+ sidechain-rotatable flag** |
| `vv` vertex–vertex (mesh) | 9 | edge length → RBF (8 Gaussians, centres 0–4 Å) + cos(normal_i, normal_j) |
| `va` vertex→atom | 9 | distance → RBF (8 Gaussians, 0–5 Å) + cos(normal_v, unit(atom − vertex)) |

Every one of them is an **SE(3)-invariant scalar** — a distance or a cosine. No coordinate ever
enters the network, which is what makes the encoder provably rotation-invariant. The atom→vertex
direction reuses the same `va` feature with a *separate* MLP, so direction is distinguished by the
function rather than by the feature. The `sidechain_rotatable` flag is the bond-rotatability signal
the project is built around — see it on its own with `masif_show edges_aa_rot`.

Objects are created **feature-major**, so `<feature>_left` and `<feature>_right` sit next to each
other in the panel:

```
vert_si_left        vert_si_right
vert_hbond_left     vert_hbond_right
...
atom_element_left   atom_element_right
...
```

## What the GNN actually consumes (and what is just rendering)

The four channels are **per-vertex NODE features** — `vert_feat` is an `(n_vert, 4)` tensor attached
to vertex nodes. That is why they are named `vert_*`.

The **triangulated surface is not a model input.** The npz the encoder reads contains no faces at
all; they are exported here only for drawing. The mesh reaches the network in two indirect ways:

* **topology** — `vv_edge` is derived from the mesh faces, so mesh adjacency becomes the
  vertex–vertex edge set;
* **geometry** — the edge scalars `vv_dist`, `vv_cos`, `va_dist`, `va_cos`, and `si` itself
  (computed from the mesh's discrete curvature).

So `vert_si_left` is drawn as a shaded surface purely because that reads far better than a point
cloud, but the number it shows lives on a vertex node.

## Colour scales

The background is left at PyMOL's default (black). Scalar channels use **blue → white → red**. The scale is symmetric about 0 for signed channels, and
the **same range is applied to left and right**, so the two partners are directly comparable rather
than each auto-scaled to itself. Binary features read grey (0) / red (1).

Only `structure`, `vert_si_left`, `vert_si_right` and `contacts` are enabled at start (47 objects
are built, so the session would be sluggish with everything on). `edges_vv` / `edges_va` are the
heaviest layers — tens of thousands of lines on the protein side — and are built only when you ask
for them with `masif_edges`.

## What to look for

* `vert_si` — a correct surface follows the molecular skeleton lobe-for-lobe, red on convex caps and
  blue in concave grooves. The **protein pocket should be blue where the ligand sits**, and the
  ligand mostly red: complementarity is convex-meets-concave.
* `contacts` — should fan across the whole buried face of the ligand, not cluster on one spot.
* `atom_is_surface` — red atoms are the readout nodes the encoder actually emits embeddings for.
* `edges_va_dist` — should link each vertex to the atoms directly beneath it (blue = near), never
  across the molecule; the 5 Å ball cut-off is visible as the reddest edges.
* `edges_aa` — bond order by colour; `edges_aa_rot` isolates the rotatable bonds.

## The five pairs

| id | protein atoms / verts | ligand atoms / verts | contacts |
|---|---|---|---|
| pl4o9w | 1,722 / 4,317 | 41 / 257 | 232 |
| pl4xu2 | 1,955 / 4,820 | 37 / 222 | 486 |
| pl6ibk | 3,123 / 6,371 | 16 / 102 | 171 |
| pl4ivc | 2,348 / 6,293 | 24 / 133 | 246 |
| pl6e6m | 1,123 / 3,098 | 19 / 129 | 173 |
