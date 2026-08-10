# Phase-7 protein–ligand surface viewer

Five randomly drawn PDBbind complexes from the Phase-7 corpus, exported so the **ligand's MSMS
surface** can be inspected next to the protein's. Self-contained — download this folder and run it
in your local PyMOL; nothing else from the cluster is needed.

## Run

```bash
pymol p7_pl_pymol.py -- 6ibk.npz
```

or, inside PyMOL:

```
run p7_pl_pymol.py
masif_pl_all .          # list the examples
masif_pl 6ibk.npz       # load one
masif_channel charge    # recolour BOTH surfaces by another channel
masif_pl 6ibk.npz, show_dense=1   # open with the vertex/edge layers already on
```

## Objects (all toggleable in the panel)

| object | what it is |
|---|---|
| `lig_surf` | **the ligand MSMS surface**, shaded by the current channel — the Phase-7 object |
| `prot_surf` | protein pocket surface, same channel and colour scale (transparent) |
| `ligand` / `pocket` | real sticks / cartoon, from the companion PDBs |
| `lig_atoms` | ligand atom **nodes** (orange spheres) |
| `lig_verts` | ligand surface-vertex **nodes** (marine dots) |
| `lig_aa_edges` | ligand atom–atom (bond) edges, white |
| `lig_vv_edges` | ligand mesh edges, grey |
| `lig_va_edges` | ligand vertex→atom edges, yellow |
| `contacts` | the training contact pairs (≤5 Å), green |
| `prot_verts` | protein surface-vertex nodes |

`lig_vv_edges`, `lig_va_edges`, `lig_verts` and `prot_verts` start **hidden** so the session opens
responsive — enable them in the panel or pass `show_dense=1`.

The node and edge arrays are read straight out of the npz the encoder consumes, so what you see is
the actual model input, not a re-derivation of it.

## Channels

`si` (shape index, default), `hbond`, `charge`, `hphob`. Colour ramp is **blue = negative, white = 0,
red = positive**, identical for the ligand and the protein so the two are directly comparable.

**What a correct ligand surface looks like:** the mesh follows the molecular skeleton lobe-for-lobe,
and `si` shows red convex caps over the ring systems with **blue concave grooves in the saddles
between them**. A degenerate surface is uniformly red and roughly spherical — that happens for very
small ligands (≲13 atoms), where a 1.5 Å probe rolls over the whole molecule; 45 of 5,239 ligands in
the corpus are affected and are listed in `../lig_surface_degenerate.txt`.

## Files per example

```
<id>.npz            ligand + protein surfaces, channels, graph edges, contacts
<id>_ligand.pdb     the ligand (sticks)
<id>_pocket.pdb     protein residues within 12 Å of the ligand
```

## The five examples

| id | ligand atoms | verts | contacts | elements |
|---|---|---|---|---|
| 4o9w | 41 | 257 | 232 | C,N,O,P |
| 4xu2 | 37 | 222 | 486 | C,N,O,S |
| 6ibk | 16 | 102 | 171 | C,N,O,S |
| 4ivc | 24 | 133 | 246 | C,N,O |
| 6e6m | 19 | 129 | 173 | C,O |
