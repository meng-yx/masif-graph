"""PyMOL visualizer for the Phase-4 HeteroSurfaceGraph (the actual GNN input structure).

Loads a graph dump written by `p4_export_graph_viz.py` and draws the node/edge structure as CGO
objects layered over the source PDB. After running you get, in the object panel (each toggleable):

    structure     the normal PDB object(s) (cartoon/sticks as usual)
    vert_nodes    surface-vertex nodes            small marine dots
    atom_nodes    atom (heavy-atom) nodes         bigger orange dots
    aa_edges      atom-atom covalent edges        white lines
    vv_edges      vertex-vertex mesh adjacency    grey lines
    va_edges      vertex-atom links               yellow lines

Usage
-----
  # one-liner (auto-runs on the .npz):
  pymol scripts/p4_graph_pymol.py -- /path/to/graph.npz

  # or interactively inside PyMOL:
  run scripts/p4_graph_pymol.py
  masif_graph_viz /path/to/graph.npz

The two densest layers (vv_edges, va_edges) start hidden so the session opens responsive; toggle
them on in the panel (or `enable va_edges`). `masif_graph_viz npz, show_dense=1` shows all at once.
"""
from __future__ import annotations

import json
import os
import sys

from pymol import cmd

try:
    from pymol.cgo import BEGIN, LINES, VERTEX, END, COLOR, SPHERE, LINEWIDTH
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError("this script needs numpy inside PyMOL's Python: %s" % e)

# node sphere radii (Å-ish, in model units) and colours (r,g,b in [0,1])
_VERT_R, _ATOM_R = 0.1, 0.15
_VERT_C = (0.20, 0.55, 1.00)   # marine
_ATOM_C = (1.00, 0.55, 0.20)   # orange
_AA_C = (1.00, 1.00, 1.00)     # white
_VV_C = (0.55, 0.55, 0.55)     # grey
_VA_C = (1.00, 0.95, 0.10)     # yellow


def _spheres(xyz, rgb, r):
    obj = [COLOR, *rgb]
    for x, y, z in xyz:
        obj += [SPHERE, float(x), float(y), float(z), r]
    return obj


def _lines(p0, p1, rgb, width=1.0):
    obj = [LINEWIDTH, float(width), BEGIN, LINES, COLOR, *rgb]
    for a, b in zip(p0, p1):
        obj += [VERTEX, float(a[0]), float(a[1]), float(a[2]),
                VERTEX, float(b[0]), float(b[1]), float(b[2])]
    obj += [END]
    return obj


def masif_graph_viz(npz, show_dense=0, structure=1):
    """Draw the exported hetero-graph. npz: path from p4_export_graph_viz.py."""
    show_dense = int(show_dense)
    structure = int(structure)
    npz = os.path.expanduser(npz)
    d = np.load(npz, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    atom_xyz, vert_xyz = d["atom_xyz"], d["vert_xyz"]
    aa, vv, va = d["aa_edges"], d["vv_edges"], d["va_edges"]

    # normal PDB object(s), merged into a single "structure" object
    if structure:
        pdbs = [str(p) for p in d["pdb_files"]]
        present = [p for p in pdbs if os.path.exists(p)]
        if not present:
            print("[masif_graph_viz] source PDBs not found at recorded paths; skipping structure:")
            print("   " + "\n   ".join(pdbs))
        else:
            cmd.delete("structure")
            for i, p in enumerate(present):
                tmp = "structure" if i == 0 else "__mg_tmp"
                cmd.load(p, tmp)
                if i:
                    cmd.create("structure", "structure or __mg_tmp")
                    cmd.delete("__mg_tmp")
            cmd.hide("everything", "structure")
            cmd.show("cartoon", "structure")
            cmd.show("sticks", "structure")
            cmd.set("cartoon_transparency", 0.5, "structure")
            cmd.color("grey70", "structure")

    # node clouds
    cmd.load_cgo(_spheres(vert_xyz, _VERT_C, _VERT_R), "vert_nodes")
    cmd.load_cgo(_spheres(atom_xyz, _ATOM_C, _ATOM_R), "atom_nodes")

    # edge line sets (each a single toggleable CGO object)
    cmd.load_cgo(_lines(atom_xyz[aa[:, 0]], atom_xyz[aa[:, 1]], _AA_C, 2.0), "aa_edges")
    cmd.load_cgo(_lines(vert_xyz[vv[:, 0]], vert_xyz[vv[:, 1]], _VV_C, 1.0), "vv_edges")
    cmd.load_cgo(_lines(vert_xyz[va[:, 0]], atom_xyz[va[:, 1]], _VA_C, 1.0), "va_edges")

    # sensible initial visibility: nodes + covalent on; the two dense mesh layers off
    if not show_dense:
        cmd.disable("vv_edges")
        cmd.disable("va_edges")

    cmd.set("two_sided_lighting", 1)
    cmd.bg_color("black")
    cmd.zoom("vert_nodes or atom_nodes")
    cid = meta.get("complex_id", "?")
    print(f"[masif_graph_viz] {cid} ({meta.get('state')}) | "
          f"{atom_xyz.shape[0]} atom + {vert_xyz.shape[0]} vertex nodes | "
          f"aa {aa.shape[0]} / vv {vv.shape[0]} / va {va.shape[0]} edges")
    if not show_dense:
        print("[masif_graph_viz] vv_edges & va_edges hidden for speed — `enable vv_edges` / `enable va_edges` to show.")


cmd.extend("masif_graph_viz", masif_graph_viz)

# auto-run when invoked as `pymol p4_graph_pymol.py -- graph.npz`
_npz = [a for a in sys.argv if a.endswith(".npz")]
if _npz:
    masif_graph_viz(_npz[0])
