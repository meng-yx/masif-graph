"""PyMOL viewer for a Phase-7 protein-ligand pair: both MSMS surfaces, the graph, and the channels.

Companion to `p7_export_pl_viz.py`, in the same spirit as the Phase-4 pair
(`p4_export_graph_viz.py` / `p4_graph_pymol.py`). Everything it draws comes out of the `.npz` and the
two small `.pdb` files next to it, so the folder is self-contained — copy it anywhere and run.

Usage
-----
  # one-liner (auto-runs on the .npz):
  pymol p7_pl_pymol.py -- 5czm.npz

  # or interactively inside PyMOL:
  run p7_pl_pymol.py
  masif_pl 5czm.npz                 # load one example
  masif_pl 5czm.npz, show_dense=1   # also show the vertex/edge layers immediately
  masif_channel charge              # recolour BOTH surfaces by another channel
  masif_pl_all .                    # load every .npz in a directory

Objects created (each toggleable in the panel)
---------------------------------------------
  ligand              ligand as sticks (from <id>_ligand.pdb)
  pocket              protein pocket residues (from <id>_pocket.pdb)
  lig_surf            LIGAND MSMS surface, shaded by the current channel      <- the Phase-7 object
  prot_surf           protein pocket surface, same channel, same colour scale
  lig_atoms           ligand atom NODES                    orange spheres
  lig_verts           ligand surface-vertex NODES          marine dots
  lig_aa_edges        ligand atom-atom (bond) edges        white lines
  lig_vv_edges        ligand mesh edges                    grey lines
  lig_va_edges        ligand vertex->atom edges            yellow lines
  contacts            training contact pairs (<=5 A)       green lines
  prot_verts          protein surface-vertex nodes         pale blue dots

The dense layers (lig_vv_edges, lig_va_edges, prot_verts) start hidden so the session opens
responsive; enable them from the panel or pass `show_dense=1`.

Channels: si (shape index), hbond, charge, hphob. `si` is the default because it is the one that
tells you whether the surface geometry is right at a glance — a real molecular surface has red
convex caps with blue concave grooves between the ring systems; a degenerate one is uniformly red.
"""
from __future__ import annotations

import glob
import os
import sys

from pymol import cmd

try:
    import numpy as np
    from pymol.cgo import (ALPHA, BEGIN, COLOR, END, LINES, LINEWIDTH, NORMAL, SPHERE, TRIANGLES,
                           VERTEX)
except ImportError as exc:                                          # pragma: no cover
    raise ImportError("this script needs numpy inside PyMOL's Python: %s" % exc)

CHANNELS = ["si", "hbond", "charge", "hphob"]
_STATE = {"npz": None, "channel": "si"}

# node radii / colours, matching the Phase-4 viewer's conventions
_VERT_R, _ATOM_R = 0.10, 0.22
_VERT_C = (0.20, 0.55, 1.00)
_ATOM_C = (1.00, 0.55, 0.20)
_PVERT_C = (0.55, 0.75, 0.95)
_AA_C = (1.00, 1.00, 1.00)
_VV_C = (0.55, 0.55, 0.55)
_VA_C = (1.00, 0.95, 0.10)
_CT_C = (0.20, 0.85, 0.35)


def _ramp(v):
    """Diverging blue -> white -> red over [-1, 1]; v is a float in that range."""
    t = max(-1.0, min(1.0, float(v)))
    if t >= 0:
        return (1.0, 1.0 - 0.85 * t, 1.0 - 0.90 * t)      # white -> red
    return (1.0 + 0.90 * t, 1.0 + 0.75 * t, 1.0)          # white -> blue


def _surface_cgo(verts, faces, normals, vals):
    """Per-vertex-coloured triangles. PyMOL shades them properly given NORMALs."""
    obj = [BEGIN, TRIANGLES]
    for f in faces:
        for i in f:
            r, g, b = _ramp(vals[i])
            n = normals[i]
            obj += [COLOR, r, g, b, NORMAL, float(n[0]), float(n[1]), float(n[2]),
                    VERTEX, float(verts[i][0]), float(verts[i][1]), float(verts[i][2])]
    obj.append(END)
    return obj


def _spheres(xyz, rgb, rad):
    obj = [COLOR, float(rgb[0]), float(rgb[1]), float(rgb[2])]
    for p in xyz:
        obj += [SPHERE, float(p[0]), float(p[1]), float(p[2]), float(rad)]
    return obj


def _lines(pa, pb, rgb, width=1.0):
    obj = [LINEWIDTH, float(width), BEGIN, LINES, COLOR, float(rgb[0]), float(rgb[1]), float(rgb[2])]
    for a, b in zip(pa, pb):
        obj += [VERTEX, float(a[0]), float(a[1]), float(a[2]),
                VERTEX, float(b[0]), float(b[1]), float(b[2])]
    obj.append(END)
    return obj


def _chan_idx(name):
    name = str(name).strip().lower()
    if name not in CHANNELS:
        raise ValueError("channel must be one of %s (got %r)" % (CHANNELS, name))
    return CHANNELS.index(name)


def masif_channel(channel="si"):
    """Recolour both surfaces by another channel, keeping the current view."""
    if _STATE["npz"] is None:
        print("[masif] load an example first: masif_pl <file.npz>")
        return
    z = np.load(_STATE["npz"])
    k = _chan_idx(channel)
    _STATE["channel"] = CHANNELS[k]
    view = cmd.get_view()
    cmd.delete("lig_surf"); cmd.delete("prot_surf")
    cmd.load_cgo(_surface_cgo(z["lig_verts"], z["lig_faces"], z["lig_normals"],
                              z["lig_feat"][:, k]), "lig_surf")
    if len(z["prot_faces"]):
        pn = _vertex_normals(z["prot_verts"], z["prot_faces"])
        cmd.load_cgo(_surface_cgo(z["prot_verts"], z["prot_faces"], pn,
                                  z["prot_feat"][:, k]), "prot_surf")
        cmd.set("cgo_transparency", 0.55, "prot_surf")
    cmd.set_view(view)
    print("[masif] channel = %s   (blue = negative, white = 0, red = positive)" % CHANNELS[k])


def _vertex_normals(verts, faces):
    """Area-weighted vertex normals (the protein mesh is exported without its own normals)."""
    n = np.zeros_like(verts, dtype=float)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(ln, 1e-12, None)


def masif_pl(npz_path, show_dense=0, channel="si"):
    """Load one exported protein-ligand example."""
    npz_path = os.path.abspath(str(npz_path).strip())
    if not os.path.exists(npz_path):
        print("[masif] no such file: %s" % npz_path)
        return
    z = np.load(npz_path)
    stem = npz_path[:-4]
    _STATE["npz"] = npz_path
    for o in ("ligand", "pocket", "lig_surf", "prot_surf", "lig_atoms", "lig_verts",
              "lig_aa_edges", "lig_vv_edges", "lig_va_edges", "contacts", "prot_verts"):
        cmd.delete(o)

    if os.path.exists(stem + "_ligand.pdb"):
        cmd.load(stem + "_ligand.pdb", "ligand")
        cmd.show_as("sticks", "ligand"); cmd.color("grey30", "ligand and elem C")
        cmd.util.cnc("ligand")
    if os.path.exists(stem + "_pocket.pdb"):
        cmd.load(stem + "_pocket.pdb", "pocket")
        cmd.show_as("cartoon", "pocket"); cmd.color("palecyan", "pocket")
        cmd.set("cartoon_transparency", 0.5, "pocket")

    masif_channel(channel)

    lx = z["lig_atom_xyz"]; lv = z["lig_verts"]
    cmd.load_cgo(_spheres(lx, _ATOM_C, _ATOM_R), "lig_atoms")
    cmd.load_cgo(_spheres(lv, _VERT_C, _VERT_R), "lig_verts")
    cmd.load_cgo(_spheres(z["prot_verts"], _PVERT_C, _VERT_R), "prot_verts")

    aa = z["lig_aa_edge"]
    if len(aa):
        cmd.load_cgo(_lines(lx[aa[:, 0]], lx[aa[:, 1]], _AA_C, 2.0), "lig_aa_edges")
    vv = z["lig_vv_edge"]
    if len(vv):
        cmd.load_cgo(_lines(lv[vv[:, 0]], lv[vv[:, 1]], _VV_C, 1.0), "lig_vv_edges")
    va = z["lig_va_edge"]
    if len(va):
        cmd.load_cgo(_lines(lv[va[:, 0]], lx[va[:, 1]], _VA_C, 1.0), "lig_va_edges")
    cp, cl = z["contact_prot_xyz"], z["contact_lig_xyz"]
    if len(cp):
        cmd.load_cgo(_lines(cl, cp, _CT_C, 1.5), "contacts")

    cmd.set("cgo_line_width", 1.0)
    cmd.set("two_sided_lighting", 1)
    cmd.set("ray_shadows", 0)
    cmd.bg_color("white")
    if not int(show_dense):
        for o in ("lig_vv_edges", "lig_va_edges", "prot_verts", "lig_verts"):
            cmd.disable(o)
    cmd.orient("ligand" if os.path.exists(stem + "_ligand.pdb") else "lig_surf")
    cmd.zoom("lig_surf", 3.0)

    import json
    rep = json.loads(str(z["report"])) if "report" in z.files else {}
    print("[masif] %s | ligand %s atoms, %s verts, %s faces | %s contacts"
          % (rep.get("id", "?"), rep.get("lig_atoms", "?"), rep.get("lig_verts", "?"),
             rep.get("lig_faces", "?"), rep.get("n_contacts", "?")))
    print("[masif] objects: lig_surf prot_surf ligand pocket lig_atoms lig_verts "
          "lig_aa_edges lig_vv_edges lig_va_edges contacts prot_verts")
    print("[masif] try:  masif_channel hbond | charge | hphob      (default si)")


def masif_pl_all(folder="."):
    """List the examples in a folder (load one with masif_pl)."""
    fs = sorted(glob.glob(os.path.join(str(folder).strip(), "*.npz")))
    if not fs:
        print("[masif] no .npz in %s" % folder)
        return
    print("[masif] %d example(s):" % len(fs))
    for f in fs:
        print("   masif_pl %s" % f)


cmd.extend("masif_pl", masif_pl)
cmd.extend("masif_channel", masif_channel)
cmd.extend("masif_pl_all", masif_pl_all)

# `pymol p7_pl_pymol.py -- foo.npz` -> auto-load
_args = [a for a in sys.argv[1:] if a.endswith(".npz")]
if _args:
    masif_pl(_args[0])
