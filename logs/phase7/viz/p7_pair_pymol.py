"""PyMOL viewer for a MaSIF-graph TRAINING PAIR — every feature the GNN consumes, per side.

A training example is a pair of interacting partners, so everything here comes in a `_left` /
`_right` version that toggles independently:

    left  = protein A
    right = protein B (PPI) or ligand B (protein-ligand)

Run
---
    pymol p7_pair_pymol.py -- pl6ibk.npz          # auto-load
or, inside PyMOL:
    run p7_pair_pymol.py
    masif_list .                    # what is in this folder
    masif_pair pl6ibk.npz           # load a pair
    masif_pair pl6ibk.npz, dense=1  # also build the vv/va edge layers up front
    masif_show vert_charge          # enable vert_charge_left + vert_charge_right, hide the rest
    masif_show surf_si
    masif_show atom_hbond_donor
    masif_edges                     # build the dense edge layers on demand

Objects
-------
    structure                     the PDB — EXACTLY the atoms that are graph nodes, in node order
                                  (chain A = left, chain L/B = right)

    surf_{si,hbond,charge,hphob}_{left,right}     surface, colour-coded by that vertex channel
    vert_{si,hbond,charge,hphob}_{left,right}     the same channel as coloured vertex NODES
    atom_<feature>_{left,right}                   every one of the 26 atom-node features
    atom_element_{left,right}                     element identity (categorical)
    atom_hybridization_{left,right}               sp / sp2 / sp3 (categorical)
    edges_aa_{left,right}                         atom-atom covalent edges, coloured by bond order
    edges_aa_rot_{left,right}                     the rotatable-bond subset only
    edges_vv_{left,right}                         vertex-vertex mesh edges
    edges_va_{left,right}                         vertex->atom edges
    contacts                                      the training positives linking left to right

Scalar channels use a blue -> white -> red ramp over the channel's own range, printed on load, and
the SAME range is used for left and right so the two sides are directly comparable. Binary features
are grey (0) / red (1).

Only `structure`, `surf_si_*` and `contacts` are enabled at start; everything else is built and
disabled so the session opens responsive. Toggle in the object panel or use `masif_show`.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from pymol import cmd

try:
    import numpy as np
    from pymol.cgo import BEGIN, COLOR, END, LINES, LINEWIDTH, NORMAL, SPHERE, TRIANGLES, VERTEX
except ImportError as exc:                                          # pragma: no cover
    raise ImportError("this script needs numpy inside PyMOL's Python: %s" % exc)

_S = {"npz": None, "meta": None}

_ELEM_RGB = {"C": (0.30, 0.30, 0.32), "N": (0.20, 0.35, 0.95), "O": (0.90, 0.15, 0.15),
             "S": (0.90, 0.78, 0.20), "P": (1.00, 0.55, 0.10), "F": (0.35, 0.85, 0.35),
             "Cl": (0.55, 0.75, 0.25), "Br": (0.65, 0.35, 0.15), "I": (0.55, 0.25, 0.70),
             "other": (0.90, 0.20, 0.90)}
_BOND_RGB = [(1.0, 1.0, 1.0), (0.35, 0.75, 1.0), (1.0, 0.55, 0.15), (0.75, 0.30, 0.85)]
_VERT_R, _ATOM_R = 0.11, 0.26


def _ramp(t):
    """t in [-1,1] -> blue/white/red."""
    t = max(-1.0, min(1.0, float(t)))
    if t >= 0:
        return (1.0, 1.0 - 0.85 * t, 1.0 - 0.90 * t)
    return (1.0 + 0.90 * t, 1.0 + 0.75 * t, 1.0)


def _norm(vals, lo=None, hi=None):
    v = np.asarray(vals, dtype=float)
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    if hi - lo < 1e-9:
        return np.zeros_like(v), lo, hi
    if lo < 0 < hi:                      # signed -> keep 0 at white, symmetric scale
        m = max(abs(lo), abs(hi))
        return v / m, -m, m
    return 2.0 * (v - lo) / (hi - lo) - 1.0, lo, hi


def _spheres(xyz, rgb_per_point, rad):
    obj = []
    for p, c in zip(xyz, rgb_per_point):
        obj += [COLOR, float(c[0]), float(c[1]), float(c[2]),
                SPHERE, float(p[0]), float(p[1]), float(p[2]), float(rad)]
    return obj


def _surface(verts, faces, normals, vals):
    obj = [BEGIN, TRIANGLES]
    cols = [_ramp(v) for v in vals]
    for f in faces:
        for i in f:
            r, g, b = cols[i]
            n = normals[i]
            obj += [COLOR, r, g, b, NORMAL, float(n[0]), float(n[1]), float(n[2]),
                    VERTEX, float(verts[i][0]), float(verts[i][1]), float(verts[i][2])]
    obj.append(END)
    return obj


def _lines(pa, pb, cols, width=1.0):
    obj = [LINEWIDTH, float(width), BEGIN, LINES]
    for a, b, c in zip(pa, pb, cols):
        obj += [COLOR, float(c[0]), float(c[1]), float(c[2]),
                VERTEX, float(a[0]), float(a[1]), float(a[2]),
                VERTEX, float(b[0]), float(b[1]), float(b[2])]
    obj.append(END)
    return obj


def _build_side(z, tag, meta, dense):
    """Create every per-side object for one partner. Returns the names created."""
    made = []
    axyz = z["%s_atom_xyz" % tag]
    af = z["%s_atom_feat" % tag]
    vxyz = z["%s_vert_xyz" % tag]
    vn = z["%s_vert_normal" % tag]
    vf = z["%s_vert_feat" % tag]
    faces = z["%s_faces" % tag]
    names = meta["atom_features"]

    # ---- vertex channels: as a coloured SURFACE and as coloured vertex NODES ----
    for k, ch in enumerate(meta["vert_features"]):
        if len(vxyz) == 0:
            continue
        t, lo, hi = _norm(vf[:, k])
        if len(faces):
            cmd.load_cgo(_surface(vxyz, faces, vn, t), "surf_%s_%s" % (ch, tag))
            made.append("surf_%s_%s" % (ch, tag))
        cmd.load_cgo(_spheres(vxyz, [_ramp(x) for x in t], _VERT_R), "vert_%s_%s" % (ch, tag))
        made.append("vert_%s_%s" % (ch, tag))

    # ---- atom features: element + hybridization categorical, the rest as scalar ramps ----
    el = af[:, 0:10]
    ei = el.argmax(1)
    cmd.load_cgo(_spheres(axyz, [_ELEM_RGB.get(meta["elements"][i], (1, 0, 1)) for i in ei],
                          _ATOM_R), "atom_element_%s" % tag)
    made.append("atom_element_%s" % tag)
    hy = af[:, 16:19]
    hcol = [(0.4, 0.4, 0.4), (0.95, 0.45, 0.1), (0.15, 0.55, 0.95)]
    cmd.load_cgo(_spheres(axyz, [hcol[i] if hy[j].max() > 0 else (0.85, 0.85, 0.85)
                                 for j, i in enumerate(hy.argmax(1))], _ATOM_R),
                 "atom_hybridization_%s" % tag)
    made.append("atom_hybridization_%s" % tag)
    for k in list(range(10, 16)) + list(range(19, 26)):
        t, lo, hi = _norm(af[:, k])
        cmd.load_cgo(_spheres(axyz, [_ramp(x) for x in t], _ATOM_R),
                     "atom_%s_%s" % (names[k], tag))
        made.append("atom_%s_%s" % (names[k], tag))

    # ---- edges ----
    aa = z["%s_aa_edge" % tag]
    if len(aa):
        order = z["%s_aa_order" % tag].argmax(1)
        cmd.load_cgo(_lines(axyz[aa[:, 0]], axyz[aa[:, 1]],
                            [_BOND_RGB[o] for o in order], 2.5), "edges_aa_%s" % tag)
        made.append("edges_aa_%s" % tag)
        rot = z["%s_aa_rot" % tag] > 0.5
        if rot.any():
            cmd.load_cgo(_lines(axyz[aa[rot, 0]], axyz[aa[rot, 1]],
                                [(1.0, 0.2, 0.8)] * int(rot.sum()), 3.0), "edges_aa_rot_%s" % tag)
            made.append("edges_aa_rot_%s" % tag)
    if dense:
        made += _build_edges_side(z, tag)
    return made


def _build_edges_side(z, tag):
    made = []
    axyz = z["%s_atom_xyz" % tag]; vxyz = z["%s_vert_xyz" % tag]
    vv = z["%s_vv_edge" % tag]
    if len(vv) and not cmd.get_names("objects", 0).count("edges_vv_%s" % tag):
        cmd.load_cgo(_lines(vxyz[vv[:, 0]], vxyz[vv[:, 1]],
                            [(0.55, 0.55, 0.55)] * len(vv), 1.0), "edges_vv_%s" % tag)
        made.append("edges_vv_%s" % tag)
    va = z["%s_va_edge" % tag]
    if len(va):
        cmd.load_cgo(_lines(vxyz[va[:, 0]], axyz[va[:, 1]],
                            [(1.0, 0.95, 0.10)] * len(va), 1.0), "edges_va_%s" % tag)
        made.append("edges_va_%s" % tag)
    return made


def masif_edges():
    """Build the dense vertex-vertex / vertex-atom edge layers (skipped by default)."""
    if _S["npz"] is None:
        print("[masif] load a pair first")
        return
    z = np.load(_S["npz"])
    made = []
    for tag in ("left", "right"):
        made += _build_edges_side(z, tag)
    for m in made:
        cmd.disable(m)
    print("[masif] built %s (disabled; enable in the panel)" % ", ".join(made) if made else
          "[masif] nothing to build")


def masif_pair(npz_path, dense=0):
    npz_path = os.path.abspath(str(npz_path).strip())
    if not os.path.exists(npz_path):
        print("[masif] no such file: %s" % npz_path)
        return
    cmd.delete("all")
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    _S["npz"], _S["meta"] = npz_path, meta

    pdb = npz_path[:-4] + ".pdb"
    if os.path.exists(pdb):
        cmd.load(pdb, "structure")
        cmd.hide("everything", "structure")
        cmd.show("lines", "structure")
        cmd.color("grey60", "structure")
        cmd.util.cnc("structure")

    made = []
    for tag in ("left", "right"):
        made += _build_side(z, tag, meta, int(dense))

    ca = z["contacts_atom"]
    if len(ca):
        lx = z["left_atom_xyz"]; rx = z["right_atom_xyz"]
        cmd.load_cgo(_lines(lx[ca[:, 0]], rx[ca[:, 1]], [(0.15, 0.9, 0.35)] * len(ca), 1.5),
                     "contacts")
        made.append("contacts")

    cmd.set("two_sided_lighting", 1); cmd.set("ray_shadows", 0); cmd.bg_color("white")
    keep = {"structure", "surf_si_left", "surf_si_right", "contacts"}
    for m in made:
        if m not in keep:
            cmd.disable(m)
    cmd.orient("structure" if os.path.exists(pdb) else "surf_si_left")

    L, R = meta["left"], meta["right"]
    print("[masif] %s (%s)" % (meta["id"], meta["kind"]))
    print("        left  = %-10s atoms %5d  verts %5d  surf-atoms %4d  aa %5d  vv %6d  va %6d"
          % (meta["left_label"], L["atoms"], L["verts"], L["surf_atoms"], L["aa"], L["vv"], L["va"]))
    print("        right = %-10s atoms %5d  verts %5d  surf-atoms %4d  aa %5d  vv %6d  va %6d"
          % (meta["right_label"], R["atoms"], R["verts"], R["surf_atoms"], R["aa"], R["vv"], R["va"]))
    print("        contacts (training positives): %d" % meta["n_contacts"])
    print("[masif] %d objects. shown: structure, surf_si_left/right, contacts" % len(made))
    print("[masif] masif_show <prefix>   e.g. vert_charge | surf_hphob | atom_hbond_donor | edges_aa")
    print("[masif] masif_edges           build the dense vv/va layers")
    print("[masif] atom features: %s" % ", ".join(meta["atom_features"]))


def masif_show(prefix):
    """Enable <prefix>_left and <prefix>_right, hide the other feature layers."""
    prefix = str(prefix).strip()
    names = cmd.get_names("objects")
    for n in names:
        if n == "structure":
            continue
        if n in ("%s_left" % prefix, "%s_right" % prefix):
            cmd.enable(n)
        elif n.startswith(("surf_", "vert_", "atom_", "edges_")):
            cmd.disable(n)
    hits = [n for n in names if n in ("%s_left" % prefix, "%s_right" % prefix)]
    print("[masif] showing %s" % (", ".join(hits) if hits else "(nothing matched %r)" % prefix))


def masif_list(folder="."):
    fs = sorted(glob.glob(os.path.join(str(folder).strip(), "*.npz")))
    if not fs:
        print("[masif] no .npz in %s" % folder)
        return
    for f in fs:
        try:
            m = json.loads(str(np.load(f)["meta"]))
            print("   masif_pair %s      # %s: %s + %s, %d contacts"
                  % (os.path.basename(f), m["kind"], m["left_label"], m["right_label"],
                     m["n_contacts"]))
        except Exception:                                           # noqa: BLE001
            print("   masif_pair %s" % os.path.basename(f))


cmd.extend("masif_pair", masif_pair)
cmd.extend("masif_show", masif_show)
cmd.extend("masif_edges", masif_edges)
cmd.extend("masif_list", masif_list)

_a = [x for x in sys.argv[1:] if x.endswith(".npz")]
if _a:
    masif_pair(_a[0])
