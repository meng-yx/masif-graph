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
    masif_pair pl6ibk.npz, dense=0  # skip the dense vv/va edge layers (they are ~0.2 s)
    masif_show vert_charge          # enable vert_charge_left + vert_charge_right, hide the rest
    masif_show atom_hbond_donor
    masif_edges                     # (re)build the dense edge layers if they were skipped

Objects
-------
    structure                     the PDB — EXACTLY the atoms that are graph nodes, in node order
                                  (chain A = left, chain L/B = right)

    vert_{si,hbond,charge,hphob}_{left,right}     per-VERTEX NODE features, drawn as a shaded
                                  surface for legibility (the surface itself is not a GNN input)
    atom_<feature>_{left,right}                   every one of the 26 atom-node features
    atom_element_{left,right}                     element identity (categorical)
    atom_hybridization_{left,right}               sp / sp2 / sp3 (categorical)
    edges_aa_{left,right}                         atom-atom covalent edges, coloured by bond order
    edges_aa_rot_{left,right}                     the rotatable-bond subset only
    edges_vv_{dist,cos}_{left,right}              mesh edges, coloured by their edge FEATURES
    edges_va_{dist,cos}_{left,right}              vertex->atom edges, likewise
    contacts                                      the training positives linking left to right

Scalar channels use a blue -> white -> red ramp over the channel's own range, printed on load, and
the SAME range is used for left and right so the two sides are directly comparable. Binary features
are grey (0) / red (1).

Objects are created FEATURE-MAJOR, so `<feature>_left` and `<feature>_right` sit next to each other
in the panel. ALL of them are built (including the dense vv/va edge layers), but only `structure`,
`vert_si_*` and `contacts` start *enabled* so the session opens responsive — the rest are present in
the panel, just switched off. Toggle there or use `masif_show`.
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

_S = {"npz": None, "meta": None, "ranges": {}}

_ELEM_RGB = {"C": (0.62, 0.62, 0.66), "N": (0.20, 0.35, 0.95), "O": (0.90, 0.15, 0.15),
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


def _rng(z, key, k):
    """Colour range shared by BOTH partners for one channel.

    Scaling each side to itself would make the two partners look comparable when they are not --
    the point of a shared encoder is that a value means the same thing on either side."""
    vals = []
    for tag in ("left", "right"):
        a = z["%s_%s" % (tag, key)]
        if len(a):
            vals.append(a[:, k])
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return lo - 1e-9, hi + 1e-9
    if lo < 0 < hi:                      # signed -> symmetric about 0 so white means zero
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def _scale(v, lo, hi):
    return 2.0 * (np.asarray(v, float) - lo) / (hi - lo) - 1.0


def _build_all(z, meta, dense):
    """Create every object, FEATURE-MAJOR so <feature>_left and <feature>_right are adjacent."""
    made = []
    A = {t: z["%s_atom_xyz" % t] for t in ("left", "right")}
    V = {t: z["%s_vert_xyz" % t] for t in ("left", "right")}
    ranges = {}

    # ---- vertex channels ----
    # Named `vert_*` because in the GNN these are per-VERTEX NODE features: `vert_feat` is an
    # (n_vert, 4) tensor on vertex nodes. The triangulated surface is NOT a model input -- the npz
    # the encoder reads has no faces at all. The mesh reaches the network only as vv-edge
    # connectivity plus the edge scalars derived from it. It is drawn as a surface here purely
    # because a shaded surface is far easier to read than a point cloud.
    for k, ch in enumerate(meta["vert_features"]):
        lo, hi = _rng(z, "vert_feat", k)
        ranges["vert_" + ch] = (lo, hi)
        for tag in ("left", "right"):
            if len(V[tag]) == 0:
                continue
            t = _scale(z["%s_vert_feat" % tag][:, k], lo, hi)
            faces = z["%s_faces" % tag]
            nm = "vert_%s_%s" % (ch, tag)
            if len(faces):
                cmd.load_cgo(_surface(V[tag], faces, z["%s_vert_normal" % tag], t), nm)
            else:                     # no mesh available -> fall back to the raw node cloud
                cmd.load_cgo(_spheres(V[tag], [_ramp(x) for x in t], _VERT_R), nm)
            made.append(nm)

    # ---- atom nodes: categorical first, then every scalar feature ----
    for tag in ("left", "right"):
        ei = z["%s_atom_feat" % tag][:, 0:10].argmax(1)
        cmd.load_cgo(_spheres(A[tag], [_ELEM_RGB.get(meta["elements"][i], (1, 0, 1)) for i in ei],
                              _ATOM_R), "atom_element_%s" % tag)
        made.append("atom_element_%s" % tag)
    hcol = [(0.70, 0.70, 0.70), (0.95, 0.45, 0.10), (0.15, 0.55, 0.95)]
    for tag in ("left", "right"):
        hy = z["%s_atom_feat" % tag][:, 16:19]
        cmd.load_cgo(_spheres(A[tag], [hcol[i] if hy[j].max() > 0 else (0.85, 0.85, 0.85)
                                       for j, i in enumerate(hy.argmax(1))], _ATOM_R),
                     "atom_hybridization_%s" % tag)
        made.append("atom_hybridization_%s" % tag)
    names = meta["atom_features"]
    for k in list(range(10, 16)) + list(range(19, 26)):
        lo, hi = _rng(z, "atom_feat", k)
        ranges["atom_" + names[k]] = (lo, hi)
        for tag in ("left", "right"):
            t = _scale(z["%s_atom_feat" % tag][:, k], lo, hi)
            cmd.load_cgo(_spheres(A[tag], [_ramp(x) for x in t], _ATOM_R),
                         "atom_%s_%s" % (names[k], tag))
            made.append("atom_%s_%s" % (names[k], tag))

    # ---- edges ----
    for tag in ("left", "right"):
        aa = z["%s_aa_edge" % tag]
        if len(aa):
            order = z["%s_aa_order" % tag].argmax(1)
            cmd.load_cgo(_lines(A[tag][aa[:, 0]], A[tag][aa[:, 1]],
                                [_BOND_RGB[o] for o in order], 2.5), "edges_aa_%s" % tag)
            made.append("edges_aa_%s" % tag)
    for tag in ("left", "right"):
        aa = z["%s_aa_edge" % tag]
        rot = z["%s_aa_rot" % tag] > 0.5 if len(aa) else np.zeros(0, bool)
        if len(aa) and rot.any():
            cmd.load_cgo(_lines(A[tag][aa[rot, 0]], A[tag][aa[rot, 1]],
                                [(1.0, 0.2, 0.8)] * int(rot.sum()), 3.0), "edges_aa_rot_%s" % tag)
            made.append("edges_aa_rot_%s" % tag)
    if dense:
        made += _build_edges(z)
    _S["ranges"] = ranges
    return made


def _build_edges(z):
    """Dense mesh / vertex-atom layers, COLOURED BY THEIR EDGE FEATURES.

    vv and va edges are not bare connectivity: each carries a distance (RBF-expanded before the
    message MLP) and a cos-angle. Both are drawn, so every scalar the GNN sees on an edge is
    visible. Ranges are shared left/right, as for the node features."""
    made = []
    have = set(cmd.get_names("objects"))
    for kind in ("vv", "va"):
        for feat in ("dist", "cos"):
            key = "%s_%s" % (kind, feat)
            vals = [z["%s_%s" % (t, key)] for t in ("left", "right")
                    if len(z["%s_%s_edge" % (t, kind)])]
            if not vals:
                continue
            v = np.concatenate(vals)
            lo, hi = float(v.min()), float(v.max())
            if lo < 0 < hi:
                m = max(abs(lo), abs(hi)); lo, hi = -m, m
            if hi - lo < 1e-9:
                hi = lo + 1e-9
            for tag in ("left", "right"):
                nm = "edges_%s_%s_%s" % (kind, feat, tag)
                if nm in have:
                    continue
                e = z["%s_%s_edge" % (tag, kind)]
                if not len(e):
                    continue
                t = _scale(z["%s_%s" % (tag, key)], lo, hi)
                src = z["%s_vert_xyz" % tag]
                dst = z["%s_vert_xyz" % tag] if kind == "vv" else z["%s_atom_xyz" % tag]
                cmd.load_cgo(_lines(src[e[:, 0]], dst[e[:, 1]], [_ramp(x) for x in t], 1.0), nm)
                made.append(nm)
    return made


def masif_edges():
    """Build the dense vertex-vertex / vertex-atom edge layers (skipped by default)."""
    if _S["npz"] is None:
        print("[masif] load a pair first")
        return
    made = _build_edges(np.load(_S["npz"]))
    for m in made:
        cmd.disable(m)
    print("[masif] built %s (disabled; enable in the panel)" % ", ".join(made) if made else
          "[masif] nothing to build")


def masif_pair(npz_path, dense=1):
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
        cmd.show_as("lines", "structure")
        cmd.util.cbc("structure")      # carbons by chain -> left (A) and right (L/B) differ
        cmd.util.cnc("structure")      # heteroatoms by element

    made = _build_all(z, meta, int(dense))

    ca = z["contacts_atom"]
    if len(ca):
        lx = z["left_atom_xyz"]; rx = z["right_atom_xyz"]
        cmd.load_cgo(_lines(lx[ca[:, 0]], rx[ca[:, 1]], [(0.15, 0.9, 0.35)] * len(ca), 1.5),
                     "contacts")
        made.append("contacts")

    cmd.set("two_sided_lighting", 1); cmd.set("ray_shadows", 0)
    keep = {"structure", "vert_si_left", "vert_si_right", "contacts"}
    for m in made:
        if m not in keep:
            cmd.disable(m)
    cmd.orient("structure" if os.path.exists(pdb) else "vert_si_left")

    L, R = meta["left"], meta["right"]
    print("[masif] %s (%s)" % (meta["id"], meta["kind"]))
    print("        left  = %-10s atoms %5d  verts %5d  surf-atoms %4d  aa %5d  vv %6d  va %6d"
          % (meta["left_label"], L["atoms"], L["verts"], L["surf_atoms"], L["aa"], L["vv"], L["va"]))
    print("        right = %-10s atoms %5d  verts %5d  surf-atoms %4d  aa %5d  vv %6d  va %6d"
          % (meta["right_label"], R["atoms"], R["verts"], R["surf_atoms"], R["aa"], R["vv"], R["va"]))
    print("        contacts (training positives): %d" % meta["n_contacts"])
    print("[masif] %d objects, ordered feature-major (<feature>_left, <feature>_right, ...)."
          % len(made))
    print("        shown at start: structure, vert_si_left/right, contacts")
    print("[masif] WHAT THE GNN CONSUMES: atom-node features (26-D), vertex-node features (4-D),")
    print("        and 3 edge types. vert_<ch>_* are per-VERTEX NODE features -- drawn as a shaded")
    print("        surface only because it reads better than a point cloud. The triangulated")
    print("        surface itself is NOT an input: the npz the encoder reads has no faces. The")
    print("        mesh reaches the network only as vv-edge connectivity plus the edge scalars")
    print("        (vv_dist/vv_cos, va_dist/va_cos) and si, all derived from its geometry.")
    rr = _S.get("ranges", {})
    print("[masif] shared left/right colour ranges: %s"
          % ", ".join("%s[%.2f,%.2f]" % (k.split("_", 1)[1], v[0], v[1])
                      for k, v in rr.items() if k.startswith("vert_")))
    print("[masif] masif_show <prefix>   e.g. vert_charge | vert_hphob | atom_hbond_donor | edges_aa")
    print("[masif] masif_edges           rebuild the vv/va layers if loaded with dense=0")
    ef = meta.get("edge_features", {})
    for k in ("aa", "vv", "va"):
        if k in ef:
            print("        edge %-2s features: %s" % (k, "; ".join(ef[k])))
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
        elif n.startswith(("vert_", "atom_", "edges_")):
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
