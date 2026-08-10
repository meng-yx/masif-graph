#!/usr/bin/env python
"""Phase-7 — render a QC sheet per protein-ligand example from `p7_export_pl_viz.py` output.

Six panels, chosen so each one can falsify a different way the ligand surface could be wrong:
  1-4  the ligand mesh shaded by each MaSIF channel (si / hbond / charge / hphob). A degenerate
       "blob" shows up as a featureless sphere; a broken channel shows up as flat colour.
  5    ligand mesh + the protein pocket mesh in the same frame. A frame or pose error shows up as
       the two surfaces sitting apart instead of interdigitating.
  6    channel histograms, ligand vs the protein reference, so offsets are visible rather than
       asserted.

Usage: python scripts/p7_pl_viz_figs.py logs/phase7/viz/*.npz --out notebooks/figs
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection             # noqa: E402

CHAN = ["si (shape index)", "hbond", "charge", "hydrophobicity"]
CMAP = ["coolwarm", "PuOr", "bwr", "BrBG"]
ELEM_C = {"C": "0.25", "N": "tab:blue", "O": "tab:red", "S": "gold", "P": "tab:orange",
          "F": "tab:green", "Cl": "tab:olive", "Br": "sienna", "I": "purple", "H": "0.8"}


LIGHT = np.array([0.4, 0.35, 0.85])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _shade(tri, fc, ambient=0.45):
    """Lambertian shading from the true face normals.

    Without this the mesh renders as a flat silhouette and you cannot tell a real molecular surface
    from a sphere -- which is exactly the failure mode these sheets exist to catch."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(ln, 1e-12, None)
    diff = np.abs(n @ LIGHT)
    k = (ambient + (1 - ambient) * diff)[:, None]
    fc = np.array(fc, dtype=float).reshape(-1, 4).copy()
    fc[:, :3] = np.clip(fc[:, :3] * k, 0, 1)
    return fc


def _mesh(ax, verts, faces, vals=None, cmap="coolwarm", color=None, alpha=1.0, vmin=-1, vmax=1,
          shade=True):
    if len(faces) == 0:
        return
    tri = verts[faces]
    if vals is not None:
        fv = vals[faces].mean(1)
        fc = plt.get_cmap(cmap)((np.clip(fv, vmin, vmax) - vmin) / (vmax - vmin))
    else:
        fc = np.tile(np.array(matplotlib.colors.to_rgba(color)), (len(tri), 1))
    if shade:
        fc = _shade(tri, fc)
    # depth-sort so nearer triangles are drawn last (matplotlib does not z-sort collections well)
    order = np.argsort(tri[:, :, 2].mean(1))
    pc = Poly3DCollection(tri[order], facecolors=fc[order], edgecolors="none", alpha=alpha)
    ax.add_collection3d(pc)


def _equal(ax, pts, pad=1.0):
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)


def sheet(npz_path, out_png, prot_ref=None):
    z = np.load(npz_path, allow_pickle=False)
    rep = json.loads(str(z["report"]))
    lv, lf, lfe = z["lig_verts"], z["lig_faces"], z["lig_feat"]
    lx, le, lb = z["lig_atom_xyz"], z["lig_atom_elem"], z["lig_bonds"]
    pv, pf, pfe = z["prot_verts"], z["prot_faces"], z["prot_feat"]

    fig = plt.figure(figsize=(22, 10.5))
    fig.suptitle(
        f"{rep['id']}  —  ligand MSMS surface QC   |   {rep['lig_atoms']} atoms, "
        f"{rep['lig_verts']} verts ({rep['verts_per_atom']}/atom), {rep['lig_surface_area_A2']} Å² "
        f"({rep['lig_area_per_heavy_atom']} Å²/atom), elements {', '.join(rep['elements'])}   |   "
        f"{rep['n_contacts']} protein contacts, closest ligand-surface→protein-atom "
        f"{rep['min_lig_surfvert_to_prot_atom']} Å",
        fontsize=12)

    # panel 1: the molecule itself, no surface -- the reference the surface must match
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    for a, b in lb:
        ax.plot(*zip(lx[a], lx[b]), color="0.3", lw=2.0)
    ax.scatter(*lx.T, c=[ELEM_C.get(e, "magenta") for e in le], s=70,
               edgecolors="k", linewidths=0.5, depthshade=False)
    for e in sorted(set(le.tolist())):
        m = le == e
        ax.scatter([], [], [], c=ELEM_C.get(e, "magenta"), label=e, s=40)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    _equal(ax, lv)
    ax.set_title("molecule (atoms + bonds)", fontsize=10)

    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 2, projection="3d")
        _mesh(ax, lv, lf, lfe[:, i], CMAP[i])
        _equal(ax, lv)
        lo, hi, mu = rep["lig_channel_ranges"][["si", "hbond", "charge", "hphob"][i]]
        ax.set_title(f"{CHAN[i]}   [{lo:+.2f}, {hi:+.2f}]  mean {mu:+.2f}", fontsize=10)

    # frame check: draw the actual contact pairs -- the least ambiguous evidence of interdigitation
    ax = fig.add_subplot(2, 4, 6, projection="3d")
    _mesh(ax, pv, pf, color="0.62", alpha=0.30)
    _mesh(ax, lv, lf, color="tab:orange", alpha=1.0)
    _equal(ax, lv, pad=7.0)
    ax.set_title("ligand surface in the protein pocket surface\n(frame check: must interdigitate)",
                 fontsize=10)

    ax = fig.add_subplot(2, 4, 7, projection="3d")
    cp, cl = z["contact_prot_xyz"], z["contact_lig_xyz"]
    for a, b in lb:
        ax.plot(*zip(lx[a], lx[b]), color="0.3", lw=1.6)
    if len(cp):
        sub = np.linspace(0, len(cp) - 1, min(len(cp), 220)).astype(int)
        for k in sub:
            ax.plot(*zip(cl[k], cp[k]), color="tab:green", lw=0.6, alpha=0.5)
        ax.scatter(*cp[sub].T, c="tab:blue", s=9, depthshade=False, label="protein surface atom")
    ax.scatter(*lx.T, c="tab:orange", s=22, edgecolors="k", linewidths=0.3, depthshade=False,
               label="ligand atom")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    _equal(ax, np.vstack([lx, cp]) if len(cp) else lx, pad=2.0)
    ax.set_title(f"training contacts ({rep['n_contacts']} pairs <=5 A)", fontsize=10)

    ax = fig.add_subplot(2, 4, 8)
    for i in range(4):
        ax.hist(lfe[:, i], bins=40, range=(-1, 1), histtype="step", lw=1.8,
                label=f"lig {CHAN[i].split()[0]}", density=True)
    if prot_ref is not None:
        ax.hist(prot_ref[:, 0], bins=40, range=(-1, 1), histtype="stepfilled", alpha=0.18,
                color="k", label="protein si (ref)", density=True)
    ax.set_xlabel("channel value"); ax.set_ylabel("density")
    ax.set_title("ligand channel distributions", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=115)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", default="notebooks/figs")
    ap.add_argument("--prot-ref", default=None, help="npz whose prot_feat is the reference histogram")
    args = ap.parse_args()
    ref = None
    if args.prot_ref and os.path.exists(args.prot_ref):
        ref = np.load(args.prot_ref)["prot_feat"]
    for p in args.npz:
        pid = os.path.basename(p)[:-4]
        print(sheet(p, os.path.join(args.out, f"p7_ligsurf_{pid}.png"), ref), flush=True)


if __name__ == "__main__":
    main()
