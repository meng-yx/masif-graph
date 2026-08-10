#!/usr/bin/env python
"""Phase-7 — export a protein-ligand pair (both MSMS surfaces) for inspection.

Companion to the Phase-4 pair (`p4_export_graph_viz.py` / `p4_graph_pymol.py`), but aimed at the
question Phase 7 actually needs answered: **is the ligand's surface generated correctly?**

Writes one `.npz` per complex holding, in ONE coordinate frame:
  ligand   verts / faces / normals / feat(si,hbond,charge,hphob) / atom coords / elements / bonds
  protein  verts / faces / feat, cropped to a ball around the ligand, + surface-atom coords
  contacts the (protein surface atom, ligand atom) pairs the training pair is built from

The protein's surface geometry comes from the reference `.ply` and its channels from the Phase-6C
npz; those two are row-aligned by construction (`build_hetero_graph` asserts the `.ply` vertex order
equals the precompute vertex order, and no subsampling is applied), which is what lets the channels
be attached to the mesh without re-running anything.

Both molecules are in the crystal frame — the ligand is at its experimental pose and the protein
surface came from the same crystal — so a frame error would show up immediately as a gap between the
two surfaces. `min_lig_to_prot_surface_dist` in the report is that check.

Usage:
  python scripts/p7_export_pl_viz.py 5czm --out logs/phase7/viz/5czm.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.graph.hetero import load_ply_geometry
from masif_graph.p6.pl_graph import load_ligand

SURF7 = "/work/upthomae/Meng/phase7/lig_surf"
NPZ7 = "/work/upthomae/Meng/phase7/npz_pl"


def _undirected(edge_2xE):
    """(2,E) directed-both-ways -> unique undirected (M,2), same convention as p4_export_graph_viz."""
    e = np.asarray(edge_2xE)
    if e.ndim != 2 or e.shape[1] == 0:
        return np.zeros((0, 2), np.int64)
    return np.unique(np.sort(e.T, axis=1), axis=0)


def _write_lig_pdb(mol, idx, path):
    from rdkit import Chem
    sub = Chem.RWMol(mol)
    conf = mol.GetConformer()
    with open(path, "w") as fh:
        for k, a in enumerate(idx):
            at = mol.GetAtomWithIdx(a)
            p = conf.GetAtomPosition(a)
            fh.write("HETATM%5d %-4s LIG L   1    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                     % (k + 1, (at.GetSymbol() + str(k + 1))[:4], p.x, p.y, p.z, at.GetSymbol()))
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in idx and j in idx:
                fh.write("CONECT%5d%5d\n" % (idx.index(i) + 1, idx.index(j) + 1))
        fh.write("END\n")
    _ = sub


def _write_pocket_pdb(cid, lig_xyz, path, radius=12.0):
    """Whole residues of the protein chain within `radius` of the ligand."""
    from masif_graph.io.reference import PDB_DIR
    src = os.path.join(PDB_DIR, f"{cid}_A.pdb")
    lines, keys, xyz = [], [], []
    with open(src) as fh:
        for line in fh:
            if line[:6] in ("ATOM  ", "HETATM") and len(line) >= 54:
                lines.append(line.rstrip("\n"))
                keys.append(line[21] + line[22:27])
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not lines:
        return 0
    xyz = np.array(xyz)
    near = cKDTree(lig_xyz).query(xyz, k=1)[0] <= radius
    keep_res = {k for k, n in zip(keys, near) if n}
    with open(path, "w") as fh:
        n = 0
        for line, k in zip(lines, keys):
            if k in keep_res:
                fh.write(line + "\n"); n += 1
        fh.write("END\n")
    return n


def export(pid, out, crop=12.0, surf_dir=SURF7, npz_dir=NPZ7, pdbbind="data/pdbbind"):
    cid = f"pl{pid}"
    lv = np.load(f"{surf_dir}/lig{pid}_verts.npy")
    lf = np.load(f"{surf_dir}/lig{pid}_faces.npy")
    ln = np.load(f"{surf_dir}/lig{pid}_normals.npy")
    lfe = np.load(f"{surf_dir}/lig{pid}_feat.npy")

    # Mirror `pl_graph.ligand_graph` EXACTLY: it uses Chem.RemoveHs, which retains hydrogens RDKit
    # considers non-removable (charged N, stereo-defining), so the model's ligand nodes are not
    # strictly heavy atoms. Selecting heavy atoms here instead would silently misalign the contact
    # indices against the graph the encoder actually consumes.
    from rdkit import Chem
    mol = Chem.RemoveHs(load_ligand(pdbbind, pid))
    conf = mol.GetConformer()
    heavy = [a.GetIdx() for a in mol.GetAtoms()]
    lig_xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in heavy])
    lig_el = np.array([mol.GetAtomWithIdx(i).GetSymbol() for i in heavy], dtype="U2")
    hmap = {a: k for k, a in enumerate(heavy)}
    bonds = np.array([[hmap[b.GetBeginAtomIdx()], hmap[b.GetEndAtomIdx()]] for b in mol.GetBonds()
                      if b.GetBeginAtomIdx() in hmap and b.GetEndAtomIdx() in hmap], dtype=np.int64)

    # protein: geometry from the .ply, channels from the npz (row-aligned by construction)
    pv, pf, pn = load_ply_geometry(cid, "A")
    z = np.load(f"{npz_dir}/{cid}__holo__p1.npz")
    pfe = z["vert_feat"]
    if len(pfe) != len(pv):
        raise ValueError(f"{pid}: protein vert_feat {len(pfe)} != ply verts {len(pv)}")
    prot_atom = np.asarray(z["coord"], np.float64)

    keep = np.flatnonzero(cKDTree(lig_xyz).query(pv, k=1)[0] <= crop)
    remap = -np.ones(len(pv), np.int64); remap[keep] = np.arange(len(keep))
    fmask = np.all(np.isin(pf, keep), axis=1)
    pf_c = remap[pf[fmask]]

    cz = np.load(f"{npz_dir}/{cid}__contacts.npz")
    pos = cz["pos"].reshape(-1, 2)

    # the ligand's GRAPH edges, straight from the npz the encoder consumes, so the viewer shows the
    # actual model input rather than a re-derivation of it
    lz = np.load(f"{npz_dir}/{cid}__holo__p2.npz")
    lig_vv = _undirected(lz["vv_edge"])
    lig_va = np.stack([lz["va_v"], lz["va_a"]], 1) if len(lz["va_v"]) else np.zeros((0, 2), np.int64)
    lig_aa = _undirected(lz["aa_edge"])

    rep = {"id": pid, "lig_atoms": int(len(lig_xyz)), "lig_verts": int(len(lv)),
           "lig_faces": int(len(lf)), "verts_per_atom": round(len(lv) / len(lig_xyz), 2),
           "elements": sorted(set(lig_el.tolist())),
           "prot_verts_total": int(len(pv)), "prot_verts_cropped": int(len(keep)),
           "n_contacts": int(len(pos)),
           "lig_channel_ranges": {n: [round(float(lfe[:, i].min()), 3), round(float(lfe[:, i].max()), 3),
                                      round(float(lfe[:, i].mean()), 3)]
                                  for i, n in enumerate(["si", "hbond", "charge", "hphob"])}}
    # frame check: the two surfaces must interdigitate, not sit apart
    d_lv_pa = cKDTree(prot_atom).query(lv, k=1)[0]
    rep["min_lig_surfvert_to_prot_atom"] = round(float(d_lv_pa.min()), 2)
    rep["frac_lig_surf_within_5A_of_protein"] = round(float((d_lv_pa <= 5.0).mean()), 3)
    # surface area of the ligand mesh (a blob would be ~spherical; report it for eyeballing)
    tri = lv[lf]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    rep["lig_surface_area_A2"] = round(float(area), 1)
    rep["lig_area_per_heavy_atom"] = round(float(area / len(lig_xyz)), 1)

    # companion PDBs so PyMOL can show real sticks/cartoon without anything else from the cluster
    stem = out[:-4]
    _write_lig_pdb(mol, heavy, stem + "_ligand.pdb")
    n_pocket = _write_pocket_pdb(cid, lig_xyz, stem + "_pocket.pdb", radius=crop)
    rep["pocket_pdb_atoms"] = n_pocket

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, lig_verts=lv, lig_faces=lf, lig_normals=ln, lig_feat=lfe,
                        lig_vv_edge=lig_vv, lig_va_edge=lig_va, lig_aa_edge=lig_aa,
                        lig_atom_xyz=lig_xyz, lig_atom_elem=lig_el, lig_bonds=bonds,
                        prot_verts=pv[keep], prot_faces=pf_c, prot_feat=pfe[keep],
                        prot_atom_xyz=prot_atom, contacts=pos,
                        contact_prot_xyz=prot_atom[pos[:, 0]] if len(pos) else np.zeros((0, 3)),
                        contact_lig_xyz=lig_xyz[pos[:, 1]] if len(pos) else np.zeros((0, 3)),
                        report=json.dumps(rep))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--outdir", default="logs/phase7/viz")
    ap.add_argument("--crop", type=float, default=12.0)
    args = ap.parse_args()
    for pid in args.ids:
        try:
            r = export(pid, os.path.join(args.outdir, f"{pid}.npz"), crop=args.crop)
        except Exception as exc:                                    # noqa: BLE001
            r = {"id": pid, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
