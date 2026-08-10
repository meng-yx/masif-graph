#!/usr/bin/env python
"""Export a complete TRAINING PAIR (both partners, in full) for the PyMOL viewer.

A training example is always a pair of interacting partners — protein A + protein B for PPI, or
protein A + ligand B for protein-ligand — so both sides are exported whole. Nothing is cropped: the
protein's entire atom set, entire vertex set and entire edge set go out, because the GNN consumes
all of it.

Everything written here is read straight out of the npz the encoder consumes (plus the reference
`.ply` for vertex coordinates, which the npz does not store), so the viewer shows the actual model
input rather than a re-derivation of it.

Companion `.pdb` holds **exactly the atoms that are graph nodes**, in graph-node order: chain A =
left, chain L (or B) = right. So atom index i in the npz is the i-th atom of that chain in the PDB.

Usage:
  python scripts/p7_export_pair_viz.py 6ibk 4ivc --outdir logs/phase7/viz
  python scripts/p7_export_pair_viz.py --kind ppi 1A99_C_D --outdir logs/phase7/viz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from masif_graph.graph.hetero import load_ply_geometry
from masif_graph.io.reference import PDB_DIR, parse_heavy_atoms

ATOM_FEATURES = [                      # the 26-D unified atom vector (src/masif_graph/p6/atoms.py)
    "elem_C", "elem_N", "elem_O", "elem_S", "elem_P", "elem_F", "elem_Cl", "elem_Br", "elem_I",
    "elem_other", "is_ligand", "is_backbone", "aromatic", "degree", "is_surface", "in_ring",
    "hyb_sp", "hyb_sp2", "hyb_sp3", "hbond_donor", "hbond_acceptor", "formal_charge",
    "flex_depth", "electronegativity", "valence", "covalent_radius"]
VERT_FEATURES = ["si", "hbond", "charge", "hphob"]
# edge features the GNN consumes, per edge type (dataset.py: D_AA=5, D_VV=9, D_VA=9)
EDGE_FEATURES = {"aa": ["bond_order(one-hot x4)", "sidechain_rotatable"],
                 "vv": ["length -> RBF(8, 0-4 A)", "cos(normal_i, normal_j)"],
                 "va": ["distance -> RBF(8, 0-5 A)", "cos(normal_v, unit(atom - vertex))"]}
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other"]


def _undirected(e2):
    e = np.asarray(e2)
    if e.ndim != 2 or e.shape[1] == 0:
        return np.zeros((0, 2), np.int64)
    return np.unique(np.sort(e.T, axis=1), axis=0)


def _side_from_npz(npz_path, vert_xyz, faces, normals, atom_xyz, atom_elem):
    z = np.load(npz_path)
    # take the stored canonical undirected edge list AS-IS: vv_dist / vv_cos are aligned to it
    # column-for-column, and re-uniquing it would risk silently reordering the features
    vv = np.asarray(z["vv_edge"]).T.astype(np.int64) if z["vv_edge"].shape[1] else \
        np.zeros((0, 2), np.int64)
    if len(vv) != len(z["vv_dist"]):
        raise ValueError("vv_edge %d != vv_dist %d" % (len(vv), len(z["vv_dist"])))
    va = (np.stack([z["va_v"], z["va_a"]], 1).astype(np.int64)
          if len(z["va_v"]) else np.zeros((0, 2), np.int64))
    aa = _undirected(z["aa_edge"])
    # aa_order/aa_rot are stored per DIRECTED edge; take the first occurrence per undirected pair
    dir_e = np.asarray(z["aa_edge"])
    key = {}
    for k in range(dir_e.shape[1]):
        key.setdefault((min(dir_e[0, k], dir_e[1, k]), max(dir_e[0, k], dir_e[1, k])), k)
    idx = np.array([key[(a, b)] for a, b in aa], np.int64) if len(aa) else np.zeros(0, np.int64)
    return {
        "atom_xyz": atom_xyz.astype(np.float32),
        "atom_feat": z["atom_feat"].astype(np.float32),
        "atom_elem": np.asarray(atom_elem, dtype="U2"),
        "vert_xyz": vert_xyz.astype(np.float32),
        "vert_normal": normals.astype(np.float32),
        "vert_feat": z["vert_feat"].astype(np.float32),
        "faces": np.asarray(faces, np.int64),
        "aa_edge": aa, "aa_order": z["aa_order"][idx] if len(idx) else np.zeros((0, 4), np.float32),
        "aa_rot": z["aa_rot"][idx] if len(idx) else np.zeros(0, np.float32),
        "vv_edge": vv,
        "vv_dist": np.asarray(z["vv_dist"], np.float32),
        "vv_cos": np.asarray(z["vv_cos"], np.float32),
        "va_edge": va,
        "va_dist": np.asarray(z["va_dist"], np.float32),
        "va_cos": np.asarray(z["va_cos"], np.float32),
        "surf_node_idx": z["surf_node_idx"].astype(np.int64),
    }


def _pdb_lines(atom_xyz, atom_elem, chain, resname, start_serial=1, resid_of=None):
    out = []
    for i, (p, e) in enumerate(zip(atom_xyz, atom_elem)):
        rn, ri = (resname, 1) if resid_of is None else resid_of(i)
        rec = "ATOM  " if resid_of is not None else "HETATM"
        out.append("%s%5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                   % (rec, start_serial + i, (str(e) + str(i + 1))[:4], rn[:3], chain, ri,
                      p[0], p[1], p[2], str(e)[:2]))
    return out


def export_pl(pid, outdir, surf_dir, npz_dir, pdbbind="data/pdbbind"):
    """protein A + ligand B."""
    cid = f"pl{pid}"
    # ---- left: the protein, IN FULL ----
    pv, pf, pn = load_ply_geometry(cid, "A")
    pdb_path = os.path.join(PDB_DIR, f"{cid}_A.pdb")
    pxyz, pelem, presid, pname = parse_heavy_atoms(pdb_path)
    left = _side_from_npz(f"{npz_dir}/{cid}__holo__p1.npz", pv, pf, pn, pxyz, pelem)
    if len(left["vert_feat"]) != len(pv):
        raise ValueError(f"{pid}: protein vert_feat {len(left['vert_feat'])} != ply verts {len(pv)}")
    if len(left["atom_feat"]) != len(pxyz):
        raise ValueError(f"{pid}: protein atom_feat {len(left['atom_feat'])} != pdb atoms {len(pxyz)}")

    # ---- right: the ligand, IN FULL ----
    lv = np.load(f"{surf_dir}/lig{pid}_verts.npy")
    lf = np.load(f"{surf_dir}/lig{pid}_faces.npy")
    ln = np.load(f"{surf_dir}/lig{pid}_normals.npy")
    lz = np.load(f"{npz_dir}/{cid}__holo__p2.npz")
    lxyz = np.asarray(lz["coord"], np.float64)
    from rdkit import Chem
    from masif_graph.p6.pl_graph import load_ligand
    mol = Chem.RemoveHs(load_ligand(pdbbind, pid))
    lelem = np.array([a.GetSymbol() for a in mol.GetAtoms()], dtype="U2")
    right = _side_from_npz(f"{npz_dir}/{cid}__holo__p2.npz", lv, lf, ln, lxyz, lelem)

    pos = np.load(f"{npz_dir}/{cid}__contacts.npz")["pos"].reshape(-1, 2)
    return _finish(cid, outdir, left, right, pos, "pl", presid, pname,
                   right_chain="L", right_resname="LIG")


def export_ppi(cid, outdir, npz_dir):
    """protein A + protein B (the .ply files must still exist for both chains)."""
    pdb, c1, c2 = cid.split("_")
    sides, meta = [], []
    for chain, pid in ((c1, "p1"), (c2, "p2")):
        v, f, n = load_ply_geometry(pdb, chain)
        xyz, elem, resid, name = parse_heavy_atoms(os.path.join(PDB_DIR, f"{pdb}_{chain}.pdb"))
        sides.append(_side_from_npz(f"{npz_dir}/{cid}__holo__{pid}.npz", v, f, n, xyz, elem))
        meta.append((resid, name))
    pos = np.load(f"{npz_dir}/{cid}__contacts.npz")["pos"].reshape(-1, 2)
    return _finish(cid, outdir, sides[0], sides[1], pos, "ppi", meta[0][0], meta[0][1],
                   right_chain="B", right_resname=None, right_resid=meta[1][0], right_name=meta[1][1])


def _finish(cid, outdir, left, right, pos, kind, lresid, lname,
            right_chain="L", right_resname="LIG", right_resid=None, right_name=None):
    os.makedirs(outdir, exist_ok=True)
    # contacts are (left surface ROW, right surface ROW); map to atom indices for drawing
    ca = left["surf_node_idx"][pos[:, 0]] if len(pos) else np.zeros(0, np.int64)
    cb = right["surf_node_idx"][pos[:, 1]] if len(pos) else np.zeros(0, np.int64)

    # ---- the PDB: exactly the graph's atom nodes, in graph-node order ----
    def prot_resid(resids, names):
        def f(i):
            _c, seq, rn = str(resids[i]).split(":")
            return rn, int(seq)
        return f
    lines = _pdb_lines(left["atom_xyz"], left["atom_elem"], "A", "UNK",
                       resid_of=prot_resid(lresid, lname))
    n0 = len(lines)
    if right_resid is not None:
        lines += _pdb_lines(right["atom_xyz"], right["atom_elem"], right_chain, "UNK",
                            start_serial=n0 + 1, resid_of=prot_resid(right_resid, right_name))
    else:
        lines += _pdb_lines(right["atom_xyz"], right["atom_elem"], right_chain, right_resname,
                            start_serial=n0 + 1)
        for a, b in right["aa_edge"]:
            lines.append("CONECT%5d%5d" % (n0 + a + 1, n0 + b + 1))
    with open(os.path.join(outdir, f"{cid}.pdb"), "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")

    meta = {"id": cid, "kind": kind, "atom_features": ATOM_FEATURES,
            "vert_features": VERT_FEATURES, "edge_features": EDGE_FEATURES, "elements": ELEMENTS,
            "left_label": "protein A", "right_label": "ligand B" if kind == "pl" else "protein B",
            "left": {"atoms": int(len(left["atom_xyz"])), "verts": int(len(left["vert_xyz"])),
                     "faces": int(len(left["faces"])), "surf_atoms": int(len(left["surf_node_idx"])),
                     "aa": int(len(left["aa_edge"])), "vv": int(len(left["vv_edge"])),
                     "va": int(len(left["va_edge"]))},
            "right": {"atoms": int(len(right["atom_xyz"])), "verts": int(len(right["vert_xyz"])),
                      "faces": int(len(right["faces"])),
                      "surf_atoms": int(len(right["surf_node_idx"])),
                      "aa": int(len(right["aa_edge"])), "vv": int(len(right["vv_edge"])),
                      "va": int(len(right["va_edge"]))},
            "n_contacts": int(len(pos))}

    payload = {"contacts_atom": np.stack([ca, cb], 1) if len(ca) else np.zeros((0, 2), np.int64),
               "meta": json.dumps(meta)}
    for tag, side in (("left", left), ("right", right)):
        for k, v in side.items():
            payload[f"{tag}_{k}"] = v
    np.savez_compressed(os.path.join(outdir, f"{cid}.npz"), **payload)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--kind", choices=["pl", "ppi"], default="pl")
    ap.add_argument("--outdir", default="logs/phase7/viz")
    ap.add_argument("--surf-dir", default="/work/upthomae/Meng/phase7/lig_surf")
    ap.add_argument("--npz-dir", default=None)
    args = ap.parse_args()
    npz = args.npz_dir or ("/work/upthomae/Meng/phase7/npz_pl" if args.kind == "pl"
                           else "/work/upthomae/Meng/phase6C/npz_ppi")
    for i in args.ids:
        try:
            m = (export_pl(i, args.outdir, args.surf_dir, npz) if args.kind == "pl"
                 else export_ppi(i, args.outdir, npz))
        except Exception as exc:                                    # noqa: BLE001
            m = {"id": i, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
