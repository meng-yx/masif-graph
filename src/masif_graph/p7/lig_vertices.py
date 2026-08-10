"""Phase-7 — attach the ligand's MSMS surface to its existing Phase-6C atom graph.

Phase 6C emitted ligand npz with `n_vert = 0`: atoms and bonds only. This adds the vertex side —
`vert_feat`, mesh `vv` edges, and `va` vertex-atom edges — so a ligand graph becomes structurally
identical to a protein chain graph and the encoder's `agg_va` pathway stops being identically zero
for ligand atoms.

**Exactly one thing changes relative to Phase 6C.** The atom features, the bond edges, the readout
index (`surf_node_idx` = every heavy atom) and the contact arrays are all carried through untouched,
so the Phase-6C vs Phase-7 comparison is a controlled A/B on the presence of a surface. In
particular `is_surface` stays 1 for every ligand atom rather than being recomputed from vertex
ownership: recomputing it would be more "correct" but would change a second variable, and every
ligand heavy atom is exposed anyway.

Vertex→atom edges are built against the npz's stored `coord` array (heavy-atom coordinates in npz
row order), not against the surface molecule's atom order. The surface is built from a mol with
explicit hydrogens whose ordering need not match, and matching on geometry is order-proof.
"""
from __future__ import annotations

import os

import numpy as np

from masif_graph.graph.hetero import _mesh_edges_from_faces, _vertex_atom_edges

VA_RADIUS = 5.0
VA_KMAX = 8


def surface_arrays(prefix):
    """Load the four npys written by `scripts/p7_lig_surface.py`."""
    return (np.load(prefix + "_verts.npy"), np.load(prefix + "_faces.npy"),
            np.load(prefix + "_normals.npy"), np.load(prefix + "_feat.npy"))


def attach(npz_path, surf_prefix, out_path, va_radius=VA_RADIUS, va_kmax=VA_KMAX) -> dict:
    """Rewrite one ligand npz with its surface attached. Returns a report."""
    z = np.load(npz_path)
    payload = {k: z[k] for k in z.files}
    if int(z["vert_feat"].shape[0]) > 0:
        return {"ok": True, "skipped": "already has vertices"}
    verts, faces, normals, feat = surface_arrays(surf_prefix)
    atom_coords = np.asarray(z["coord"], dtype=np.float64)
    if len(verts) != len(feat):
        return {"ok": False, "err": "verts %d != feat %d" % (len(verts), len(feat))}

    vv = _mesh_edges_from_faces(np.asarray(faces, np.int64), len(verts))
    if len(vv):
        di = verts[vv[:, 0]] - verts[vv[:, 1]]
        vv_dist = np.linalg.norm(di, axis=1)
        vv_cos = np.sum(normals[vv[:, 0]] * normals[vv[:, 1]], axis=1)
    else:
        vv_dist = np.zeros(0); vv_cos = np.zeros(0)

    va_v, va_a = _vertex_atom_edges(verts, atom_coords, radius=va_radius, k_max=va_kmax)
    if len(va_v):
        vec = atom_coords[va_a] - verts[va_v]
        va_dist = np.linalg.norm(vec, axis=1)
        unit = vec / np.clip(va_dist, 1e-9, None)[:, None]
        va_cos = np.sum(normals[va_v] * unit, axis=1)
    else:
        va_dist = np.zeros(0); va_cos = np.zeros(0)

    payload.update(
        vert_feat=np.nan_to_num(feat.astype(np.float32)),
        vv_edge=vv.T.astype(np.int64) if len(vv) else np.zeros((2, 0), np.int64),
        vv_dist=vv_dist.astype(np.float32), vv_cos=vv_cos.astype(np.float32),
        va_v=va_v.astype(np.int64), va_a=va_a.astype(np.int64),
        va_dist=va_dist.astype(np.float32), va_cos=va_cos.astype(np.float32),
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = "%s.part%d.npz" % (out_path, os.getpid())
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)
    n_atom = int(payload["atom_feat"].shape[0])
    return {"ok": True, "n_vert": int(len(verts)), "n_atom": n_atom,
            "n_vv": int(vv.shape[0]) if len(vv) else 0, "n_va": int(len(va_v)),
            "atoms_with_vertex": int(len(np.unique(va_a))),
            "vert_per_atom": round(len(verts) / max(n_atom, 1), 2)}


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="one PDBbind id per line (no 'pl' prefix)")
    ap.add_argument("--src", required=True, help="Phase-6C npz dir")
    ap.add_argument("--dst", required=True, help="Phase-7 npz dir")
    ap.add_argument("--surf-dir", required=True)
    ap.add_argument("--copy-protein", action="store_true",
                    help="also carry the protein npz + contacts across (unchanged)")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    nok = 0
    for pid in ids:
        cid = "pl" + pid
        out = os.path.join(args.dst, "%s__holo__p2.npz" % cid)
        prefix = os.path.join(args.surf_dir, "lig" + pid)
        rep = {"id": pid}
        if os.path.exists(out):
            rep.update(ok=True, skipped="exists")
        elif not os.path.exists(prefix + "_feat.npy"):
            rep.update(ok=False, err="no surface")
        else:
            try:
                rep.update(attach(os.path.join(args.src, "%s__holo__p2.npz" % cid), prefix, out))
            except Exception as exc:                                # noqa: BLE001
                rep.update(ok=False, err="%s: %s" % (type(exc).__name__, exc))
        if rep.get("ok") and args.copy_protein:
            # symlink, not copy: the protein side and the contacts are bit-identical to Phase 6C by
            # design (that is what makes this a controlled A/B), so duplicating ~11 GB buys nothing.
            for suf in ("__holo__p1.npz", "__contacts.npz"):
                s, d = os.path.join(args.src, cid + suf), os.path.join(args.dst, cid + suf)
                if os.path.exists(s) and not os.path.lexists(d):
                    os.symlink(os.path.abspath(s), d)
        nok += bool(rep.get("ok"))
        print(json.dumps(rep), flush=True)
    print(json.dumps({"DONE": True, "ok": nok, "n": len(ids)}))


if __name__ == "__main__":
    main()
