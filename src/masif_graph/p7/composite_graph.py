"""Phase-7 S3 — assemble a composite (protein + drug) hetero graph for the axis-3 query.

The Phase-6C neosurface query was `z(protein surface atoms near the drug) ⊕ z(drug atoms)`, drawn
from two graphs whose surfaces were computed independently — so the protein's pocket was an empty
cavity and the joint shape a partner recognises was never represented. Here the two molecules share
ONE atom set and ONE surface (built by `scripts/p7_composite_surface.py`), which is what a
neosurface actually is.

D7-4: inference only. Never a training input.

The query patch keeps the Phase-6C *definition* — surface atoms within `lig_radius` of the drug,
plus the drug's own atoms — so the axis-3 comparison is a controlled swap of the representation
underneath the same patch rule.

Output npz uses the standard chain-graph contract plus two extras: `readout_is_lig` (which readout
rows are drug atoms) and `coord` (readout atom coordinates, for the near-drug test at eval time).
"""
from __future__ import annotations

import os

import numpy as np

from masif_graph.graph.hetero import _mesh_edges_from_faces, _vertex_atom_edges
from masif_graph.io.reference import PDB_DIR, parse_heavy_atoms

VA_RADIUS = 5.0
VA_KMAX = 8


def build(sys_id, chain, pid, neosurf_npz_dir, surf_dir, out_dir,
          va_radius=VA_RADIUS, va_kmax=VA_KMAX) -> dict:
    """sys_id like 'nb6QTL', chain like 'A', pid 'p1'|'p2' (which subunit owns this composite)."""
    prot = np.load(os.path.join(neosurf_npz_dir, "%s__holo__%s.npz" % (sys_id, pid)))
    lig = np.load(os.path.join(neosurf_npz_dir, "%s__lig.npz" % sys_id))
    pref = os.path.join(surf_dir, "cmp_%s_%s" % (sys_id, chain))
    verts = np.load(pref + "_verts.npy")
    faces = np.load(pref + "_faces.npy")
    normals = np.load(pref + "_normals.npy")
    feat = np.load(pref + "_feat.npy")

    # protein atom coordinates for ALL atoms: the npz only stores surface-atom coords, and a
    # composite vertex can legitimately sit within 5 A of a sub-surface atom.
    pdb_path = os.path.join(PDB_DIR, "%s_%s.pdb" % (sys_id, chain))
    coords_p, _el, _rid, _nm = parse_heavy_atoms(pdb_path)
    af_p = prot["atom_feat"]
    if len(coords_p) != af_p.shape[0]:
        return {"ok": False, "err": "atom count %d != npz %d" % (len(coords_p), af_p.shape[0])}
    coords_l = np.asarray(lig["coord"], dtype=np.float64)
    af_l = lig["atom_feat"]
    n_p, n_l = af_p.shape[0], af_l.shape[0]

    atom_feat = np.concatenate([af_p, af_l], 0).astype(np.float32)
    coords = np.concatenate([coords_p, coords_l], 0)
    aa_edge = np.concatenate([prot["aa_edge"], lig["aa_edge"] + n_p], axis=1).astype(np.int64) \
        if lig["aa_edge"].shape[1] else prot["aa_edge"].astype(np.int64)
    aa_order = np.concatenate([prot["aa_order"], lig["aa_order"]], 0).astype(np.float32)
    aa_rot = np.concatenate([prot["aa_rot"], lig["aa_rot"]], 0).astype(np.float32)

    vv = _mesh_edges_from_faces(np.asarray(faces, np.int64), len(verts))
    if len(vv):
        vv_dist = np.linalg.norm(verts[vv[:, 0]] - verts[vv[:, 1]], axis=1)
        vv_cos = np.sum(normals[vv[:, 0]] * normals[vv[:, 1]], axis=1)
    else:
        vv_dist = np.zeros(0); vv_cos = np.zeros(0)

    va_v, va_a = _vertex_atom_edges(verts, coords, radius=va_radius, k_max=va_kmax)
    if len(va_v):
        vec = coords[va_a] - verts[va_v]
        va_dist = np.linalg.norm(vec, axis=1)
        unit = vec / np.clip(va_dist, 1e-9, None)[:, None]
        va_cos = np.sum(normals[va_v] * unit, axis=1)
    else:
        return {"ok": False, "err": "no vertex-atom edges"}

    readout = np.unique(va_a)                       # atoms that own at least one composite vertex
    n_surf = len(readout)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s_%s__cmp.npz" % (sys_id, chain))
    tmp = "%s.part%d.npz" % (out, os.getpid())
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            atom_feat=atom_feat, aa_edge=aa_edge, aa_order=aa_order, aa_rot=aa_rot,
            vert_feat=np.nan_to_num(feat.astype(np.float32)),
            vv_edge=vv.T.astype(np.int64) if len(vv) else np.zeros((2, 0), np.int64),
            vv_dist=vv_dist.astype(np.float32), vv_cos=vv_cos.astype(np.float32),
            va_v=va_v.astype(np.int64), va_a=va_a.astype(np.int64),
            va_dist=va_dist.astype(np.float32), va_cos=va_cos.astype(np.float32),
            surf_node_idx=readout.astype(np.int64), n_surf=np.int64(n_surf),
            desc_straight=np.zeros((n_surf, 80), np.float32),
            desc_flipped=np.zeros((n_surf, 80), np.float32),
            coord=coords[readout].astype(np.float32),
            readout_is_lig=(readout >= n_p).astype(np.int64),
            keys=np.array(["c:%d:%d" % (i, r) for i, r in enumerate(readout)], dtype="S24"),
        )
    os.replace(tmp, out)
    return {"ok": True, "sys": sys_id, "chain": chain, "pid": pid,
            "n_atom_prot": n_p, "n_atom_lig": n_l, "n_vert": int(len(verts)),
            "n_readout": int(n_surf), "n_readout_lig": int((readout >= n_p).sum()),
            "n_va": int(len(va_v))}


def main():
    import argparse
    import json

    from masif_graph.p6.neosurf import parse_benchmark

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--neosurf-npz", required=True, help="Phase-6C neosurf npz dir")
    ap.add_argument("--surf-dir", required=True, help="Phase-7 composite surface npys")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    nok = 0
    for e in parse_benchmark(args.bench):
        for pid, ch in (("p1", e["c1"]), ("p2", e["c2"])):
            if not os.path.exists(os.path.join(args.surf_dir,
                                               "cmp_%s_%s_feat.npy" % (e["sys"], ch))):
                print(json.dumps({"sys": e["sys"], "chain": ch, "ok": False, "err": "no surface"}))
                continue
            try:
                rep = build(e["sys"], ch, pid, args.neosurf_npz, args.surf_dir, args.out)
            except Exception as exc:                                # noqa: BLE001
                rep = {"sys": e["sys"], "chain": ch, "ok": False,
                       "err": "%s: %s" % (type(exc).__name__, exc)}
            nok += bool(rep.get("ok"))
            print(json.dumps(rep), flush=True)
    print(json.dumps({"DONE": True, "ok": nok}))


if __name__ == "__main__":
    main()
