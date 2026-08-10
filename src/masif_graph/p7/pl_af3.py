"""Phase-7 S5 — the AF3-apo protein side of a protein-ligand complex.

The project's north star is holo→apo robustness. Phase 5 tested it for protein–protein; Phase 7
finally tests it on the ligand axis. Per the standing project rule, **the protein varies and the
ligand stays at its experimental pose** — an apo ligand conformation is not a meaningful object, so
there is nothing to predict for the ligand side.

`masif_graph.af3.build_pdb` already relabels an AF3 model to holo residue numbering *and* superposes
it into the holo frame by CA correspondence, so the crystal ligand's coordinates remain valid
against the AF3 protein without any new alignment code (Phase-7 risk R5, closed by reuse).

Emitted next to the holo artefacts, in the state convention `p5.retrieval_bench` already uses:
    {cid}__af3__p1.npz        AF3 protein graph (26-D, same builder as holo)
    {cid}__af3contacts.npz    contacts recomputed on the AF3 structure vs the SAME ligand pose

Contacts must be recomputed rather than reused: the AF3 surface has its own atom rows, and — this
is the measurement, not a nuisance — a predicted apo backbone genuinely contacts the ligand
differently. `retention` records how much of the holo contact set survives.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.graph.hetero import build_hetero_graph
from masif_graph.io.reference import PDB_DIR, PRECOMP_DIR
from masif_graph.p6.atoms import DIM
from masif_graph.p6.pl_graph import (POS_CUT, POS_SC_CUT, _ChainShim, load_ligand, ligand_graph,
                                     save_protein_npz)
from masif_graph.surface.atoms import build_surface_atoms


def build_af3(pid, out_dir, pdbbind_dir, va_radius=5.0, va_kmax=8) -> dict:
    """Build the AF3 protein graph + its contacts against the experimental ligand pose."""
    cid = f"pl{pid}"
    sid = f"{cid}AF_A"                      # the id the .sif pipeline was run under
    pc = os.path.join(PRECOMP_DIR, sid)
    xyz = [np.load(os.path.join(pc, f"p1_{a}.npy")) for a in ("X", "Y", "Z")]
    verts = np.column_stack(xyz).astype(np.float64)
    pdb_path = os.path.join(PDB_DIR, f"{sid}.pdb")
    chain = _ChainShim(sid, f"{cid}AF", "A", verts, pdb_path)

    zero = np.zeros((len(verts), 80), np.float32)
    surf = build_surface_atoms(verts, chain.atom_coords, chain.atom_element, chain.atom_resid,
                               zero, zero, ops=("mean",))
    g = build_hetero_graph(chain, surf, pdb_path, va_radius=va_radius, va_kmax=va_kmax,
                           unified_atom_feat=True)
    if g.atom_feat.shape[1] != DIM:
        raise ValueError(f"{pid}: af3 atom_feat dim {g.atom_feat.shape[1]} != {DIM}")

    mol = load_ligand(pdbbind_dir, pid)
    if mol is None:
        return {"pid": pid, "ok": False, "err": "ligand unreadable"}
    _lf, lcoord, _e, _o, _r = ligand_graph(mol)

    tree = cKDTree(lcoord)
    pos, pos_sc = [], []
    for r, c in enumerate(surf.coord):
        for j in tree.query_ball_point(c, POS_CUT):
            pos.append((r, j))
            if np.linalg.norm(lcoord[j] - c) <= POS_SC_CUT:
                pos_sc.append((r, j))

    os.makedirs(out_dir, exist_ok=True)
    save_protein_npz(os.path.join(out_dir, f"{cid}__af3__p1.npz"), chain, surf, g)
    tmp = os.path.join(out_dir, f"{cid}__af3contacts.npz.part.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, pos=np.asarray(pos, np.int64).reshape(-1, 2),
                            pos_sc=np.asarray(pos_sc, np.int64).reshape(-1, 2))
    os.replace(tmp, os.path.join(out_dir, f"{cid}__af3contacts.npz"))

    rep = {"pid": pid, "ok": True, "n_atom": int(chain.n_atom), "n_vert": int(len(verts)),
           "n_surf": int(g.n_surf), "n_pos": len(pos), "n_pos_sc": len(pos_sc),
           "n_iface_prot": int(len({p[0] for p in pos})),
           "n_iface_lig": int(len({p[1] for p in pos}))}
    holo_c = os.path.join(out_dir, f"{cid}__contacts.npz")
    if os.path.exists(holo_c):
        nh = np.load(holo_c)["pos"].reshape(-1, 2).shape[0]
        rep["holo_n_pos"] = int(nh)
        rep["contact_ratio_af3_over_holo"] = round(len(pos) / max(nh, 1), 3)
    if len(pos) == 0:
        rep.update(ok=False, err="AF3 model makes no contact with the crystal ligand pose")
    return rep


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdbbind", default="/scratch/ymeng/masif-graph/data/pdbbind")
    args = ap.parse_args()
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    nok = 0
    for pid in ids:
        pid = pid[2:] if pid.startswith("pl") else pid
        if os.path.exists(os.path.join(args.out, f"pl{pid}__af3contacts.npz")):
            print(json.dumps({"pid": pid, "ok": True, "skipped": "cached"}), flush=True)
            nok += 1
            continue
        try:
            r = build_af3(pid, args.out, args.pdbbind)
        except Exception as exc:                                    # noqa: BLE001
            r = {"pid": pid, "ok": False, "err": f"{type(exc).__name__}: {exc}"}
        nok += bool(r.get("ok"))
        print(json.dumps(r), flush=True)
    print(json.dumps({"DONE": True, "ok": nok, "n": len(ids)}))


if __name__ == "__main__":
    main()
