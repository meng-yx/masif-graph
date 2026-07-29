"""Export the Phase-4 HeteroSurfaceGraph (the actual GNN input) to a flat .npz for PyMOL.

Runs in the `masif-graph` conda env (needs the reference surfaces/PDBs staged locally). It rebuilds
the exact per-chain graph the encoder consumes via `build_hetero_graph`, so node coordinates are
perfectly aligned to the edge index arrays, then dumps node coords + edges (both chains concatenated
into one coordinate frame) for the companion PyMOL script `p4_graph_pymol.py`.

Usage:
  conda activate masif-graph
  python scripts/p4_export_graph_viz.py 1A0H_E_D --out /tmp/1A0H_graph.npz
  python scripts/p4_export_graph_viz.py 1A0H_E_D --state af3 --max-vert 3000 --out g.npz

Then in PyMOL:
  pymol scripts/p4_graph_pymol.py -- /tmp/1A0H_graph.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from masif_graph.io.reference import load_complex, complex_is_available, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.graph.hetero import build_hetero_graph
from masif_graph.p4.precompute import af3_state_id


def _undirected(edge_2xE: np.ndarray) -> np.ndarray:
    """(2,E) directed-both-ways -> unique undirected (M,2) with row0<row1."""
    if edge_2xE.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    e = np.sort(edge_2xE.T, axis=1)
    return np.unique(e, axis=0)


def build_one_chain(ch, va_radius, va_kmax, max_vert):
    surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                               ch.desc_straight, ch.desc_flipped, ops=("mean",))
    pdb = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
    g = build_hetero_graph(ch, surf, pdb, va_radius=va_radius, va_kmax=va_kmax, max_vert=max_vert)
    return g, pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("complex_id", help="e.g. 1A0H_E_D (PDBID_chainA_chainB, from data/lists)")
    ap.add_argument("--state", choices=["holo", "af3"], default="holo",
                    help="holo crystal (default) or the Phase-3 AF3 model of the same complex")
    ap.add_argument("--va-radius", type=float, default=5.0, help="vertex->atom ball radius (Å); training default 5.0")
    ap.add_argument("--va-kmax", type=int, default=8, help="max atoms per vertex; training default 8")
    ap.add_argument("--max-vert", type=int, default=None,
                    help="subsample vertices to this many for a snappier session (default: all)")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    state_id = args.complex_id if args.state == "holo" else af3_state_id(args.complex_id)
    if not complex_is_available(state_id):
        raise SystemExit(f"complex {state_id} not available in the staged reference data")
    p1, p2 = load_complex(state_id)

    atom_xyz, atom_surf, vert_xyz = [], [], []
    aa_e, vv_e, va_e = [], [], []
    pdb_files, chain_spans = [], []
    a_off = v_off = 0
    for ch in (p1, p2):
        g, pdb = build_one_chain(ch, args.va_radius, args.va_kmax, args.max_vert)
        na, nv = g.atom_coords.shape[0], g.vert_coords.shape[0]
        atom_xyz.append(np.asarray(g.atom_coords, np.float32))
        atom_surf.append(np.asarray(g.is_surface_atom, bool))
        vert_xyz.append(np.asarray(g.vert_coords, np.float32))
        aa_e.append(_undirected(g.aa_edge) + a_off)                      # atom idx
        vv_e.append(_undirected(g.vv_edge) + v_off)                      # vertex idx
        va_pairs = np.unique(np.stack([g.va_v, g.va_a], axis=1), axis=0) if g.va_v.shape[0] else np.zeros((0, 2), np.int64)
        va_e.append(va_pairs + np.array([v_off, a_off]))                 # [vertex idx, atom idx]
        pdb_files.append(pdb)
        chain_spans.append({"chain": f"{ch.pdb_id}_{ch.chain_ids}", "n_atom": int(na), "n_vert": int(nv),
                            "n_surf_atom": int(atom_surf[-1].sum())})
        a_off += na
        v_off += nv

    atom_xyz = np.concatenate(atom_xyz, 0)
    atom_surf = np.concatenate(atom_surf, 0)
    vert_xyz = np.concatenate(vert_xyz, 0)
    aa_edges = np.concatenate(aa_e, 0).astype(np.int32)
    vv_edges = np.concatenate(vv_e, 0).astype(np.int32)
    va_edges = np.concatenate(va_e, 0).astype(np.int32)

    meta = {"complex_id": args.complex_id, "state": args.state, "state_id": state_id,
            "va_radius": args.va_radius, "va_kmax": args.va_kmax, "max_vert": args.max_vert,
            "chains": chain_spans}
    np.savez_compressed(args.out, atom_xyz=atom_xyz, atom_is_surf=atom_surf, vert_xyz=vert_xyz,
                        aa_edges=aa_edges, vv_edges=vv_edges, va_edges=va_edges,
                        pdb_files=np.array(pdb_files), meta=json.dumps(meta))
    print(f"wrote {args.out}")
    print(f"  atom nodes   : {atom_xyz.shape[0]:6d}  ({atom_surf.sum()} surface, {(~atom_surf).sum()} buried)")
    print(f"  vertex nodes : {vert_xyz.shape[0]:6d}")
    print(f"  aa edges     : {aa_edges.shape[0]:6d}  (covalent)")
    print(f"  vv edges     : {vv_edges.shape[0]:6d}  (mesh)")
    print(f"  va edges     : {va_edges.shape[0]:6d}  (vertex-atom)")
    print(f"  source PDBs  : {', '.join(pdb_files)}")


if __name__ == "__main__":
    main()
