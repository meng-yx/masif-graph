"""M3 preprocessing PASS 1 (run in the `masif-graph` env — needs biotite).

Per (complex, state, chain) save the chem-graph tensors + per-vertex frozen descriptors + the
vertex->surface-atom pooling map + surface-atom identity keys, as a cross-env `.npz`. Per complex
save the holo sc-contact positive pairs. PASS 2 (`precompute_surf.py`, atomsurf env) adds the
DiffusionNet operators; the training dataset joins them by filename.

Output: <out>/{cid}__{state}__{pid}.npz  and  <out>/{cid}__contacts.npz
state in {"holo","af3"}; af3 id = {PDBID}AF_{C1}_{C2}.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from masif_graph.io.reference import load_complex, complex_is_available, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts
from masif_graph.m3.chem_graph import build_chem_graph
from masif_graph.graph.dataset import graph_to_tensors


def af3_id(holo_id):
    p, c1, c2 = holo_id.split("_")
    return f"{p}AF_{c1}_{c2}"


def _keys(chain, surf):
    ks = []
    for r in surf.atom_idx:
        chn, seq, _rn = chain.atom_resid[r].split(":")
        ks.append(f"{chn}:{seq}:{chain.atom_name[r]}")
    # fixed-width byte strings (NOT an object array) so it loads cross-numpy-version (no pickle).
    return np.array(ks, dtype="S24")


def save_chain(cid, state_id, pid, chain, out_dir):
    surf = build_surface_atoms(chain.verts, chain.atom_coords, chain.atom_element, chain.atom_resid,
                               chain.desc_straight, chain.desc_flipped, ops=("mean",))
    g = build_chem_graph(chain, surf, os.path.join(PDB_DIR, f"{chain.pdb_id}_{chain.chain_ids}.pdb"))
    gt = graph_to_tensors(g, use_rotatable=True, device="cpu")
    out = os.path.join(out_dir, f"{cid}__{'holo' if state_id == cid else 'af3'}__{pid}.npz")
    np.savez_compressed(
        out,
        desc_straight=chain.desc_straight.astype(np.float32),   # (V,80) per-vertex frozen desc
        desc_flipped=chain.desc_flipped.astype(np.float32),
        vertex_surf_idx=surf.vertex_surf_idx.astype(np.int64),  # (V,) vertex->surface-atom row
        n_surf=np.int64(surf.coord.shape[0]),
        node_feat=gt["node_feat"].numpy().astype(np.float32),
        cov_edge=gt["cov_edge"].numpy().astype(np.int64),
        cov_feat=gt["cov_feat"].numpy().astype(np.float32),
        surf_idx=gt["surf_idx"].numpy().astype(np.int64),
        keys=_keys(chain, surf),
    )
    return surf


def process_complex(cid, out_dir):
    done = {}
    for state_id, state in ((cid, "holo"), (af3_id(cid), "af3")):
        if not complex_is_available(state_id):
            continue
        p1, p2 = load_complex(state_id)
        s1 = save_chain(cid, state_id, "p1", p1, out_dir)
        s2 = save_chain(cid, state_id, "p2", p2, out_dir)
        done[state] = (p1, p2, s1, s2)
    # holo contacts (complementarity positives), in holo surface-atom rows
    if "holo" in done:
        p1, p2, s1, s2 = done["holo"]
        vp, _ = vertex_contacts(p1.verts, p2.verts, pos_cutoff=1.0, sc1=p1.sc, sc_band=(0.5, 1.0))
        pos = atom_positives_from_vertex_contacts(vp, s1.vertex_surf_idx, s2.vertex_surf_idx)
        np.savez_compressed(os.path.join(out_dir, f"{cid}__contacts.npz"),
                            pos=np.asarray(pos, dtype=np.int64).reshape(-1, 2))
    return list(done.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    ok = 0
    for cid in ids:
        try:
            states = process_complex(cid, args.out)
            print(f"{cid}: states={states}", flush=True)
            if states:
                ok += 1
        except Exception as e:
            print(f"{cid}: FAIL {type(e).__name__}: {e}", flush=True)
    print(f"\ngraph-precompute done: {ok}/{len(ids)} complexes -> {args.out}")


if __name__ == "__main__":
    main()
