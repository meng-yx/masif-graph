"""Phase-4 PASS-1: hetero graph -> cross-env `.npz` (run in the `masif-graph` env on Jed).

Per (complex, chain) save the HeteroSurfaceGraph tensors (atom nodes + covalent edges, vertex nodes +
mesh edges + vertex-atom edges, all invariant edge scalars) + the mean-pooled frozen MaSIF descriptors
(for the frozen ceiling reference) + surface-atom identity keys. Per complex save the holo sc-filtered
contact positives (surface-atom row pairs). Byte-string keys, no pickle -> loads under the py3.8/numpy-1.23
training env (`dataset.py`), exactly the Phase-3 cross-version pattern.

Output: <out>/{cid}__holo__{pid}.npz  and  <out>/{cid}__contacts.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from masif_graph.io.reference import load_complex, complex_is_available, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts
from masif_graph.graph.hetero import build_hetero_graph


def af3_state_id(holo_id: str) -> str:
    """Phase-3 AF3 model id for a holo complex (matches experiments/run_m1_af3.af3_id)."""
    pdb, c1, c2 = holo_id.split("_")
    return f"{pdb}AF_{c1}_{c2}"


def rp_state_id(holo_id: str) -> str:
    """Phase-2 fixed-backbone FASPR-repack id (scripts/repack_one.sh writes `{PDB}RP_{c1}_{c2}`).

    Used by Phase-8 A1.2: a repack keeps sidechain IDENTITY and changes only the rotamer, which is
    the conformational perturbation an apo structure actually applies — the one A1's feature
    shuffling does not test.
    """
    pdb, c1, c2 = holo_id.split("_")
    return f"{pdb}RP_{c1}_{c2}"


def _keys(chain, surf):
    ks = []
    for r in surf.atom_idx:
        chn, seq, _rn = chain.atom_resid[r].split(":")
        ks.append(f"{chn}:{seq}:{chain.atom_name[r]}")
    return np.array(ks, dtype="S24")  # fixed-width bytes: cross-numpy-version safe


def _surf_node_idx(g):
    """Atom node index per surface row (ordered by surface row). z[r] = atom_state[idx[r]]."""
    idx = np.nonzero(g.atom_surf_row >= 0)[0]
    order = np.argsort(g.atom_surf_row[idx])
    return idx[order].astype(np.int64)


def save_chain(cid, pid, chain, surf, g, out_dir, max_vert, state="holo"):
    sni = _surf_node_idx(g)
    assert sni.shape[0] == g.n_surf, (sni.shape, g.n_surf)
    out = os.path.join(out_dir, f"{cid}__{state}__{pid}.npz")
    np.savez_compressed(
        out,
        atom_feat=g.atom_feat.astype(np.float32),
        aa_edge=g.aa_edge.astype(np.int64),
        aa_order=g.aa_order.astype(np.float32),
        aa_rot=g.aa_rot.astype(np.float32),
        vert_feat=g.vert_feat.astype(np.float32),
        vv_edge=g.vv_edge.astype(np.int64),
        vv_dist=g.vv_dist.astype(np.float32),
        vv_cos=g.vv_cos.astype(np.float32),
        va_v=g.va_v.astype(np.int64),
        va_a=g.va_a.astype(np.int64),
        va_dist=g.va_dist.astype(np.float32),
        va_cos=g.va_cos.astype(np.float32),
        surf_node_idx=sni,
        n_surf=np.int64(g.n_surf),
        desc_straight=surf.emb_straight["mean"].astype(np.float32),  # (n_surf, 80) frozen ceiling ref
        desc_flipped=surf.emb_flipped["mean"].astype(np.float32),
        coord=surf.coord.astype(np.float32),                         # (n_surf,3) for neg sampling
        keys=_keys(chain, surf),
    )


def process_complex(cid, out_dir, va_radius, va_kmax, max_vert, state="holo"):
    """Build + save per-chain graphs for one complex in the given conformational STATE.

    state='holo': the crystal complex `cid`, plus the holo-defined contact positives.
    state='af3' : the Phase-3 AF3 model of `cid` (loaded via `af3_state_id`), saved under the SAME
                  holo `cid` prefix + '__af3__' so eval can identity-join AF3 rows to holo contacts.
                  Contacts are NOT recomputed (they are defined by the holo complex geometry).
    """
    state_id = {"holo": lambda x: x, "af3": af3_state_id, "rp": rp_state_id}[state](cid)
    if not complex_is_available(state_id):
        return False
    p1, p2 = load_complex(state_id)
    surfs = {}
    for pid, ch in (("p1", p1), ("p2", p2)):
        surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                                   ch.desc_straight, ch.desc_flipped, ops=("mean",))
        pdb = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
        g = build_hetero_graph(ch, surf, pdb, va_radius=va_radius, va_kmax=va_kmax, max_vert=max_vert)
        save_chain(cid, pid, ch, surf, g, out_dir, max_vert, state=state)
        surfs[pid] = surf
    if state != "holo":
        return sum(s.coord.shape[0] for s in surfs.values())  # n surface atoms built (contacts are holo-only)
    # holo contacts -> owner surface-atom pairs (p1 row, p2 row).
    # Primary set `pos` = ALL touching-surface vertex contacts (pos_cutoff=1.0, no sc gate): dense &
    # stable for correspondence training/eval. Also store the sc-filtered set `pos_sc` (MaSIF's
    # cleanest-separation gate, sparse) for reference. The learned-vs-frozen comparison stays fair
    # because the frozen ceiling is computed on whichever identical pair set eval uses.
    vp_all, _ = vertex_contacts(p1.verts, p2.verts, pos_cutoff=1.0, sc1=None)
    pos = atom_positives_from_vertex_contacts(vp_all, surfs["p1"].vertex_surf_idx, surfs["p2"].vertex_surf_idx)
    vp_sc, _ = vertex_contacts(p1.verts, p2.verts, pos_cutoff=1.0, sc1=p1.sc, sc_band=(0.5, 1.0))
    pos_sc = atom_positives_from_vertex_contacts(vp_sc, surfs["p1"].vertex_surf_idx, surfs["p2"].vertex_surf_idx)
    np.savez_compressed(os.path.join(out_dir, f"{cid}__contacts.npz"),
                        pos=np.asarray(pos, dtype=np.int64).reshape(-1, 2),
                        pos_sc=np.asarray(pos_sc, dtype=np.int64).reshape(-1, 2))
    return len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--va-radius", type=float, default=5.0)
    ap.add_argument("--va-kmax", type=int, default=8)
    ap.add_argument("--max-vert", type=int, default=0, help="0 = no cap")
    ap.add_argument("--state", choices=["holo", "af3", "rp"], default="holo",
                    help="conformational state (holo crystal | Phase-3 AF3 model | FASPR repack)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    max_vert = args.max_vert if args.max_vert > 0 else None
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    ok = 0
    for cid in ids:
        try:
            npos = process_complex(cid, args.out, args.va_radius, args.va_kmax, max_vert, state=args.state)
            if npos is False:
                print(f"{cid}: unavailable", flush=True)
            else:
                unit = "contacts" if args.state == "holo" else "surf-atoms"
                print(f"{cid}: OK ({npos} {unit})", flush=True)
                ok += 1
        except Exception as e:
            print(f"{cid}: FAIL {type(e).__name__}: {e}", flush=True)
    print(f"\np4 precompute done: {ok}/{len(ids)} complexes -> {args.out}")


if __name__ == "__main__":
    main()
