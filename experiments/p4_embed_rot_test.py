"""M0 gate, final item: EMBEDDING-level rotation invariance through the encoder.

The M0 numpy gate proved edge/node features are invariant (max-diff 0.0). This closes the loop: build
the graph on original vs SE(3)-transformed geometry, convert BOTH to the encoder's dict format, run a
(random-init) HeteroEncoder, and assert the per-surface-atom embedding z is invariant. A pass here
means no coordinate leaks into the network — the descriptor learns chemistry, not pose.

  python experiments/p4_embed_rot_test.py
"""
from __future__ import annotations

import os

import numpy as np
import torch

from masif_graph.io.reference import load_complex, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.graph.hetero import build_hetero_graph, build_atom_graph, load_ply_geometry, random_se3
from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.dataset import rbf, D_AA, D_VV, D_VA, VV_RBF_N, VV_RBF_DMAX, VA_RBF_N, VA_RBF_DMAX


def graph_to_encoder_dict(g, device="cpu"):
    """In-memory HeteroSurfaceGraph -> the exact dict HeteroEncoder consumes (mirrors dataset.load_chain_graph)."""
    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    li = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)
    aa_edge = li(g.aa_edge)
    aa_feat = (torch.cat([t(g.aa_order).reshape(-1, 4), t(g.aa_rot).reshape(-1, 1)], dim=1)
               if aa_edge.shape[1] > 0 else torch.zeros(0, D_AA, device=device))
    vv = li(g.vv_edge); vv_dist = t(g.vv_dist); vv_cos = t(g.vv_cos)
    if vv.shape[1] > 0:
        vv_edge = torch.cat([vv, vv.flip(0)], dim=1)
        vv_dist = torch.cat([vv_dist, vv_dist]); vv_cos = torch.cat([vv_cos, vv_cos])
        vv_feat = torch.cat([rbf(vv_dist, VV_RBF_N, VV_RBF_DMAX), vv_cos[:, None]], dim=1)
    else:
        vv_edge = torch.zeros(2, 0, dtype=torch.long, device=device); vv_feat = torch.zeros(0, D_VV, device=device)
    va_v = li(g.va_v); va_a = li(g.va_a)
    va_feat = (torch.cat([rbf(t(g.va_dist), VA_RBF_N, VA_RBF_DMAX), t(g.va_cos)[:, None]], dim=1)
               if va_v.shape[0] > 0 else torch.zeros(0, D_VA, device=device))
    # surf_node_idx: atom node index per surface row, ordered by surf row
    idx = np.nonzero(g.atom_surf_row >= 0)[0]
    sni = li(idx[np.argsort(g.atom_surf_row[idx])])
    return {"atom_feat": t(g.atom_feat), "vert_feat": t(g.vert_feat),
            "aa_edge": aa_edge, "aa_feat": aa_feat, "vv_edge": vv_edge, "vv_feat": vv_feat,
            "va_v": va_v, "va_a": va_a, "va_feat": va_feat, "surf_node_idx": sni}


def main():
    torch.manual_seed(0)
    enc = HeteroEncoder(14, 4, D_AA, D_VV, D_VA, d=48, d_out=32, n_layers=3).eval()
    worst = 0.0
    for cid in ["1A0G_A_B", "1A22_A_B", "1A1U_A_C"]:
        p1, p2 = load_complex(cid)
        ch = p1
        surf = build_surface_atoms(ch.verts, ch.atom_coords, ch.atom_element, ch.atom_resid,
                                   ch.desc_straight, ch.desc_flipped, ops=("mean",))
        pdb = os.path.join(PDB_DIR, f"{ch.pdb_id}_{ch.chain_ids}.pdb")
        g0 = build_hetero_graph(ch, surf, pdb)
        R, tr = random_se3(seed=7)
        _, _, normals0 = load_ply_geometry(ch.pdb_id, ch.chain_ids)
        atom0 = build_atom_graph(ch, surf, pdb).coords
        g1 = build_hetero_graph(ch, surf, pdb, _override_verts=ch.verts @ R.T + tr,
                                _override_normals=normals0 @ R.T, _override_atom_coords=atom0 @ R.T + tr)
        with torch.no_grad():
            z0 = enc(graph_to_encoder_dict(g0)); z1 = enc(graph_to_encoder_dict(g1))
        md = float((z0 - z1).abs().max())
        worst = max(worst, md)
        print(f"{cid} {ch.pid}: z shape={tuple(z0.shape)}  embedding rotation max|Δ| = {md:.2e}")
    ok = worst < 1e-4
    print(f"\n>>> EMBEDDING ROTATION GATE: {'PASS' if ok else 'FAIL'} (worst max|Δ|={worst:.2e}, tol=1e-4) <<<")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
