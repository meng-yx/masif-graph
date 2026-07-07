"""M3 preprocessing PASS 2 (run in the `atomsurf_h100` env — needs diffusion_net).

Per (complex, state, chain) compute + cache the DiffusionNet operators from the MaSIF `.ply` mesh.
Joined with PASS-1 graph `.npz` by filename at training time. Operators are conformation-specific
but feature-independent, so they are computed once here (the expensive step, ~3–9 s/surface).

Output: <out>/{cid}__{state}__{pid}.surf.pt  (dict of operator tensors + verts/faces).
"""
from __future__ import annotations

import argparse
import os

import torch

from masif_graph.io.reference import SURFACE_DIR
from masif_graph.m3.surface_encoder import load_ply_mesh, build_surface_object


def af3_pdbid(pdb_id):
    return f"{pdb_id}AF"


def surf_ply_path(pdb_id, chain, state):
    pid = pdb_id if state == "holo" else af3_pdbid(pdb_id)
    return os.path.join(SURFACE_DIR, f"{pid}_{chain}.ply")


def save_surf(cid, state, pid_label, pdb_id, chain, out_dir, k_eig=128):
    ply = surf_ply_path(pdb_id, chain, state)
    if not os.path.exists(ply):
        return False
    verts, faces, feats, normals = load_ply_mesh(ply)
    s = build_surface_object(verts, faces, feats, k_eig=k_eig)
    out = os.path.join(out_dir, f"{cid}__{state}__{pid_label}.surf.pt")
    torch.save({"verts": verts, "faces": faces, "mass": s.mass, "L": s.L,
                "evals": s.evals, "evecs": s.evecs, "gradX": s.gradX, "gradY": s.gradY,
                "n_vert": verts.shape[0]}, out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-eig", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    ok = 0
    for cid in ids:
        pdb, c1, c2 = cid.split("_")
        for state in ("holo", "af3"):
            got = 0
            for pid_label, chain in (("p1", c1), ("p2", c2)):
                try:
                    if save_surf(cid, state, pid_label, pdb, chain, args.out, args.k_eig):
                        got += 1
                except Exception as e:
                    print(f"{cid} {state} {pid_label}: FAIL {type(e).__name__}: {e}", flush=True)
            if got:
                print(f"{cid} {state}: {got}/2 surfaces", flush=True)
                ok += got
    print(f"\nsurf-precompute done: {ok} chain-surfaces -> {args.out}")


if __name__ == "__main__":
    main()
