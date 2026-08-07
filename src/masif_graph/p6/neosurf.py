"""Phase-6 C(c).3 axis 3 — the neosurface benchmark corpus (ligand-induced ternary complexes).

Source: `masif-neosurf-af2/computational_benchmark/benchmark_pdbs.txt` — 14 drug-induced protein
complexes, each split into two subunits, with the small molecule assigned to one of them
(`PDB, chains1, chains2, CODE_authchain, <pdb>_<label_asym>_<CODE>`). That is exactly the
neosurface claim: subunit A's surface *plus the bound drug* should recruit subunit B.

Emitted per system, in the same npz contract as everywhere else in Phase 6:
    {sys}__holo__p1.npz / __p2.npz / __contacts.npz   the two protein subunits + their interface
    {sys}__lig.npz                                     the drug as a ligand atom graph (Path B)
    {sys}__ligcontacts.npz                             pos (owner_surf_row, ligand_row) + owner id

Keeping the drug in its own file is what makes the **ligand-present vs ligand-absent contrast**
possible: the query patch is built by the eval script as `subunit interface atoms (+ drug atoms)`,
so the control is literally the same run with the drug embeddings omitted.

Bound-pose ligand geometry with correct bond orders comes from the RCSB ModelServer instance
endpoint (`models.rcsb.org/v1/<pdb>/ligand?label_asym_id=..&encoding=sdf`) — the ideal-coordinate
SDF would have the right bonds but the wrong pose, and the wrong pose means wrong contacts.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.graph.hetero import build_hetero_graph
from masif_graph.io.reference import PDB_DIR, PRECOMP_DIR
from masif_graph.p6.pl_graph import (POS_CUT, POS_SC_CUT, _ChainShim, ligand_graph,
                                     save_ligand_npz, save_protein_npz)
from masif_graph.pairs.construct import atom_positives_from_vertex_contacts, vertex_contacts
from masif_graph.surface.atoms import build_surface_atoms

MODELSERVER = "https://models.rcsb.org/v1/{pdb}/ligand?label_asym_id={asym}&encoding=sdf"
PREFIX = "nb"          # keeps benchmark ids from colliding with anything else in the shared tree


def parse_benchmark(path):
    """`PDB,chains1,chains2,CODE_chain,sdfname` -> list of dicts."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 5:
            continue
        code, lig_chain = f[3].split("_")
        out.append({"pdb": f[0].upper(), "c1": f[1], "c2": f[2], "code": code,
                    "lig_chain": lig_chain, "asym": f[4].split("_")[1],
                    "sys": f"{PREFIX}{f[0].upper()}"})
    return out


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = f"{path}.part{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def fetch_ligand(pdb, asym, cache_dir):
    """Bound-pose ligand SDF -> sanitized RDKit heavy-atom mol."""
    from rdkit import Chem

    p = fetch(MODELSERVER.format(pdb=pdb.lower(), asym=asym),
              os.path.join(cache_dir, f"{pdb}_{asym}.sdf"))
    mol = Chem.SDMolSupplier(p, sanitize=True, removeHs=True)[0]
    if mol is None:
        mol = Chem.SDMolSupplier(p, sanitize=False, removeHs=True)[0]
        if mol is not None:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                             ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return mol


def _load_chain(sys_id, pid, chains):
    pc = os.path.join(PRECOMP_DIR, f"{sys_id}_{chains[0]}_{chains[1]}")
    xyz = [np.load(os.path.join(pc, f"{pid}_{a}.npy")) for a in ("X", "Y", "Z")]
    verts = np.column_stack(xyz).astype(np.float64)
    idx = 0 if pid == "p1" else 1
    pdb_path = os.path.join(PDB_DIR, f"{sys_id}_{chains[idx]}.pdb")
    chain = _ChainShim(f"{sys_id}_{chains[0]}_{chains[1]}", sys_id, chains[idx], verts, pdb_path)
    chain.pid = pid
    zero = np.zeros((len(verts), 80), np.float32)
    surf = build_surface_atoms(verts, chain.atom_coords, chain.atom_element, chain.atom_resid,
                               zero, zero, ops=("mean",))
    g = build_hetero_graph(chain, surf, pdb_path, unified_atom_feat=True)
    return chain, surf, g, verts


def build_system(entry, out_dir, sdf_cache) -> dict:
    """Assemble the two subunit graphs, their interface, the drug graph and its contacts."""
    sys_id, chains = entry["sys"], (entry["c1"], entry["c2"])
    os.makedirs(out_dir, exist_ok=True)
    parts = {}
    for pid in ("p1", "p2"):
        parts[pid] = _load_chain(sys_id, pid, chains)
        save_protein_npz(os.path.join(out_dir, f"{sys_id}__holo__{pid}.npz"),
                         parts[pid][0], parts[pid][1], parts[pid][2])

    # protein-protein interface, the same dense definition p4.precompute uses
    vp, _ = vertex_contacts(parts["p1"][3], parts["p2"][3], pos_cutoff=1.0, sc1=None)
    pos = atom_positives_from_vertex_contacts(vp, parts["p1"][1].vertex_surf_idx,
                                              parts["p2"][1].vertex_surf_idx)
    with open(os.path.join(out_dir, f"{sys_id}__contacts.npz.part.npz"), "wb") as fh:
        np.savez_compressed(fh, pos=np.asarray(pos, np.int64).reshape(-1, 2),
                            pos_sc=np.asarray(pos, np.int64).reshape(-1, 2))
    os.replace(os.path.join(out_dir, f"{sys_id}__contacts.npz.part.npz"),
               os.path.join(out_dir, f"{sys_id}__contacts.npz"))

    mol = fetch_ligand(entry["pdb"], entry["asym"], sdf_cache)
    if mol is None:
        return {"sys": sys_id, "ok": False, "err": "ligand SDF unreadable", "n_pp_pos": len(pos)}
    lfeat, lcoord, la_e, la_o, la_r = ligand_graph(mol)
    save_ligand_npz(os.path.join(out_dir, f"{sys_id}__lig.npz"),
                    lfeat, lcoord, la_e, la_o, la_r)

    # owner subunit = the one whose chain set contains the drug's auth chain; fall back to whichever
    # subunit the drug actually touches (some entries label the drug with the partner's chain id)
    owner = 1 if entry["lig_chain"] in chains[0] else 2
    tree = cKDTree(lcoord)
    counts = []
    for pid in ("p1", "p2"):
        sc = parts[pid][1].coord
        counts.append(int((tree.query_ball_point(sc, POS_CUT, return_length=True) > 0).sum()))
    if counts[owner - 1] == 0 and max(counts) > 0:
        owner = 1 + int(np.argmax(counts))
    opid = f"p{owner}"
    surf_o = parts[opid][1]
    lp, lp_sc = [], []
    for r, c in enumerate(surf_o.coord):
        for j in tree.query_ball_point(c, POS_CUT):
            lp.append((r, j))
            if np.linalg.norm(lcoord[j] - c) <= POS_SC_CUT:
                lp_sc.append((r, j))
    with open(os.path.join(out_dir, f"{sys_id}__ligcontacts.npz.part.npz"), "wb") as fh:
        np.savez_compressed(fh, pos=np.asarray(lp, np.int64).reshape(-1, 2),
                            pos_sc=np.asarray(lp_sc, np.int64).reshape(-1, 2),
                            owner=np.int64(owner))
    os.replace(os.path.join(out_dir, f"{sys_id}__ligcontacts.npz.part.npz"),
               os.path.join(out_dir, f"{sys_id}__ligcontacts.npz"))
    return {"sys": sys_id, "ok": True, "owner": owner,
            "n_surf": [int(parts[p][2].n_surf) for p in ("p1", "p2")],
            "n_pp_pos": len(pos), "n_lig": int(lfeat.shape[0]), "n_lig_pos": len(lp),
            "lig_contact_counts": counts}


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="benchmark_pdbs.txt")
    ap.add_argument("--only", default=None, help="comma-separated PDB ids to build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sdf-cache", default="/scratch/ymeng/masif-graph/data/neosurf_sdf")
    args = ap.parse_args()
    entries = parse_benchmark(args.bench)
    if args.only:
        keep = {x.strip().upper() for x in args.only.split(",")}
        entries = [e for e in entries if e["pdb"] in keep]
    nok = 0
    for e in entries:
        try:
            r = build_system(e, args.out, args.sdf_cache)
        except Exception as exc:  # noqa: BLE001
            r = {"sys": e["sys"], "ok": False, "err": f"{type(exc).__name__}: {exc}"}
        nok += bool(r.get("ok"))
        print(json.dumps(r), flush=True)
    print(f"\nneosurf build: {nok}/{len(entries)}")


if __name__ == "__main__":
    main()
