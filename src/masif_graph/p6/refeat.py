"""Phase-6 C(b).0 — re-featurize the Phase-4/5 PPI `.npz` from 14-D to the unified 26-D atom space.

Why not just re-run `p4.precompute`? Its inputs are gone: the /scratch 30-day cleanup wiped the
reference surface tree (only 16/4,872 training complexes still have a `.ply`; see the running log
§1.1). The npz themselves survived on `/work`, and the 14->26-D change touches **only
`atom_feat`** — vertex features, all three edge types, `surf_node_idx` and the contacts are
dimension-independent. So we patch `atom_feat` in place.

23 of the 26 dims are already in the stored 14-D vector or derivable from it; the missing three
(H-bond donor / acceptor / formal charge) need per-atom **name + residue name**, which only the
chain PDB has. Those chain PDBs are re-derivable from a fresh RCSB download: extracting the chain
the way the reference `extractPDB` did reproduced the reference heavy-atom table **exactly on
40/40 test chains** (identical resid/name/element sequence and coordinates).

Nothing is trusted blind. `refeat_chain` recomputes the *14-D* vector from the re-derived atom
table and requires it to equal the stored one, and checks the stored surface `keys` against the
re-derived (chain, resseq, name) at every surface row. A chain that fails either check is dropped
rather than patched, so a mis-ordered atom table can never silently poison training.
"""
from __future__ import annotations

import os
import types
import urllib.error
import urllib.request

import numpy as np

from masif_graph.graph.build import build_atom_graph
from masif_graph.io.reference import parse_heavy_atoms
from masif_graph.m3.chem_graph import element_chem_features
from masif_graph.p6.atoms import DIM, protein_features

RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
# 14-D layout (graph/hetero.build_hetero_graph): 6 element one-hot + backbone + aromatic + degree
# + is_surface + flex + 3 element-chem.
F14 = 14


# ---------------------------------------------------------------------------------------------
# chain PDB recovery
# ---------------------------------------------------------------------------------------------
def fetch_raw_pdb(pdb_id: str, cache_dir: str) -> str:
    """Download `{pdb}.pdb` from RCSB into `cache_dir` (cached). Returns the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    urllib.request.urlretrieve(RCSB_URL.format(pdb=pdb_id.upper()), tmp)
    os.replace(tmp, path)
    return path


def extract_chain_pdb(raw_pdb: str, out_pdb: str, chain_ids: str) -> str:
    """Re-implementation of the reference `input_output/extractPDB.py` selection rules.

    Keeps standard residues plus SEQRES-declared modified amino acids of the requested chains, and
    drops alternate locations other than A/1 — the same rules the reference used, so the resulting
    heavy-atom table matches the one baked into the stored npz."""
    from Bio.PDB import PDBIO, PDBParser, Select, Selection, StructureBuilder
    from Bio.SeqUtils import IUPACData

    protein_letters = {x.upper() for x in IUPACData.protein_letters_3to1}

    class NotDisordered(Select):
        def accept_atom(self, atom):
            return not atom.is_disordered() or atom.get_altloc() in ("A", "1")

    modified = set()
    with open(raw_pdb) as fh:
        for line in fh:
            if line[:6] == "SEQRES":
                modified.update(line.split()[4:])
    modified -= protein_letters

    struct = PDBParser(QUIET=True).get_structure("i", raw_pdb)
    model = Selection.unfold_entities(struct, "M")[0]
    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("o"); sb.init_seg(" "); sb.init_model(0)
    out = sb.get_structure()
    wanted = set(chain_ids)
    for chain in model:
        if chain.get_id() not in wanted:
            continue
        sb.init_chain(chain.get_id())
        for residue in chain:
            het = residue.get_id()
            if het[0] == " " or het[0][-3:] in modified:
                out[0][chain.get_id()].add(residue)
    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    writer = PDBIO()
    writer.set_structure(out)
    writer.save(out_pdb, select=NotDisordered())
    return out_pdb


# ---------------------------------------------------------------------------------------------
# feature reconstruction
# ---------------------------------------------------------------------------------------------
def atom_table_features(pdb_path: str, is_surface: np.ndarray):
    """Chain PDB + the authoritative is_surface mask -> (feat14, feat26).

    `feat14` reproduces exactly what `graph/hetero.build_hetero_graph` wrote into the stored npz
    (base 10 + flex/8 + element-chem 3); it exists purely so the caller can assert the re-derived
    atom table is row-aligned with the stored one."""
    coords, elements, resids, names = parse_heavy_atoms(pdb_path)
    n = len(coords)
    if n != len(is_surface):
        raise ValueError(f"atom count {n} != stored {len(is_surface)}")
    chain = types.SimpleNamespace(n_atom=n, atom_coords=coords, atom_element=elements,
                                  atom_resid=resids, atom_name=names)
    surf = types.SimpleNamespace(full_to_surf=np.where(is_surface, np.arange(n), -1).astype(np.int64))
    ag = build_atom_graph(chain, surf, pdb_path, with_spatial=False)

    chem = element_chem_features(elements)
    flex_norm = (np.clip(ag.flex_depth, 0, 8) / 8.0).astype(np.float32)[:, None]
    feat14 = np.concatenate([ag.node_feat, flex_norm, chem], axis=1).astype(np.float32)

    aromatic = ag.node_feat[:, 7]
    degree = np.rint(ag.node_feat[:, 8] * 6.0).astype(np.int64)
    resnames = np.asarray([r.split(":")[2] for r in resids], dtype=object)
    feat26 = protein_features(elements, names, resnames, aromatic, degree, is_surface, ag.flex_depth)
    return feat14, feat26, resids, names


def _surface_keys(resids, names, surf_node_idx):
    return np.array([f"{resids[a].split(':')[0]}:{resids[a].split(':')[1]}:{names[a]}"
                     for a in surf_node_idx], dtype="S24")


def refeat_chain(npz_path: str, pdb_path: str, out_path: str, strict: bool = True) -> dict:
    """Rewrite one chain npz with 26-D `atom_feat`. Returns a check report.

    Every other array is copied through untouched. Two hard checks gate the write:
      * `feat14_exact`  — the re-derived 14-D vector equals the stored one (bit-for-bit);
      * `keys_match`    — the stored surface `keys` equal the re-derived (chain,resseq,name).
    """
    z = np.load(npz_path)
    stored = z["atom_feat"]
    if stored.shape[1] == DIM:
        return {"ok": True, "skipped": "already 26-D"}
    is_surface = stored[:, 9] > 0.5
    feat14, feat26, resids, names = atom_table_features(pdb_path, is_surface)

    rep = {"n_atom": int(stored.shape[0]), "n_surf": int(z["n_surf"])}
    rep["feat14_exact"] = bool(np.array_equal(feat14, stored))
    if not rep["feat14_exact"]:
        rep["feat14_maxdiff"] = float(np.abs(feat14 - stored).max())
        rep["feat14_bad_rows"] = int((np.abs(feat14 - stored).max(1) > 0).sum())
    keys_new = _surface_keys(resids, names, z["surf_node_idx"])
    rep["keys_match"] = bool(np.array_equal(keys_new, z["keys"]))
    rep["ok"] = rep["feat14_exact"] and rep["keys_match"]
    if strict and not rep["ok"]:
        return rep

    payload = {k: z[k] for k in z.files}
    payload["atom_feat"] = feat26.astype(np.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part.npz"          # savez_compressed appends .npz unless already present
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)
    rep["written"] = out_path
    return rep


def refeat_complex(src_dir, dst_dir, cid, pdb_cache, chain_dir, state="holo", pdb_suffix=""):
    """Re-featurize both chains of one complex + copy the contacts npz. Returns a report dict."""
    pdb_id, c1, c2 = cid.split("_")
    reports = {}
    for pid, ch in (("p1", c1), ("p2", c2)):
        src = os.path.join(src_dir, f"{cid}__{state}__{pid}.npz")
        dst = os.path.join(dst_dir, f"{cid}__{state}__{pid}.npz")
        if not os.path.exists(src):
            reports[pid] = {"ok": False, "err": "missing src"}
            continue
        if os.path.exists(dst):
            reports[pid] = {"ok": True, "skipped": "exists"}
            continue
        try:
            chain_pdb = os.path.join(chain_dir, f"{pdb_id}{pdb_suffix}_{ch}.pdb")
            if not os.path.exists(chain_pdb):
                raw = fetch_raw_pdb(pdb_id, pdb_cache)
                extract_chain_pdb(raw, chain_pdb, ch)
            reports[pid] = refeat_chain(src, chain_pdb, dst)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            reports[pid] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
    src_c = os.path.join(src_dir, f"{cid}__contacts.npz")
    dst_c = os.path.join(dst_dir, f"{cid}__contacts.npz")
    if os.path.exists(src_c) and not os.path.exists(dst_c):
        import shutil
        shutil.copyfile(src_c, dst_c)
    return reports


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--state", default="holo", help="holo | af3 (af3 uses the '<PDB>AF' chain PDBs)")
    ap.add_argument("--pdb-cache", default="/scratch/ymeng/masif-graph/data/rcsb_cache")
    ap.add_argument("--chain-dir", default="/scratch/ymeng/masif-graph/data/chain_pdbs")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    suffix = "AF" if args.state == "af3" else ""
    out, nok = {}, 0
    for cid in ids:
        rep = refeat_complex(args.src, args.dst, cid, args.pdb_cache, args.chain_dir,
                             state=args.state, pdb_suffix=suffix)
        good = all(r.get("ok") for r in rep.values()) and len(rep) == 2
        nok += good
        out[cid] = rep
        print(f"{cid}: {'OK' if good else 'FAIL ' + json.dumps(rep)}", flush=True)
    print(f"\nrefeat done: {nok}/{len(ids)} complexes -> {args.dst}")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        json.dump(out, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
