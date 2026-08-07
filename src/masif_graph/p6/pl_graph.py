"""Phase-6 C(a).3 — Path-B protein-ligand graph builder (PDBbind -> the Phase-4 npz contract).

Path B (locked, HEAD 410238d): the ligand is **heavy atoms as graph nodes**, not a computed
ligand surface — the `.sif score_binder` ligand-surface path is WIP/buggy at 5k scale.

The key structural choice here is that a protein-ligand complex is emitted as **exactly the same
artefact as a PPI complex**: two independently-encodable chain graphs plus a contact list.

    {cid}__holo__p1.npz   protein: atoms + surface vertices + aa/vv/va edges (26-D atom feats)
    {cid}__holo__p2.npz   ligand : atoms + covalent edges only, n_vert = 0, every atom a readout
    {cid}__contacts.npz   pos (<=5.0 A) / pos_sc (<=4.0 A) as (protein_surf_row, ligand_row)

The two sides must be encoded **independently** — the same rule that makes PPI retrieval
non-trivial. If ligand atoms were injected into the protein graph, protein vertices would message
into them and the ligand embedding would already encode its own protein, so retrieval would be
free and the metric meaningless. Keeping them as two graphs means `p4.dataset.ComplexP4`, the
chain-level retrieval loss and the whole Phase-4 training loop work on protein-ligand pairs with
no new code path — the mixture is just a longer id list.

The ligand graph reuses the atom-atom edge feature layout (`D_AA` = 4-D bond order one-hot + a
rotatable flag), so one encoder consumes both molecule types.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.graph.hetero import build_hetero_graph
from masif_graph.io.reference import PDB_DIR, PRECOMP_DIR, parse_heavy_atoms
from masif_graph.p6.atoms import DIM, ligand_features
from masif_graph.surface.atoms import build_surface_atoms

POS_CUT = 5.0      # protein-heavy-atom <-> ligand-heavy-atom contact (the dense positive set)
POS_SC_CUT = 4.0   # tighter "clean" contact set (analogue of the PPI sc-filtered positives)
LIG_CHAIN_CUT = 6.0  # a protein chain is kept if any heavy atom is this close to the ligand


# ---------------------------------------------------------------------------------------------
# 1. protein PDB preparation (runs BEFORE the .sif surface build)
# ---------------------------------------------------------------------------------------------
def ligand_coords(pdbbind_dir: str, pid: str) -> np.ndarray:
    """Ligand heavy-atom coordinates from the SDF (fallback mol2). Geometry only."""
    from rdkit import Chem

    sdf = os.path.join(pdbbind_dir, pid, f"{pid}_ligand.sdf")
    mol = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=True)[0] if os.path.exists(sdf) else None
    if mol is None:
        mol2 = os.path.join(pdbbind_dir, pid, f"{pid}_ligand.mol2")
        mol = Chem.MolFromMol2File(mol2, sanitize=False, removeHs=True)
    if mol is None:
        raise ValueError(f"{pid}: no readable ligand")
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
                     if mol.GetAtomWithIdx(i).GetAtomicNum() > 1], dtype=np.float64)


def _is_heavy(line: str) -> bool:
    el = line[76:78].strip() or "".join(c for c in line[12:16].strip() if c.isalpha())[:1]
    return el.upper() not in ("H", "D")


def prep_protein_pdb(pdbbind_dir: str, pid: str, out_pdb: str, max_atoms: int = 8000) -> dict:
    """Write the pocket-bearing protein chains as ONE pseudo-chain 'A' for the `.sif` pipeline.

    PDBbind `_protein.pdb` files carry 1-24 chains but the ligand only touches a few. Keeping every
    chain would triangulate assemblies of 24k+ atoms for no gain, and *dropping* a contacting chain
    would silently delete part of the pocket. So: keep exactly the chains with a heavy atom within
    `LIG_CHAIN_CUT` of the ligand, merge them into chain 'A', and renumber residues sequentially so
    (chain, resseq) stays unique across the merge (the whole pipeline keys atoms on that pair).
    """
    lig = ligand_coords(pdbbind_dir, pid)
    tree = cKDTree(lig)
    src = os.path.join(pdbbind_dir, pid, f"{pid}_protein.pdb")
    lines_by_res, order, chains_seen = {}, [], set()
    with open(src) as fh:
        for line in fh:
            if line[:6] != "ATOM  ":
                continue
            key = (line[21], line[22:27])          # chain + resseq&icode
            if key not in lines_by_res:
                lines_by_res[key] = []
                order.append(key)
            lines_by_res[key].append(line)
            chains_seen.add(line[21])

    keep_chains = set()
    for key, lines in lines_by_res.items():
        if key[0] in keep_chains:
            continue
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
        if (tree.query_ball_point(xyz, LIG_CHAIN_CUT, return_length=True) > 0).any():
            keep_chains.add(key[0])

    kept = [k for k in order if k[0] in keep_chains]
    # PDBbind `_protein.pdb` files are protonated, so cap on HEAVY atoms (what MSMS/the graph sees)
    n_heavy = sum(sum(_is_heavy(l) for l in lines_by_res[k]) for k in kept)
    info = {"pid": pid, "chains_all": len(chains_seen), "chains_kept": len(keep_chains),
            "n_res": len(kept), "n_heavy": n_heavy, "n_lig_atom": int(len(lig))}
    if n_heavy == 0:
        info["skip"] = "no protein chain within cutoff"
        return info
    if n_heavy > max_atoms:
        info["skip"] = f"n_heavy {n_heavy} > {max_atoms}"     # MSMS cost blows up on assemblies
        return info

    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    with open(out_pdb, "w") as out:
        serial = 0
        for new_res, key in enumerate(kept, start=1):
            for line in lines_by_res[key]:
                if len(line) < 54:
                    continue
                serial += 1
                # cols 1-6 record | 7-11 serial | 12-21 name/altloc/resname | 22 chain
                # | 23-26 resseq | 27 icode | 28+ unchanged
                out.write(f"{line[:6]}{serial:5d}{line[11:21]}A{new_res:4d} {line[27:]}")
        out.write("END\n")
    info["written"] = out_pdb
    return info


# ---------------------------------------------------------------------------------------------
# 2. graph assembly (runs AFTER the .sif produced the .ply + 04b precompute)
# ---------------------------------------------------------------------------------------------
class _ChainShim:
    """The subset of `io.reference.Chain` that `build_hetero_graph` actually reads."""

    def __init__(self, complex_id, pdb_id, chain_ids, verts, pdb_path):
        self.complex_id, self.pid = complex_id, "p1"
        self.pdb_id, self.chain_ids = pdb_id, chain_ids
        self.verts = verts
        (self.atom_coords, self.atom_element,
         self.atom_resid, self.atom_name) = parse_heavy_atoms(pdb_path)

    @property
    def n_vert(self):
        return len(self.verts)

    @property
    def n_atom(self):
        return len(self.atom_coords)


def load_ligand(pdbbind_dir: str, pid: str):
    """RDKit heavy-atom molecule for a PDBbind ligand (SDF preferred, mol2 fallback)."""
    from rdkit import Chem

    sdf = os.path.join(pdbbind_dir, pid, f"{pid}_ligand.sdf")
    mol = None
    if os.path.exists(sdf):
        mol = Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)[0]
    if mol is None:
        mol2 = os.path.join(pdbbind_dir, pid, f"{pid}_ligand.mol2")
        if os.path.exists(mol2):
            mol = Chem.MolFromMol2File(mol2, sanitize=True, removeHs=True)
    if mol is None:  # last resort: skip sanitization (aromaticity may be wrong; flagged by caller)
        if os.path.exists(sdf):
            mol = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=True)[0]
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                return None
    return mol


def ligand_graph(mol):
    """RDKit mol -> (atom_feat 26-D, coords, aa_edge, aa_order, aa_rot) with the protein layout."""
    from rdkit import Chem

    mol = Chem.RemoveHs(mol)
    n = mol.GetNumAtoms()
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)
    feat = ligand_features(mol)                       # (n, 26); is_surface defaults to 1

    rot = set()
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")   # standard rotatable-bond SMARTS
    for a, b in mol.GetSubstructMatches(patt):
        rot.add(frozenset((a, b)))

    order_map = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.AROMATIC: 2}
    src, dst, oh, rf = [], [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        o = 2 if bond.GetIsAromatic() else order_map.get(bond.GetBondType(), 3)
        r = 1.0 if frozenset((i, j)) in rot else 0.0
        for a, b in ((i, j), (j, i)):
            src.append(a); dst.append(b); oh.append(o); rf.append(r)
    aa_edge = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    aa_order = np.zeros((len(oh), 4), np.float32)
    if oh:
        aa_order[np.arange(len(oh)), np.array(oh)] = 1.0
    return feat.astype(np.float32), coords, aa_edge, aa_order, np.array(rf, np.float32)


def _empty(shape, dtype=np.float32):
    return np.zeros(shape, dtype)


def save_ligand_npz(path, feat, coords, aa_edge, aa_order, aa_rot):
    """Ligand chain npz: atoms only (n_vert = 0), every heavy atom is a readout node."""
    n = feat.shape[0]
    with open(path + ".part.npz", "wb") as fh:
        np.savez_compressed(
            fh,
            atom_feat=feat, aa_edge=aa_edge, aa_order=aa_order, aa_rot=aa_rot,
            vert_feat=_empty((0, 4)), vv_edge=np.zeros((2, 0), np.int64),
            vv_dist=_empty(0), vv_cos=_empty(0),
            va_v=np.zeros(0, np.int64), va_a=np.zeros(0, np.int64),
            va_dist=_empty(0), va_cos=_empty(0),
            surf_node_idx=np.arange(n, dtype=np.int64), n_surf=np.int64(n),
            desc_straight=_empty((n, 80)), desc_flipped=_empty((n, 80)),   # frozen ref: N/A here
            coord=coords.astype(np.float32),
            keys=np.array([f"L:0:{i}" for i in range(n)], dtype="S24"),
        )
    os.replace(path + ".part.npz", path)


def save_protein_npz(path, chain, surf, g):
    idx = np.nonzero(g.atom_surf_row >= 0)[0]
    sni = idx[np.argsort(g.atom_surf_row[idx])].astype(np.int64)
    assert sni.shape[0] == g.n_surf, (sni.shape, g.n_surf)
    keys = np.array([f"{chain.atom_resid[r].split(':')[0]}:{chain.atom_resid[r].split(':')[1]}:"
                     f"{chain.atom_name[r]}" for r in surf.atom_idx], dtype="S24")
    with open(path + ".part.npz", "wb") as fh:
        np.savez_compressed(
            fh,
            atom_feat=g.atom_feat.astype(np.float32),
            aa_edge=g.aa_edge.astype(np.int64), aa_order=g.aa_order.astype(np.float32),
            aa_rot=g.aa_rot.astype(np.float32),
            vert_feat=g.vert_feat.astype(np.float32),
            vv_edge=g.vv_edge.astype(np.int64), vv_dist=g.vv_dist.astype(np.float32),
            vv_cos=g.vv_cos.astype(np.float32),
            va_v=g.va_v.astype(np.int64), va_a=g.va_a.astype(np.int64),
            va_dist=g.va_dist.astype(np.float32), va_cos=g.va_cos.astype(np.float32),
            surf_node_idx=sni, n_surf=np.int64(g.n_surf),
            desc_straight=_empty((g.n_surf, 80)), desc_flipped=_empty((g.n_surf, 80)),
            coord=surf.coord.astype(np.float32), keys=keys,
        )
    os.replace(path + ".part.npz", path)


def build_complex(pid, out_dir, pdbbind_dir, va_radius=5.0, va_kmax=8) -> dict:
    """Assemble both npz + contacts for one PDBbind complex. `.sif` outputs must already exist."""
    cid = f"pl{pid}"
    sid = f"{cid}_A"                                   # the id the .sif pipeline was run under
    pc = os.path.join(PRECOMP_DIR, sid)
    xyz = [np.load(os.path.join(pc, f"p1_{a}.npy")) for a in ("X", "Y", "Z")]
    verts = np.column_stack(xyz).astype(np.float64)
    pdb_path = os.path.join(PDB_DIR, f"{sid}.pdb")
    chain = _ChainShim(sid, cid, "A", verts, pdb_path)

    zero_desc = np.zeros((len(verts), 80), np.float32)
    surf = build_surface_atoms(verts, chain.atom_coords, chain.atom_element, chain.atom_resid,
                               zero_desc, zero_desc, ops=("mean",))
    g = build_hetero_graph(chain, surf, pdb_path, va_radius=va_radius, va_kmax=va_kmax,
                           unified_atom_feat=True)
    if g.atom_feat.shape[1] != DIM:
        raise ValueError(f"{pid}: protein atom_feat dim {g.atom_feat.shape[1]} != {DIM}")

    mol = load_ligand(pdbbind_dir, pid)
    if mol is None:
        return {"pid": pid, "ok": False, "err": "ligand unreadable"}
    lfeat, lcoord, la_edge, la_order, la_rot = ligand_graph(mol)

    # contacts: protein SURFACE atom <-> ligand heavy atom
    tree = cKDTree(lcoord)
    pos, pos_sc = [], []
    for r, c in enumerate(surf.coord):
        for j in tree.query_ball_point(c, POS_CUT):
            pos.append((r, j))
            if np.linalg.norm(lcoord[j] - c) <= POS_SC_CUT:
                pos_sc.append((r, j))

    os.makedirs(out_dir, exist_ok=True)
    save_protein_npz(os.path.join(out_dir, f"{cid}__holo__p1.npz"), chain, surf, g)
    save_ligand_npz(os.path.join(out_dir, f"{cid}__holo__p2.npz"),
                    lfeat, lcoord, la_edge, la_order, la_rot)
    with open(os.path.join(out_dir, f"{cid}__contacts.npz.part.npz"), "wb") as fh:
        np.savez_compressed(fh, pos=np.asarray(pos, np.int64).reshape(-1, 2),
                            pos_sc=np.asarray(pos_sc, np.int64).reshape(-1, 2))
    os.replace(os.path.join(out_dir, f"{cid}__contacts.npz.part.npz"),
               os.path.join(out_dir, f"{cid}__contacts.npz"))
    return {"pid": pid, "ok": True, "n_atom": int(chain.n_atom), "n_vert": int(len(verts)),
            "n_surf": int(g.n_surf), "n_lig": int(lfeat.shape[0]),
            "n_pos": len(pos), "n_pos_sc": len(pos_sc),
            "n_iface_prot": int(len(set(p[0] for p in pos))),
            "n_iface_lig": int(len(set(p[1] for p in pos)))}


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdbbind", default="/scratch/ymeng/masif-graph/data/pdbbind")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    reps, nok = [], 0
    for pid in ids:
        try:
            r = build_complex(pid, args.out, args.pdbbind)
        except Exception as exc:  # noqa: BLE001 - one bad complex must not kill the chunk
            r = {"pid": pid, "ok": False, "err": f"{type(exc).__name__}: {exc}"}
        nok += bool(r.get("ok"))
        reps.append(r)
        print(json.dumps(r), flush=True)
    print(f"\npl_graph done: {nok}/{len(ids)}")
    if args.report:
        json.dump(reps, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
