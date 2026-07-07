"""Per-complex AF3-vs-holo diagnostics: mean pLDDT and CA-RMSD (whole-chain + interface).

Used to stratify the M1 gap by AF3 confidence and by conformational deviation from the holo crystal.
AF3 models are relabelled to holo numbering, so holo<->AF3 CA atoms match by (chain,resseq,name).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from masif_graph.io.reference import PDB_DIR, parse_heavy_atoms
from masif_graph.af3.relabel import read_cif_atoms

MODELS_ROOT = "/work/upthomae/Meng/phase3_af3/models"


def af3_model_cif(pdb_id: str, chain: str) -> str | None:
    name = f"{pdb_id}_{chain}"
    hits = sorted(glob.glob(os.path.join(MODELS_ROOT, name, "**", f"{name}_model.cif"),
                            recursive=True))
    return hits[-1] if hits else None


def chain_mean_plddt(pdb_id: str, chain: str) -> float:
    cif = af3_model_cif(pdb_id, chain)
    if cif is None:
        return float("nan")
    atoms = read_cif_atoms(cif)
    if not atoms:
        return float("nan")
    return float(np.mean([a.b for a in atoms]))


def af3_atom_plddt(pdb_id: str, chain: str) -> dict:
    """{(chain,resseq,atom_name): pLDDT} for the AF3 top-ranked model, keyed by HOLO identity.

    AF3 seq_id i (1..N) maps to the i-th holo mapping residue (relabel invariant), so the returned
    keys match the (chain,resseq,name) keys used by the surface-atom identity map. Enables per-atom
    pLDDT-weighted matching (M2 lever-0) and atom-level strata. Empty dict if no model."""
    from masif_graph.af3.sequence import chain_sequence
    cif = af3_model_cif(pdb_id, chain)
    if cif is None:
        return {}
    _seq, mapres = chain_sequence(os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb"))
    atoms = read_cif_atoms(cif)
    seq_ids = sorted({a.seq_id for a in atoms})
    if len(seq_ids) != len(mapres):
        return {}
    sid2res = {sid: mapres[k] for k, sid in enumerate(seq_ids)}
    out = {}
    for a in atoms:
        r = sid2res[a.seq_id]
        out[(r.chain, r.resseq, a.atom_name)] = a.b
    return out


def _ca_by_resseq(pdb_path: str) -> dict:
    """(chain,resseq) -> CA coord, from a per-chain PDB (holo or relabelled AF3)."""
    coords, elem, resid, name = parse_heavy_atoms(pdb_path)
    out = {}
    for c, r, nm in zip(coords, resid, name):
        if str(nm) == "CA":
            chn, seq, _rn = r.split(":")
            out[(chn, seq)] = c
    return out


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal superposition of P onto Q (both (n,3))."""
    if len(P) < 3:
        return float("nan")
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def chain_ca_rmsd(pdb_id: str, chain: str, af3_pdb_path: str,
                  interface_resseqs: set | None = None) -> tuple[float, float, float, int]:
    """(whole_chain_rmsd, iface_rmsd_globalfit, iface_rmsd_localfit, n_common).

    - whole_chain_rmsd: Kabsch over ALL common CA (global conformational deviation).
    - iface_rmsd_globalfit: interface-residue residuals under the WHOLE-CHAIN fit (conflates global
      domain motion with local interface change — inflated for flexible/multi-domain chains).
    - iface_rmsd_localfit: Kabsch over the INTERFACE CA ONLY (isolates the *local* interface
      conformational change — the honest stratifier; per user guidance). This is what to trust for
      "how much did the binding surface itself move", independent of global domain motion.
    """
    holo_pdb = os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb")
    holo_ca = _ca_by_resseq(holo_pdb)
    af3_ca = _ca_by_resseq(af3_pdb_path)
    common = [k for k in holo_ca if k in af3_ca]
    if len(common) < 3:
        return float("nan"), float("nan"), float("nan"), len(common)
    H = np.array([holo_ca[k] for k in common])
    A = np.array([af3_ca[k] for k in common])
    whole = _kabsch_rmsd(A, H)
    iface_global = float("nan")
    iface_local = float("nan")
    if interface_resseqs:
        idx = [n for n, k in enumerate(common) if k[1] in interface_resseqs]
        if len(idx) >= 3:
            # global fit, interface residuals
            Ac = A - A.mean(0); Hc = H - H.mean(0)
            U, _, Vt = np.linalg.svd(Ac.T @ Hc)
            R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            Ar = Ac @ R.T
            iface_global = float(np.sqrt(np.mean(np.sum((Ar[idx] - Hc[idx]) ** 2, axis=1))))
            # LOCAL fit: superpose on interface CA only
            iface_local = _kabsch_rmsd(A[idx], H[idx])
    return whole, iface_global, iface_local, len(common)
