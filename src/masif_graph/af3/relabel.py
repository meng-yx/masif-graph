"""Relabel an AF3 monomer model (mmCIF) back to holo residue identity and write a PDB chain.

AF3 predicts the exact observed holo sequence, numbered 1..N in one chain. We rewrite each AF3
residue to its holo (chain, resseq, icode) using the strict 1:1 map from `sequence.chain_sequence`,
keeping AF3's coordinates and (standard) atom names. Output PDB is consumed by the reference
surface+descriptor pipeline under a parallel id `{PDBID}AF_{C1}_{C2}` — so all downstream
holo<->AF3 atom-identity mapping (by chain,resseq,name) is direct, exactly like the Phase-2 repack.

Dependency-free mmCIF `_atom_site` reader (no gemmi/biotite needed).
"""
from __future__ import annotations

from dataclasses import dataclass

from masif_graph.af3.sequence import Residue


@dataclass
class CifAtom:
    seq_id: int          # AF3 label_seq_id (1..N), the residue index in the predicted chain
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    resname: str
    b: float             # B-factor column (AF3 writes pLDDT here)


def read_cif_atoms(cif_path: str, model_chain: str | None = None) -> list[CifAtom]:
    """Parse ATOM records from an AF3 mmCIF `_atom_site` loop. Returns polymer (protein) atoms
    in file order. If model_chain is given, keep only that label_asym_id."""
    with open(cif_path) as fh:
        lines = fh.readlines()
    # find the _atom_site loop and its column order
    i = 0
    n = len(lines)
    atoms: list[CifAtom] = []
    while i < n:
        if lines[i].strip() == "loop_":
            # collect header tags
            j = i + 1
            tags = []
            while j < n and lines[j].lstrip().startswith("_atom_site."):
                tags.append(lines[j].strip())
                j += 1
            if tags and any(t == "_atom_site.Cartn_x" for t in tags):
                col = {t: k for k, t in enumerate(tags)}
                def gi(tag):
                    return col.get(tag, None)
                c_group = gi("_atom_site.group_PDB")
                c_atom = gi("_atom_site.label_atom_id")
                c_elem = gi("_atom_site.type_symbol")
                c_comp = gi("_atom_site.label_comp_id")
                c_asym = gi("_atom_site.label_asym_id")
                c_seq = gi("_atom_site.label_seq_id")
                c_x = gi("_atom_site.Cartn_x")
                c_y = gi("_atom_site.Cartn_y")
                c_z = gi("_atom_site.Cartn_z")
                c_b = gi("_atom_site.B_iso_or_equiv")
                k = j
                while k < n:
                    row = lines[k]
                    s = row.strip()
                    if s == "" or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                        break
                    parts = s.split()
                    if len(parts) < len(tags):
                        break
                    if c_group is not None and parts[c_group] not in ("ATOM", "HETATM"):
                        k += 1
                        continue
                    if model_chain is not None and c_asym is not None and parts[c_asym] != model_chain:
                        k += 1
                        continue
                    seq_raw = parts[c_seq] if c_seq is not None else "."
                    if seq_raw in (".", "?"):
                        k += 1
                        continue
                    name = parts[c_atom].strip('"')
                    elem = parts[c_elem].strip('"') if c_elem is not None else name[:1]
                    atoms.append(CifAtom(
                        seq_id=int(seq_raw),
                        atom_name=name,
                        element=elem,
                        x=float(parts[c_x]), y=float(parts[c_y]), z=float(parts[c_z]),
                        resname=parts[c_comp].strip('"') if c_comp is not None else "UNK",
                        b=float(parts[c_b]) if c_b is not None else 0.0,
                    ))
                    k += 1
                i = k
                continue
        i += 1
    return atoms


def _pdb_line(serial, name, resname, chain, resseq, icode, x, y, z, element, b):
    # standard PDB ATOM formatting; right-justify atom name per PDB convention
    nm = name
    if len(nm) < 4 and len(element) == 1 and not nm[:1].isdigit():
        nm = f" {nm:<3s}"
    else:
        nm = f"{nm:<4s}"
    return (
        f"ATOM  {serial:5d} {nm}{'':1s}{resname:>3s} {chain:1s}{int(resseq):4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {element:>2s}\n"
    )


def _kabsch(P, Q):
    """Rotation R + translation t mapping P onto Q (least squares): Q ~ P@R.T + t."""
    import numpy as np
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - P.mean(0) @ R.T
    return R, t


def relabel_af3_chain_to_pdb(
    cif_path: str,
    mapping_residues: list[Residue],
    holo_chain_id: str,
    out_pdb_path: str,
    model_chain: str | None = None,
    superpose_holo_ca: dict | None = None,
) -> tuple[int, int]:
    """Write AF3 chain atoms as PDB with holo numbering. Returns (n_af3_residues, n_mapped).

    Asserts the number of distinct AF3 residues == len(mapping_residues) (strict 1:1). AF3 predicts
    every input residue, so this must hold; a mismatch means the input sequence != what was predicted
    (raises), which would silently corrupt the atom map — better to fail loud.

    If `superpose_holo_ca` (a {resseq: holo_CA_xyz} map) is given, the AF3 coords are rigid-body
    superposed onto the holo chain over common CA atoms (Kabsch), placing the AF3 model in the holo
    complex frame. This is needed so two AF3 monomers (each centred at its own origin) don't overlap
    when assembled; descriptors are rotation/translation invariant so this does not change them."""
    import numpy as np
    atoms = read_cif_atoms(cif_path, model_chain=model_chain)
    seq_ids = sorted({a.seq_id for a in atoms})
    if len(seq_ids) != len(mapping_residues):
        raise ValueError(
            f"AF3 residue count {len(seq_ids)} != holo mapping {len(mapping_residues)} "
            f"({out_pdb_path}); refusing to relabel (would mismap identities)."
        )
    # AF3 seq_id i (1..N) -> mapping_residues[i-1]
    seqid_to_res = {sid: mapping_residues[k] for k, sid in enumerate(seq_ids)}

    R, t = None, None
    if superpose_holo_ca:
        af_ca, holo_ca = [], []
        for a in atoms:
            if a.atom_name == "CA":
                res = seqid_to_res[a.seq_id]
                if res.resseq in superpose_holo_ca:
                    af_ca.append([a.x, a.y, a.z])
                    holo_ca.append(superpose_holo_ca[res.resseq])
        if len(af_ca) >= 3:
            R, t = _kabsch(np.array(af_ca), np.array(holo_ca))

    serial = 0
    with open(out_pdb_path, "w") as out:
        for a in atoms:
            res = seqid_to_res[a.seq_id]
            serial += 1
            icode = res.icode if res.icode.strip() else " "
            x, y, z = a.x, a.y, a.z
            if R is not None:
                v = np.array([x, y, z]) @ R.T + t
                x, y, z = float(v[0]), float(v[1]), float(v[2])
            out.write(_pdb_line(serial, a.atom_name, a.resname, holo_chain_id,
                                 res.resseq, icode, x, y, z, a.element, a.b))
        out.write("TER\n")
    return len(seq_ids), len(mapping_residues)
