"""Extract the observed sequence + ordered residue list from a holo per-chain PDB.

This is the backbone of the leak-free holo<->AF3 residue map. We parse residues in the order
their first atom appears (standard PDB order == sequence order), keeping the FULL residue key
(chain, resseq, icode, resname) so insertion codes are not silently collapsed (the generic
`io.reference.parse_heavy_atoms` drops icode, which would mismap 100/100A).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# 3->1 letter, including the common modified residues AF3 will see as their standard parent.
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # modified / alt names -> standard parent
    "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H", "HSP": "H", "CSO": "C",
    "PTR": "Y", "SEP": "S", "TPO": "T", "MLY": "K", "KCX": "K", "LLP": "K", "CME": "C",
    "OCS": "C", "CSD": "C", "CAS": "C", "FME": "M", "NLE": "L", "ABA": "A", "ORN": "K",
}
_SOLVENT = {"HOH", "WAT", "DOD", "TIP", "SOL"}
_HYDROGEN = {"H", "D"}


@dataclass
class Residue:
    chain: str
    resseq: str   # numeric field, as string (col 23-26)
    icode: str    # insertion code (col 27), " " if none
    resname: str  # 3-letter
    one: str      # 1-letter (X if unknown)

    @property
    def key(self) -> tuple:
        """Holo identity key used to relabel AF3 atoms and to intersect atom sets."""
        return (self.chain, self.resseq, self.icode)


def chain_residues(pdb_path: str) -> list[Residue]:
    """Return residues in first-atom order (== sequence order) from a per-chain PDB.

    Includes standard AAs and known modified residues (mapped to their parent one-letter);
    excludes solvent and any residue with no mappable heavy atoms. A residue whose resname is
    not in AA3TO1 gets one='X' (flagged by the caller if numerous)."""
    residues: list[Residue] = []
    seen: set[tuple] = set()
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            if resname in _SOLVENT:
                continue
            # skip pure-hydrogen lines when deciding residue element presence is unnecessary here;
            # residue identity is what we track.
            element = line[76:78].strip().upper()
            name = line[12:16].strip()
            if element in _HYDROGEN or (element == "" and name[:1] == "H"):
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            one = AA3TO1.get(resname, "X")
            residues.append(Residue(chain, resseq, icode, resname, one))
    return residues


def chain_sequence(pdb_path: str) -> tuple[str, list[Residue]]:
    """(sequence_string, mapping_residues) with a strict 1:1 alignment: mapping_residues[i] is the
    holo residue AF3 position i (0-based) will correspond to. Non-standard residues that map to 'X'
    are excluded from BOTH the sequence and the mapping list (they will simply be absent from the
    AF3 model and thus fall out of the holo<->AF3 intersection — correct, not a leak). Callers get
    the full/X diagnostics via `chain_residues` / `n_nonstandard`."""
    all_res = chain_residues(pdb_path)
    mapping_residues = [r for r in all_res if r.one != "X"]
    seq = "".join(r.one for r in mapping_residues)
    assert len(seq) == len(mapping_residues)
    return seq, mapping_residues


def n_nonstandard(residues: list[Residue]) -> int:
    return sum(1 for r in residues if r.one == "X")
