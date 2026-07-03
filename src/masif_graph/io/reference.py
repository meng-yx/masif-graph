"""Adapters for reading the reference `masif-neosurf-af2` pipeline outputs.

Phase-1 consumes the *frozen* reference pipeline (run inside the `.sif`) as a tool. This
module reads its outputs from the new `masif-graph` conda env (no pymesh / TF needed).

Key alignment fact (verified against `read_data_from_surface.py`, which loops vertices in
order with no reordering/filtering): for a chain, row ``i`` of the descriptor arrays
corresponds to vertex ``i``, whose coordinate is ``(pK_X[i], pK_Y[i], pK_Z[i])`` and whose
interface label is ``pK_iface_labels[i]``. We therefore read vertex coordinates from the
precompute ``.npy`` (guaranteed aligned to descriptors) rather than re-reading the ``.ply``.
Per-vertex normals (only needed for M2 ICP) come from the ``.ply`` via ``plyfile`` and are
consistency-checked against the precompute coordinates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Root of the reference pipeline's data tree (all masif_opts paths are relative to here).
REF_DATA_ROOT = (
    "/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search"
)
SURFACE_DIR = os.path.join(REF_DATA_ROOT, "data_preparation/01-benchmark_surfaces")
PDB_DIR = os.path.join(REF_DATA_ROOT, "data_preparation/01-benchmark_pdbs")
PRECOMP_DIR = os.path.join(
    REF_DATA_ROOT, "data_preparation/04b-precomputation_12A/precomputation"
)
DESC_DIR = os.path.join(REF_DATA_ROOT, "descriptors/sc05/all_feat")

# Elements never counted as "heavy" for the surface-atom definition.
_HYDROGEN = {"H", "D"}
# Residue names to drop when collecting heavy atoms (solvent).
_SOLVENT = {"HOH", "WAT", "DOD", "TIP", "SOL"}


@dataclass
class Chain:
    """One preprocessed protein chain (p1 or p2 of a complex)."""

    complex_id: str
    pid: str  # "p1" or "p2"
    pdb_id: str
    chain_ids: str  # e.g. "A" or "HL"
    verts: np.ndarray  # (n_vert, 3) vertex coordinates (== descriptor row order)
    desc_straight: np.ndarray  # (n_vert, 80)
    desc_flipped: np.ndarray  # (n_vert, 80)
    iface: np.ndarray  # (n_vert,) reference MaSIF-site interface label per vertex
    sc: np.ndarray  # (n_vert,) per-vertex shape-complementarity (median of sc_labels[0]); NaN if absent
    atom_coords: np.ndarray  # (n_atom, 3) heavy-atom coordinates
    atom_element: np.ndarray  # (n_atom,) element symbols
    atom_resid: np.ndarray  # (n_atom,) "chain:resseq:resname" identifiers
    atom_name: np.ndarray  # (n_atom,) atom names

    @property
    def n_vert(self) -> int:
        return len(self.verts)

    @property
    def n_atom(self) -> int:
        return len(self.atom_coords)


def _complex_fields(complex_id: str):
    parts = complex_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"expected PDBID_C1_C2, got {complex_id!r}")
    return parts[0], parts[1], parts[2]


def parse_heavy_atoms(pdb_path: str):
    """Parse heavy atoms from a PDB file (self-contained; no biopython dependency).

    Returns (coords[m,3], element[m], resid[m], name[m]). Hydrogens/deuteriums and solvent
    residues are excluded. Alternate location indicator 'B'+ is skipped (keeps altloc ' '/'A').
    """
    coords, elements, resids, names = [], [], [], []
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
            name = line[12:16].strip()
            element = line[76:78].strip()
            if not element:  # fall back to inferring element from the atom name
                element = "".join(c for c in name if c.isalpha())[:2].capitalize()
            el_norm = element.strip().upper()
            if el_norm in _HYDROGEN or (el_norm == "" and name[:1] == "H"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            coords.append((x, y, z))
            elements.append(element.capitalize())
            resids.append(f"{chain}:{resseq}:{resname}")
            names.append(name)
    return (
        np.asarray(coords, dtype=np.float64).reshape(-1, 3),
        np.asarray(elements, dtype=object),
        np.asarray(resids, dtype=object),
        np.asarray(names, dtype=object),
    )


def _load_chain(complex_id: str, pid: str, pdb_id: str, chain_ids: str) -> Chain:
    pc = os.path.join(PRECOMP_DIR, complex_id)
    dd = os.path.join(DESC_DIR, complex_id)
    x = np.load(os.path.join(pc, f"{pid}_X.npy"))
    y = np.load(os.path.join(pc, f"{pid}_Y.npy"))
    z = np.load(os.path.join(pc, f"{pid}_Z.npy"))
    verts = np.column_stack([x, y, z]).astype(np.float64)
    desc_straight = np.load(os.path.join(dd, f"{pid}_desc_straight.npy"))
    desc_flipped = np.load(os.path.join(dd, f"{pid}_desc_flipped.npy"))
    iface = np.load(os.path.join(pc, f"{pid}_iface_labels.npy"))

    # Shape-complementarity per vertex: median over axis=1 of sc_labels[0] (reference convention
    # from masif_ppi_search_comp_desc.py). Used to reproduce the reference sc-filtered positives.
    sc_path = os.path.join(pc, f"{pid}_sc_labels.npy")
    if os.path.exists(sc_path):
        sc_raw = np.load(sc_path)
        sc = np.median(sc_raw[0], axis=1)
    else:
        sc = np.full(len(verts), np.nan)

    n = len(verts)
    if not (desc_straight.shape[0] == desc_flipped.shape[0] == len(iface) == n):
        raise ValueError(
            f"{complex_id} {pid}: row-count mismatch verts={n} "
            f"straight={desc_straight.shape} flipped={desc_flipped.shape} iface={len(iface)}"
        )
    if desc_straight.shape[1] != 80 or desc_flipped.shape[1] != 80:
        raise ValueError(f"{complex_id} {pid}: descriptor dim != 80")

    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}_{chain_ids}.pdb")
    atom_coords, atom_element, atom_resid, atom_name = parse_heavy_atoms(pdb_path)
    if len(atom_coords) == 0:
        raise ValueError(f"{complex_id} {pid}: no heavy atoms parsed from {pdb_path}")
    return Chain(
        complex_id=complex_id, pid=pid, pdb_id=pdb_id, chain_ids=chain_ids,
        verts=verts, desc_straight=desc_straight, desc_flipped=desc_flipped, iface=iface,
        sc=sc, atom_coords=atom_coords, atom_element=atom_element, atom_resid=atom_resid,
        atom_name=atom_name,
    )


def load_complex(complex_id: str):
    """Load both chains (p1, p2) of a complex. Returns (Chain, Chain)."""
    pdb_id, c1, c2 = _complex_fields(complex_id)
    p1 = _load_chain(complex_id, "p1", pdb_id, c1)
    p2 = _load_chain(complex_id, "p2", pdb_id, c2)
    return p1, p2


def complex_is_available(complex_id: str) -> bool:
    """True iff all required reference outputs for a complex exist and are non-empty."""
    pdb_id, c1, c2 = _complex_fields(complex_id)
    dd = os.path.join(DESC_DIR, complex_id)
    pc = os.path.join(PRECOMP_DIR, complex_id)
    required = [
        os.path.join(dd, "p1_desc_straight.npy"), os.path.join(dd, "p1_desc_flipped.npy"),
        os.path.join(dd, "p2_desc_straight.npy"), os.path.join(dd, "p2_desc_flipped.npy"),
        os.path.join(pc, "p1_X.npy"), os.path.join(pc, "p2_X.npy"),
        os.path.join(pc, "p1_iface_labels.npy"), os.path.join(pc, "p2_iface_labels.npy"),
        os.path.join(PDB_DIR, f"{pdb_id}_{c1}.pdb"),
        os.path.join(PDB_DIR, f"{pdb_id}_{c2}.pdb"),
    ]
    return all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required)


def load_ply_normals(pdb_id: str, chain_ids: str, expected_coords: np.ndarray | None = None):
    """Read per-vertex normals from the reference `.ply` (for M2 ICP).

    If `expected_coords` is given, asserts the `.ply` vertex order matches (so normals align
    with the descriptor rows). Requires `plyfile`.
    """
    from plyfile import PlyData

    ply_path = os.path.join(SURFACE_DIR, f"{pdb_id}_{chain_ids}.ply")
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    coords = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    normals = np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float64)
    if expected_coords is not None:
        if coords.shape != expected_coords.shape or not np.allclose(
            coords, expected_coords, atol=1e-3
        ):
            raise ValueError(
                f"{pdb_id}_{chain_ids}: .ply vertex order != precompute coords "
                "(normals would be misaligned)"
            )
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return normals / norm
