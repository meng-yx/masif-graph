"""Phase-6 unified atom featurizer: ONE feature space for protein AND ligand heavy atoms.

The Phase-4/5 encoder used a 14-D protein-only atom vector. To make protein↔protein complementarity
transfer to protein↔ligand (the neosurface goal), protein and ligand atoms must share the SAME node
features — describing *any* atom, not a residue. Layout (D=26):

  [0:10]  element one-hot  C,N,O,S,P,F,Cl,Br,I, other
  [10]    is_ligand                       (1 for ligand atoms, 0 for protein)
  [11]    is_backbone                     (protein backbone N/CA/C/O; 0 for ligand)
  [12]    aromatic
  [13]    degree / 6
  [14]    is_surface                      (atom owns >=1 surface vertex)
  [15]    in_ring
  [16:19] hybridization one-hot  sp, sp2, sp3
  [19]    hbond_donor
  [20]    hbond_acceptor
  [21]    formal_charge / 2               (clipped to [-2,2])
  [22]    flex_depth / 8                  (protein sidechain rotatability; 0 for ligand)
  [23:26] element-chem  [electronegativity, valence, covalent_radius]  (element-derived, any atom)

Protein side reuses the proven Phase-2 features (element/backbone/aromatic/degree/is_surface/flex) and
adds hybridization / H-bond / charge by simple, documented atom-name & element rules. Ligand side comes
straight from RDKit (mol2/sdf). Both return (n, 26); the encoder is retrained on this in C(c).
"""
from __future__ import annotations
import numpy as np
from masif_graph.m3.chem_graph import element_chem_features

U_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]  # + "other"
DIM = 26
_HYB = {"SP": 0, "SP2": 1, "SP3": 2}
_BACKBONE = {"N", "CA", "C", "O", "OXT"}
# protein H-bond rules by atom name (heavy-atom donors/acceptors)
_PROT_DONOR = {"N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OG", "OG1", "OH", "SG"}
_PROT_ACCEPTOR = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "ND1", "NE2", "OXT"}
_POS_ATOMS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"), ("HIS", "ND1")}
_NEG_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


def _elem_onehot(sym):
    v = np.zeros(len(U_ELEMENTS) + 1, np.float32)
    v[U_ELEMENTS.index(sym)] = 1.0 if sym in U_ELEMENTS else v.__setitem__(len(U_ELEMENTS), 1.0)
    if sym not in U_ELEMENTS:
        v[:] = 0.0; v[len(U_ELEMENTS)] = 1.0
    return v


def _hyb_onehot(name):
    v = np.zeros(3, np.float32)
    if name in _HYB:
        v[_HYB[name]] = 1.0
    return v


def ligand_features(mol, is_surface=None) -> np.ndarray:
    """RDKit mol (heavy atoms) -> (n, 26) unified features."""
    from rdkit import Chem
    HD = Chem.MolFromSmarts("[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]")
    HA = Chem.MolFromSmarts("[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),$([n;+0])]")
    don = set(sum(mol.GetSubstructMatches(HD), ())); acc = set(sum(mol.GetSubstructMatches(HA), ()))
    n = mol.GetNumAtoms()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    out = np.zeros((n, DIM), np.float32)
    for k, a in enumerate(mol.GetAtoms()):
        out[k, 0:10] = _elem_onehot(a.GetSymbol())
        out[k, 10] = 1.0                                  # is_ligand
        out[k, 12] = float(a.GetIsAromatic())
        out[k, 13] = min(a.GetDegree(), 6) / 6.0
        out[k, 14] = 1.0 if is_surface is None else float(is_surface[k])
        out[k, 15] = float(a.IsInRing())
        out[k, 16:19] = _hyb_onehot(str(a.GetHybridization()))
        out[k, 19] = float(a.GetIdx() in don)
        out[k, 20] = float(a.GetIdx() in acc)
        out[k, 21] = np.clip(a.GetFormalCharge(), -2, 2) / 2.0
        # flex (22) = 0 for ligand
    out[:, 23:26] = element_chem_features(np.array(syms))
    return out


def protein_features(elements, atom_names, resnames, aromatic, degree, is_surface, flex_depth) -> np.ndarray:
    """Protein heavy atoms (aligned arrays) -> (n, 26). Reuses Phase-2 signals; adds hybrid/hbond/charge by rule."""
    n = len(elements)
    out = np.zeros((n, DIM), np.float32)
    for k in range(n):
        e = str(elements[k]); nm = str(atom_names[k]); rn = str(resnames[k])
        out[k, 0:10] = _elem_onehot(e)
        # is_ligand=0
        out[k, 11] = 1.0 if nm in _BACKBONE else 0.0
        out[k, 12] = float(aromatic[k])
        out[k, 13] = min(int(degree[k]), 6) / 6.0
        out[k, 14] = float(is_surface[k])
        out[k, 15] = float(aromatic[k])                   # ring ~ aromatic for protein (approx)
        # hybridization by simple rule: aromatic->sp2; carbonyl/amide handled coarsely by element
        hyb = "SP2" if aromatic[k] else ("SP3" if e in ("C", "S") else "SP2")
        out[k, 16:19] = _hyb_onehot(hyb)
        out[k, 19] = 1.0 if nm in _PROT_DONOR else 0.0
        out[k, 20] = 1.0 if nm in _PROT_ACCEPTOR else 0.0
        q = 0.0
        if (rn, nm) in _POS_ATOMS: q = 0.5               # +1 shared over 2 atoms
        elif (rn, nm) in _NEG_ATOMS: q = -0.5
        out[k, 21] = q
        out[k, 22] = np.clip(flex_depth[k], 0, 8) / 8.0
    out[:, 23:26] = element_chem_features(np.asarray([str(e) for e in elements]))
    return out
