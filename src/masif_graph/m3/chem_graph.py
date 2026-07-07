"""M3 chemistry-aware atom graph = Phase-2 covalent graph + element-derived chemistry node features.

Reuses `graph.build.build_atom_graph` (biotite covalent connectivity + bond-order one-hot +
sidechain-rotatable flag + flex-depth + element/aromatic/degree nodes — all conformation-INVARIANT)
and appends the element-derived chemistry the user flagged AtomSurf as missing: **electronegativity,
typical valence, covalent radius**. These are intrinsic per-element scalars that (with bond order +
connectivity) shape the local conformational landscape, and are invariant to holo<->AF3 conformation.

Deliberately does NOT add pose-sensitive spatial/distance edges to the invariant channel (Phase-2
showed those inject conformation-sensitivity). The surface encoder carries the (moving) geometry.
"""
from __future__ import annotations

import numpy as np

from masif_graph.graph.build import build_atom_graph, _ELEMENTS

# Pauling electronegativity, typical valence, covalent radius (Å) — heavy elements we see.
_ELECTRONEG = {"C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "P": 2.19, "Se": 2.55,
               "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66, "Fe": 1.83, "Zn": 1.65,
               "Mg": 1.31, "Ca": 1.00, "Mn": 1.55, "Cu": 1.90, "Na": 0.93, "K": 0.82}
_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 5, "Se": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
_COV_RADIUS = {"C": 0.77, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "Se": 1.20, "F": 0.64,
               "Cl": 0.99, "Br": 1.14, "I": 1.33}


def element_chem_features(elements: np.ndarray) -> np.ndarray:
    """(n,3) normalized [electronegativity, valence, covalent_radius] per atom from its element."""
    out = np.zeros((len(elements), 3), dtype=np.float32)
    for k, e in enumerate(elements):
        e = str(e)
        out[k, 0] = _ELECTRONEG.get(e, 2.5) / 4.0        # ~[0,1]
        out[k, 1] = _VALENCE.get(e, 3) / 6.0             # ~[0,1]
        out[k, 2] = _COV_RADIUS.get(e, 0.9) / 1.5        # ~[0,1]
    return out


def build_chem_graph(chain, surf, pdb_path: str, **kw):
    """Phase-2 AtomGraph with 3 extra element-chemistry node features appended to `node_feat`.

    Returns the same AtomGraph object; its `node_feat` is widened from F_base(10) to F_base+3(13).
    The extra channels are conformation-invariant (element-intrinsic)."""
    g = build_atom_graph(chain, surf, pdb_path, **kw)
    # heavy-atom elements aligned to graph nodes: the graph nodes are the io.reference atoms of `chain`
    chem = element_chem_features(chain.atom_element)
    if chem.shape[0] != g.node_feat.shape[0]:
        # align by count; build_atom_graph nodes == chain heavy atoms (io.reference table)
        m = min(chem.shape[0], g.node_feat.shape[0])
        chem = chem[:m]
        pad = np.zeros((g.node_feat.shape[0], 3), dtype=np.float32)
        pad[:m] = chem
        chem = pad
    g.node_feat = np.concatenate([g.node_feat, chem], axis=1)
    return g
