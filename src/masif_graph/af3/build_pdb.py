"""CLI: relabel one AF3 chain model (mmCIF) to a holo-numbered PDB chain.

Usage: python -m masif_graph.af3.build_pdb <pdb_id> <chain> <cif_path> <out_pdb> [holo_chain_id]

The holo sequence/mapping is re-derived from the holo per-chain PDB (deterministic), so nothing
needs to be persisted between prepare and here. Prints a status line the batch script greps.
"""
from __future__ import annotations

import os
import sys

from masif_graph.af3.sequence import chain_sequence
from masif_graph.af3.relabel import relabel_af3_chain_to_pdb
from masif_graph.io.reference import PDB_DIR, parse_heavy_atoms


def holo_ca_map(pdb_id, chain):
    """{resseq: CA_xyz} for the holo chain, to superpose the AF3 model into the holo frame."""
    coords, _elem, resid, name = parse_heavy_atoms(os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb"))
    out = {}
    for c, r, nm in zip(coords, resid, name):
        if str(nm) == "CA":
            _chn, seq, _rn = r.split(":")
            out[seq] = c
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    pdb_id, chain, cif_path, out_pdb = argv[:4]
    holo_chain_id = argv[4] if len(argv) > 4 else chain
    holo_pdb = os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb")
    _seq, mapres = chain_sequence(holo_pdb)
    # superpose AF3 onto holo (into the holo complex frame) so assembled chains don't overlap
    holo_ca = holo_ca_map(pdb_id, chain)
    try:
        n_af3, n_map = relabel_af3_chain_to_pdb(cif_path, mapres, holo_chain_id, out_pdb,
                                                superpose_holo_ca=holo_ca)
    except Exception as e:
        print(f"RELABEL_STATUS {pdb_id}_{chain} FAIL {type(e).__name__}:{e}", flush=True)
        return 2
    print(f"RELABEL_STATUS {pdb_id}_{chain} OK n_res={n_af3} mapped={n_map} superposed={len(holo_ca)>0} "
          f"-> {out_pdb}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
