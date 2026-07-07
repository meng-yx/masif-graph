"""Phase-3 AF3 helpers: build AF3 monomer inputs from holo chains, relabel AF3 models back
to holo residue identity, and drive the reference surface+descriptor pipeline on AF3 models.

The central invariant: an AF3 monomer is predicted from the *exact observed sequence* of a holo
chain (residues in PDB order), so AF3 residue position i (1-based) corresponds to the i-th holo
observed residue. `relabel` rewrites AF3's 1..N numbering back to the holo (chain, resseq, icode)
so all downstream atom-identity mapping (holo <-> AF3) is direct — exactly like the Phase-2 repack.
"""
