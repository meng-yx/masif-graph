"""Prepare AF3 monomer inputs for a list of complexes.

For each complex `{PDBID}_{C1}_{C2}` we emit one AF3 input JSON per chain (monomer prediction of
the holo observed sequence). The JSON name == `{PDBID}_{chain}` so the AF3 output dir is predictable.
We also write a manifest of (complex, chain, seq_len, n_nonstd) for bookkeeping.
"""
from __future__ import annotations

import argparse
import json
import os

from masif_graph.af3.sequence import chain_sequence, chain_residues, n_nonstandard
from masif_graph.io.reference import PDB_DIR, complex_is_available


def chain_json_name(pdb_id: str, chain: str) -> str:
    return f"{pdb_id}_{chain}"


def write_chain_json(pdb_id: str, chain: str, out_dir: str, seeds=(1,)) -> dict | None:
    holo_pdb = os.path.join(PDB_DIR, f"{pdb_id}_{chain}.pdb")
    if not os.path.exists(holo_pdb):
        return {"pdb_id": pdb_id, "chain": chain, "status": "missing_pdb"}
    seq, mapres = chain_sequence(holo_pdb)
    nx = n_nonstandard(chain_residues(holo_pdb))
    if len(seq) < 16:
        return {"pdb_id": pdb_id, "chain": chain, "status": "too_short", "seq_len": len(seq)}
    name = chain_json_name(pdb_id, chain)
    obj = {
        "name": name,
        "modelSeeds": list(seeds),
        "sequences": [{"protein": {"id": "A", "sequence": seq}}],
        "dialect": "alphafold3",
        "version": 1,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(obj, fh)
    return {"pdb_id": pdb_id, "chain": chain, "status": "ok", "seq_len": len(seq),
            "n_nonstd": nx, "name": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="file with complex ids PDBID_C1_C2")
    ap.add_argument("--out", required=True, help="dir for AF3 input JSONs")
    ap.add_argument("--manifest", required=True, help="output manifest json")
    ap.add_argument("--seeds", default="1", help="comma-separated model seeds")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    ids = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]
    manifest = []
    seen_chains = set()
    for cid in ids:
        if not complex_is_available(cid):
            manifest.append({"complex": cid, "status": "holo_unavailable"})
            continue
        pdb_id, c1, c2 = cid.split("_")
        for chain in (c1, c2):
            if (pdb_id, chain) in seen_chains:
                continue
            seen_chains.add((pdb_id, chain))
            rec = write_chain_json(pdb_id, chain, args.out, seeds)
            rec["complex"] = cid
            manifest.append(rec)
            print(f"{cid} {pdb_id}_{chain}: {rec.get('status')} len={rec.get('seq_len')}", flush=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    ok = sum(1 for m in manifest if m.get("status") == "ok")
    print(f"\nwrote {ok} chain JSONs to {args.out}; manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
