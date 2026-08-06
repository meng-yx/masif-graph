#!/usr/bin/env python
"""Phase-5 M0c step 1: build a per-chain FASTA for all complexes in the MaSIF-search list.

Fetches RCSB entry FASTA (batched, cached), maps author-chain -> sequence, and emits one FASTA
record per (complex, side, chain) so we can cluster at chain level and derive a sequence-cluster-
clean test split (test clusters disjoint from train). Robust to multi-chain sides (e.g. 1A14_HL_N)
and to RCSB's 'C[auth A]' author-chain notation.

Usage: p5_build_split.py fetch   # fetch+parse -> logs/phase5/all_chains.fasta (+ missing report)
"""
from __future__ import annotations
import os, re, sys, time, urllib.request, urllib.error

ROOT = "/scratch/ymeng/masif-graph"
LISTS = f"{ROOT}/masif-neosurf-af2/masif/data/masif_ppi_search/lists"
CACHE = f"{ROOT}/tools/rcsb_fasta"
OUT = f"{ROOT}/logs/phase5/all_chains.fasta"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def read_list(name):
    return [l.strip() for l in open(f"{LISTS}/{name}") if l.strip()]


def complexes():
    seen = {}
    for split in ("training.txt", "testing.txt"):
        for cid in read_list(split):
            seen[cid] = split.replace(".txt", "")
    return seen  # cid -> 'training'|'testing'


def fetch_batch(pdb_ids):
    """Fetch a comma-separated batch of entry FASTA; return raw text (cached per single id)."""
    url = "https://www.rcsb.org/fasta/entry/" + ",".join(pdb_ids)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception as e:
            if attempt == 3:
                print(f"  batch fail {pdb_ids[0]}..({len(pdb_ids)}): {e}", flush=True)
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def parse_fasta(text):
    """RCSB entry FASTA -> {PDBID(upper): {auth_chain: sequence}}."""
    out = {}
    cur_pdb = None
    cur_chains = []
    cur_seq = []

    def flush():
        if cur_pdb and cur_chains and cur_seq:
            seq = "".join(cur_seq)
            d = out.setdefault(cur_pdb, {})
            for ch in cur_chains:
                d[ch] = seq

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            cur_seq = []
            # header: >1A0G_1|Chains A, B|desc|org   OR  |Chain C[auth A]|
            m = re.match(r">(\w{4})_\d+\|Chains?\s+([^|]+)\|", line)
            if not m:
                cur_pdb, cur_chains = None, []
                continue
            cur_pdb = m.group(1).upper()
            chain_field = m.group(2)
            chains = []
            for tok in chain_field.split(","):
                tok = tok.strip()
                a = re.search(r"\[auth\s+([A-Za-z0-9]+)\]", tok)  # prefer auth chain
                if a:
                    chains.append(a.group(1))
                else:
                    chains.append(tok.split()[0])
            cur_chains = chains
        else:
            cur_seq.append(line.strip())
    flush()
    return out


def main():
    cmap = complexes()
    pdbs = sorted({cid.split("_")[0].upper() for cid in cmap})
    print(f"{len(cmap)} complexes, {len(pdbs)} unique PDBs", flush=True)

    # fetch (batched, cached by whole-batch file)
    seqs = {}  # PDBID -> {chain: seq}
    B = 50
    todo = [p for p in pdbs]
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        cachef = f"{CACHE}/batch_{i//B:04d}.fasta"
        if os.path.exists(cachef) and os.path.getsize(cachef) > 0:
            text = open(cachef).read()
        else:
            text = fetch_batch(batch)
            if text:
                open(cachef, "w").write(text)
            time.sleep(0.2)
        seqs.update(parse_fasta(text))
        if (i // B) % 10 == 0:
            print(f"  fetched {i+len(batch)}/{len(todo)} pdbs; parsed {len(seqs)} entries", flush=True)

    # emit per-(complex, side, chain) records
    n_rec, missing = 0, []
    with open(OUT, "w") as fo:
        for cid, split in cmap.items():
            pdb = cid.split("_")[0].upper()
            sides = cid.split("_")[1:]  # e.g. ['HL','N']
            chdict = seqs.get(pdb, {})
            for si, side in enumerate(sides, 1):
                for ch in side:  # each character is an author chain id
                    s = chdict.get(ch)
                    if not s or len(s) < 20:
                        missing.append(f"{cid}\t{side}\t{ch}")
                        continue
                    fo.write(f">{cid}|s{si}:{side}|{ch}\n{s}\n")
                    n_rec += 1
    print(f"wrote {n_rec} chain records to {OUT}", flush=True)
    print(f"missing chains: {len(missing)}", flush=True)
    if missing:
        open(f"{ROOT}/logs/phase5/split_missing_chains.txt", "w").write("\n".join(missing))
        print("  (see logs/phase5/split_missing_chains.txt)", flush=True)


if __name__ == "__main__":
    main()
