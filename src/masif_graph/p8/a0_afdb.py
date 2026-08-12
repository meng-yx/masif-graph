"""Phase-8 A0 / fork F4 — how much of our corpus can AFDB actually cover?

The user's Option A (1:1 holo:apo, AFDB exact match with a local-prediction fallback) is only
attractive if AFDB covers most of the corpus; if coverage is poor the fallback IS the pipeline and
Option A collapses into Option B. That is a factual question about our chain list, answerable on a
login node for free, and it is a direct input to the D8-12 decision the user makes at the PAUSE.

Method
  1. RCSB GraphQL (batched, 100 entries per request) maps every PDB entry in our lists to its
     polymer entities: auth chain ids, UniProt accessions, and the canonical entity sequence.
  2. The AlphaFold DB API is queried once per unique accession for `uniprotSequence`.
  3. Each chain we actually use is classified:
       exact         AFDB sequence == PDB entity canonical sequence
       subsequence   the PDB construct is a contiguous piece of the AFDB sequence (domain/fragment)
       ident>=95     global alignment identity >= 95%   (point mutants, tags)
       ident<95      an AFDB entry exists but is not the same protein construct
       no_afdb       UniProt known, no AFDB model
       no_uniprot    no UniProt cross-reference (designed/engineered constructs, chimeras)
       not_protein   non-protein polymer entity

Everything is cached to disk, so a rerun costs no network traffic and the counts are reproducible.

Usage:
  python -m masif_graph.p8.a0_afdb --lists data/lists/training.txt data/lists/testing.txt \
      --out logs/phase8A/a0/afdb_coverage.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "masif-graph/0.1 (academic research; contact lab.thoma@gmail.com)"}
GRAPHQL = "https://data.rcsb.org/graphql"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/{}"

Q = """{ entries(entry_ids: [%s]) { rcsb_id
  polymer_entities { rcsb_id
    rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    rcsb_polymer_entity_align { reference_database_name reference_database_accession }
    entity_poly { pdbx_seq_one_letter_code_can rcsb_entity_polymer_type } } } }"""


def _get(url, timeout=90, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as f:
                return json.load(f)
        except Exception as e:                                          # noqa: BLE001
            if k == tries - 1:
                return {"__error__": f"{type(e).__name__}: {e}"}
            time.sleep(1.5 * (k + 1))
    return {"__error__": "unreachable"}


def fetch_entries(entry_ids, cache_path, batch=100):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [e for e in entry_ids if e not in cache]
    print(f"RCSB: {len(entry_ids)} entries, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = Q % ",".join(f'"{e}"' for e in chunk)
        d = _get(GRAPHQL + "?query=" + urllib.parse.quote(q))
        got = {}
        for e in ((d.get("data") or {}).get("entries") or []):
            if e:
                got[e["rcsb_id"].upper()] = e
        for e in chunk:                       # record misses so we do not refetch them forever
            cache[e] = got.get(e.upper())
        json.dump(cache, open(cache_path, "w"))
        print(f"  {min(i+batch, len(todo))}/{len(todo)}  (batch hit {len(got)}/{len(chunk)})", flush=True)
        time.sleep(0.2)
    return cache


def fetch_afdb(accs, cache_path, workers=6):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [a for a in accs if a not in cache]
    print(f"AFDB: {len(accs)} accessions, {len(todo)} to fetch", flush=True)

    def one(acc):
        d = _get(AFDB.format(acc), timeout=60)
        if isinstance(d, list) and d:
            return acc, {"seq": d[0].get("uniprotSequence", ""), "v": d[0].get("latestVersion")}
        return acc, None                                                # no model (or error)

    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for acc, v in ex.map(one, chunk):
                cache[acc] = v
        json.dump(cache, open(cache_path, "w"))
        print(f"  {min(i+200, len(todo))}/{len(todo)}", flush=True)
    return cache


def _identity(a, b):
    """Global alignment identity, normalised by the shorter sequence."""
    try:
        import biotite.sequence as bseq
        import biotite.sequence.align as balign
        s1, s2 = bseq.ProteinSequence(a), bseq.ProteinSequence(b)
        mat = balign.SubstitutionMatrix.std_protein_matrix()
        aln = balign.align_optimal(s1, s2, mat, gap_penalty=(-10, -1), terminal_penalty=False)[0]
        return float(balign.get_sequence_identity(aln, mode="shortest"))
    except Exception:                                                   # noqa: BLE001
        return float("nan")


def classify(entity_seq, afdb_seq, max_align_len=2000):
    if not afdb_seq:
        return "no_afdb", float("nan")
    if entity_seq == afdb_seq:
        return "exact", 1.0
    if entity_seq and entity_seq in afdb_seq:
        return "subsequence", 1.0
    if len(entity_seq) > max_align_len or len(afdb_seq) > max_align_len:
        return "too_long_to_align", float("nan")
    ident = _identity(entity_seq, afdb_seq)
    if ident != ident:                                                  # nan
        return "align_failed", ident
    return ("ident>=95" if ident >= 0.95 else "ident<95"), ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=["data/lists/training.txt", "data/lists/testing.txt"])
    ap.add_argument("--cache-dir", default="logs/phase8A/a0/cache")
    ap.add_argument("--out", default="logs/phase8A/a0/afdb_coverage.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-retry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    # One line is PDBID_sideA_sideB, and a SIDE may concatenate several chains: `1A14_HL_N` is an
    # antibody heavy+light chain against an antigen. Expanding each side into its constituent
    # single-letter chain ids is required — treating "HL" as one chain id silently loses ~11% of
    # the corpus to a fake "not found" class.
    chains, multi = set(), 0
    for p in args.lists:
        for ln in open(p):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            f = ln.split("_")
            if len(f) >= 3:
                for side in f[1:3]:
                    if len(side) > 1:
                        multi += 1
                    for c in side:
                        chains.add((f[0].upper(), c))
    entries = sorted({e for e, _ in chains})
    if args.limit:
        entries = entries[: args.limit]
        chains = {(e, c) for e, c in chains if e in set(entries)}
    print(f"{len(chains)} chain instances across {len(entries)} PDB entries "
          f"({multi} multi-chain interface sides expanded)", flush=True)

    cache_p = os.path.join(args.cache_dir, "rcsb_entries.json")
    ent = fetch_entries(entries, cache_p)
    # Retry entries the batch query returned nothing for, one at a time — a null inside a large
    # batch is usually a transient GraphQL hiccup, not an obsolete entry, and silently keeping it
    # null would understate coverage.
    nulls = [e for e in entries if ent.get(e) is None]
    if nulls and not args.no_retry:
        print(f"retrying {len(nulls)} entries individually", flush=True)
        for e in nulls:                    # drop the cached miss so the refetch actually happens
            ent.pop(e, None)
        json.dump(ent, open(cache_p, "w"))
        ent = fetch_entries(nulls, cache_p, batch=1)
        print(f"  still missing after retry: {sum(ent.get(e) is None for e in entries)}", flush=True)

    # chain -> (entity seq, accession)
    chain_info, accs = {}, set()
    for e in entries:
        rec = ent.get(e)
        if not rec:
            continue
        for pe in rec.get("polymer_entities") or []:
            ci = pe.get("rcsb_polymer_entity_container_identifiers") or {}
            ep = pe.get("entity_poly") or {}
            al = pe.get("rcsb_polymer_entity_align") or []
            acc = [a["reference_database_accession"] for a in al
                   if a.get("reference_database_name") == "UniProt"]
            for c in ci.get("auth_asym_ids") or []:
                chain_info[(e, c)] = {
                    "seq": ep.get("pdbx_seq_one_letter_code_can") or "",
                    "type": ep.get("rcsb_entity_polymer_type"),
                    "acc": acc,
                }
            accs.update(acc)

    af = fetch_afdb(sorted(accs), os.path.join(args.cache_dir, "afdb.json"))

    rows, counts = [], {}
    for key in sorted(chains):
        info = chain_info.get(key)
        if info is None:
            cls, ident, acc = "entry_or_chain_not_found", float("nan"), None
        elif info["type"] != "Protein":
            cls, ident, acc = "not_protein", float("nan"), None
        elif not info["acc"]:
            cls, ident, acc = "no_uniprot", float("nan"), None
        else:
            # a chimeric entity can carry several accessions; take the best-matching one
            best = ("no_afdb", float("nan"), info["acc"][0])
            for a in info["acc"]:
                rec = af.get(a)
                c, i = classify(info["seq"], (rec or {}).get("seq", ""))
                rank = {"exact": 0, "subsequence": 1, "ident>=95": 2, "ident<95": 3,
                        "too_long_to_align": 4, "align_failed": 5, "no_afdb": 6}
                if rank.get(c, 9) < rank.get(best[0], 9):
                    best = (c, i, a)
            cls, ident, acc = best
        counts[cls] = counts.get(cls, 0) + 1
        rows.append({"pdb": key[0], "chain": key[1], "class": cls,
                     "identity": None if ident != ident else round(ident, 4), "acc": acc})

    n = len(rows)
    usable = sum(counts.get(k, 0) for k in ("exact", "subsequence", "ident>=95"))
    out = {"n_chain_instances": n, "n_entries": len(entries), "counts": counts,
           "fractions": {k: round(v / n, 4) for k, v in sorted(counts.items())},
           "afdb_usable_fraction": round(usable / n, 4),
           "afdb_usable_definition": "exact + subsequence + ident>=95",
           "per_chain": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("=" * 78)
    print(f"AFDB COVERAGE of the PPI corpus: {n} chain instances, {len(entries)} entries")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:26s} {v:6d}  {100*v/n:5.1f}%")
    print(f"  >>> usable (exact|subsequence|>=95%): {usable}/{n} = {100*usable/n:.1f}%")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
