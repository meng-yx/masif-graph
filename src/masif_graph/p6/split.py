"""Phase-6 C(b).2 — cluster-clean split over the combined PPI + protein-ligand corpus.

Two leakage axes, per the design (`docs/16 §5`): protein **sequence cluster** (mmseqs2 @30% id)
and ligand **scaffold**. Phase 5 shipped a real train/eval leak, so the rule here is: filter
against the **actual** training ids, never the intended ones, and check afterwards.

Three id sets come out:
  * `eval_ppi`  — the frozen Phase-5 287-clean list; the do-no-harm gate. Its protein clusters are
                  *forbidden* everywhere else, in the PPI **and** the PDBbind corpus (a PDBbind
                  target homologous to an eval chain would leak into the gate just as surely).
  * `train`     — PPI complexes + PDBbind complexes, all clusters disjoint from `eval_ppi`.
  * `val_pl`    — held-out protein-ligand complexes for the mixed-held-out axis.

`val_pl` is carved by connected components of the "shares a protein cluster OR a ligand scaffold"
graph, so a val complex can be neither a homolog nor a congener of anything trained on. Scaffold
edges can fuse everything into one giant component (benzene is a scaffold), so the builder reports
component sizes and falls back to protein-cluster components alone if the scaffold graph is
degenerate — recording which rule was used, and always reporting the scaffold-unseen subset of the
val set separately so the stricter number stays visible.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

ROOT = "/scratch/ymeng/masif-graph"
MMSEQS = f"{ROOT}/tools/mmseqs/bin/mmseqs"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
}


# ---------------------------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------------------------
def pdb_sequence(pdb_path: str) -> str:
    """One-letter sequence of a chain PDB, in file order (CA atoms, standard + common modified)."""
    seq, seen = [], set()
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6] not in ("ATOM  ", "HETATM") or line[12:16].strip() != "CA":
                continue
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            seq.append(AA3.get(line[17:20].strip().upper(), "X"))
    return "".join(seq)


def write_combined_fasta(ppi_fasta, pl_ids, chain_pdb_dir, out_fasta, min_len=20):
    """PPI per-chain records (Phase-5 FASTA) + one record per PDBbind complex -> one FASTA.

    Record ids are `cid|...`; everything downstream only needs the leading `cid`."""
    n_ppi = n_pl = 0
    with open(out_fasta, "w") as out:
        with open(ppi_fasta) as fh:
            for line in fh:
                out.write(line)
                n_ppi += line.startswith(">")
        for pid in pl_ids:
            p = os.path.join(chain_pdb_dir, f"pl{pid}_A.pdb")
            if not os.path.exists(p):
                continue
            s = pdb_sequence(p)
            if len(s) < min_len:
                continue
            out.write(f">pl{pid}|pl|A\n{s}\n")
            n_pl += 1
    return {"ppi_records": n_ppi, "pl_records": n_pl}


def mmseqs_cluster(fasta, workdir, min_seq_id=0.3, cov=0.5, threads=8):
    """mmseqs2 easy-cluster -> {record_id: cluster_representative}."""
    os.makedirs(workdir, exist_ok=True)
    pref = os.path.join(workdir, "clu")
    tmp = os.path.join(workdir, "tmp")
    cmd = [MMSEQS, "easy-cluster", fasta, pref, tmp, "--min-seq-id", str(min_seq_id),
           "-c", str(cov), "--cov-mode", "1", "--threads", str(threads), "-v", "1"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    out = {}
    with open(pref + "_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            out[mem] = rep
    return out


def complex_clusters(cluster_map):
    """record_id -> rep  ==>  cid -> set(reps)  (a complex owns every one of its chains' clusters)."""
    per = {}
    for rec, rep in cluster_map.items():
        cid = rec.split("|")[0]
        per.setdefault(cid, set()).add(rep)
    return per


# ---------------------------------------------------------------------------------------------
# ligand scaffolds
# ---------------------------------------------------------------------------------------------
def ligand_scaffold(pdbbind_dir, pid):
    """Bemis-Murcko scaffold SMILES for a PDBbind ligand ('' = acyclic / unreadable)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from masif_graph.p6.pl_graph import load_ligand

    mol = load_ligand(pdbbind_dir, pid)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:  # noqa: BLE001 - unparseable ligands just lose their scaffold key
        return None


# ---------------------------------------------------------------------------------------------
# split construction
# ---------------------------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def components(items, key_lists):
    """Connected components of `items` under "shares any key". key_lists: item -> iterable of keys."""
    uf = _UF()
    key_owner = {}
    for it in items:
        uf.find(("i", it))
        for k in key_lists(it):
            if k in key_owner:
                uf.union(("i", key_owner[k]), ("i", it))
            else:
                key_owner[k] = it
    comp = {}
    for it in items:
        comp.setdefault(uf.find(("i", it)), []).append(it)
    return list(comp.values())


def build(args):
    rep = {"args": vars(args)}
    eval_ids = [l.strip() for l in open(args.eval_ids) if l.strip()]
    ppi_ids = [l.strip() for l in open(args.ppi_ids) if l.strip()]
    pl_ids = [l.strip() for l in open(args.pl_ids) if l.strip()]

    fasta = os.path.join(args.workdir, "combined.fasta")
    os.makedirs(args.workdir, exist_ok=True)
    rep["fasta"] = write_combined_fasta(args.ppi_fasta, pl_ids, args.chain_pdb_dir, fasta)
    cl = mmseqs_cluster(fasta, os.path.join(args.workdir, "mmseqs"),
                        min_seq_id=args.min_seq_id, cov=args.cov, threads=args.threads)
    per = complex_clusters(cl)
    rep["n_clustered_complexes"] = len(per)
    rep["n_clusters"] = len({r for s in per.values() for r in s})

    # --- axis 1: everything must be disjoint from the frozen PPI eval set ---
    eval_present = [c for c in eval_ids if c in per]
    forbidden = {r for c in eval_present for r in per[c]}
    rep["eval_ids"] = len(eval_ids)
    rep["eval_with_sequence"] = len(eval_present)
    rep["forbidden_clusters"] = len(forbidden)

    def clean(cid):
        s = per.get(cid)
        return bool(s) and not (s & forbidden)

    train_ppi = [c for c in ppi_ids if c not in set(eval_ids) and clean(c)]
    pl_clean = [p for p in pl_ids if clean(f"pl{p}")]
    rep["ppi_in"] = len(ppi_ids)
    rep["ppi_train_after_eval_filter"] = len(train_ppi)
    rep["ppi_dropped_homologous_to_eval"] = len(ppi_ids) - len(train_ppi)
    rep["pl_in"] = len(pl_ids)
    rep["pl_after_eval_filter"] = len(pl_clean)
    rep["pl_dropped_homologous_to_eval"] = len(pl_ids) - len(pl_clean)

    # --- axis 2: carve the protein-ligand held-out by connected components ---
    scaf = {p: ligand_scaffold(args.pdbbind, p) for p in pl_clean}
    n_scaf = len({s for s in scaf.values() if s})
    rep["pl_distinct_scaffolds"] = n_scaf

    def keys_both(p):
        ks = [("seq", r) for r in per.get(f"pl{p}", ())]
        if scaf.get(p):
            ks.append(("scaf", scaf[p]))
        return ks

    def keys_seq(p):
        return [("seq", r) for r in per.get(f"pl{p}", ())]

    comp_both = components(pl_clean, keys_both)
    biggest = max((len(c) for c in comp_both), default=0)
    rep["pl_components_both"] = {"n": len(comp_both), "largest": biggest,
                                 "largest_frac": round(biggest / max(len(pl_clean), 1), 3)}
    if biggest <= args.max_component_frac * len(pl_clean):
        comps, rule = comp_both, "protein-cluster OR ligand-scaffold"
    else:
        comps = components(pl_clean, keys_seq)
        rule = "protein-cluster only (scaffold graph degenerate; scaffold overlap reported instead)"
        rep["pl_components_seq"] = {"n": len(comps),
                                    "largest": max((len(c) for c in comps), default=0)}
    rep["pl_split_rule"] = rule

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(comps))
    val_pl, target = [], args.n_val_pl
    for i in order:
        if len(val_pl) >= target:
            break
        if len(comps[i]) <= args.max_val_component:
            val_pl.extend(comps[i])
    val_set = set(val_pl)
    train_pl = [p for p in pl_clean if p not in val_set]

    # --- verification against the ACTUAL train ids (the Phase-5 lesson) ---
    train_all = set(train_ppi) | {f"pl{p}" for p in train_pl}
    train_clusters = {r for c in train_all for r in per.get(c, ())}
    leak_eval = [c for c in eval_ids if per.get(c, set()) & train_clusters]
    leak_val = [p for p in val_pl if per.get(f"pl{p}", set()) & train_clusters]
    train_scaf = {scaf[p] for p in train_pl if scaf.get(p)}
    val_scaf_seen = [p for p in val_pl if scaf.get(p) in train_scaf]
    rep["verify"] = {
        "eval_ppi_leaking_into_train": len(leak_eval),
        "val_pl_protein_leaking_into_train": len(leak_val),
        "val_pl_scaffold_seen_in_train": len(val_scaf_seen),
        "val_pl_scaffold_unseen": len(val_pl) - len(val_scaf_seen),
    }
    rep["sizes"] = {"train_ppi": len(train_ppi), "train_pl": len(train_pl),
                    "val_pl": len(val_pl), "eval_ppi": len(eval_ids)}

    os.makedirs(args.out, exist_ok=True)
    _w = lambda name, items: open(os.path.join(args.out, name), "w").write("\n".join(items) + "\n")
    _w("train_ppi.txt", train_ppi)
    _w("train_pl.txt", [f"pl{p}" for p in train_pl])
    _w("val_pl.txt", [f"pl{p}" for p in val_pl])
    _w("val_pl_scaffold_unseen.txt", [f"pl{p}" for p in val_pl if p not in set(val_scaf_seen)])
    _w("eval_ppi.txt", eval_ids)
    json.dump(rep, open(os.path.join(args.out, "split_report.json"), "w"), indent=1)
    json.dump({p: scaf.get(p) for p in pl_clean},
              open(os.path.join(args.out, "pl_scaffolds.json"), "w"), indent=0)
    print(json.dumps(rep, indent=1))
    return rep


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi-ids", required=True)
    ap.add_argument("--pl-ids", required=True)
    ap.add_argument("--eval-ids", default=f"{ROOT}/logs/phase5/eval_sc304_clean_vs_enc.txt")
    ap.add_argument("--ppi-fasta", default=f"{ROOT}/logs/phase5/all_chains.fasta")
    ap.add_argument("--chain-pdb-dir",
                    default=f"{ROOT}/masif-neosurf-af2/masif/data/masif_ppi_search/"
                            "data_preparation/01-benchmark_pdbs")
    ap.add_argument("--pdbbind", default=f"{ROOT}/data/pdbbind")
    ap.add_argument("--workdir", default=f"{ROOT}/logs/phase6C/split_work")
    ap.add_argument("--out", default=f"{ROOT}/logs/phase6C/split")
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-val-pl", type=int, default=300)
    ap.add_argument("--max-val-component", type=int, default=25)
    ap.add_argument("--max-component-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
