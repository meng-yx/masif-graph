"""Phase-8 A0 — apo-prediction method benchmark: test-set selection, shared-MSA export, scoring.

Decides D8-12 (which method generates the apo side of the training corpus, and at what holo:apo
ratio) — or rather, produces the evidence; the choice is the user's at the Stage-A PAUSE.

Three subcommands:

  select   Build the 30-chain test set from chains that already have holo + AF3 model + MSA on
           disk, stratified by length AND by AF3 confidence. The confidence stratum matters:
           Phase-3 M2 measured inter-sample CA-RMSD of ~0.1 A for confident chains and up to ~15 A
           for uncertain ones, so an all-confident test set would score every method identically on
           the spread metric, which is the metric we are actually buying.

  msa      Export the ONE shared alignment every MSA-based method runs on. Our AF3 `_data.json`
           carries `unpairedMsa` / `pairedMsa` inline as a3m, so AF3, Protenix, Chai and Boltz can
           be compared on identical input — otherwise the benchmark would mostly be measuring five
           different MSA searches, which is the dominant cost and not the thing under test.

  score    Metrics per predicted model set: TM-score (TMalign, holo-normalised), CA-RMSD, and the
           calibrated-spread family (inter-sample RMSF, Spearman vs holo B-factor and vs pLDDT).

Usage:
  python -m masif_graph.p8.a0_bench select --n 30 --out logs/phase8A/a0/testset.json
  python -m masif_graph.p8.a0_bench msa --testset logs/phase8A/a0/testset.json --out-dir <dir>
  python -m masif_graph.p8.a0_bench score --testset ... --pred-dir ... --method chai --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

AF3_POOLS = ["/work/upthomae/Meng/phase5_af3", "/work/upthomae/Meng/phase7_af3"]
TMALIGN = "/work/upthomae/Meng/conda_envs/masif-graph/bin/TMalign"


# --------------------------------------------------------------------------- structure helpers
def read_cif_ca(path):
    """(res_id, coord, b_factor) for CA atoms of model 1. AF3 writes pLDDT into b_factor."""
    import biotite.structure.io.pdbx as pdbx
    cf = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(cf, model=1, extra_fields=["b_factor"])
    ca = arr[arr.atom_name == "CA"]
    return np.asarray(ca.res_id), np.asarray(ca.coord, float), np.asarray(ca.b_factor, float)


def read_pdb_ca(path):
    import biotite.structure.io.pdb as pdb
    arr = pdb.PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor"])
    ca = arr[(arr.atom_name == "CA") & (~arr.hetero)]
    return np.asarray(ca.res_id), np.asarray(ca.coord, float), np.asarray(ca.b_factor, float)


def _read_any_ca(path):
    return read_cif_ca(path) if path.endswith((".cif", ".mmcif")) else read_pdb_ca(path)


def tmscore(pred_path, ref_path):
    """TM-score of `pred` against `ref`, NORMALISED BY THE REFERENCE (holo) length.

    Chain 2 is the reference, so TMalign's 'normalized by length of Chain_2' line is the one to
    read: normalising by the prediction would reward a method for predicting fewer residues.
    """
    try:
        out = subprocess.run([TMALIGN, pred_path, ref_path], capture_output=True, text=True,
                             timeout=300).stdout
    except Exception as e:                                              # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    res = {}
    for ln in out.splitlines():
        if ln.startswith("TM-score=") and "Chain_2" in ln:
            res["tm_ref_normalised"] = float(ln.split("=")[1].split("(")[0])
        elif ln.startswith("TM-score=") and "Chain_1" in ln:
            res["tm_pred_normalised"] = float(ln.split("=")[1].split("(")[0])
        elif ln.startswith("Aligned length="):
            parts = ln.replace(",", " ").split()
            res["aligned_length"] = int(parts[2])
            res["tm_rmsd"] = float(parts[4])
    return res or {"error": "TMalign produced no parsable output"}


def ca_rmsd_by_resid(a_path, b_path):
    """CA-RMSD over residues present in both, matched by residue id, after optimal superposition."""
    from masif_graph.align.global_align import apply_T, kabsch
    ra, ca_, _ = _read_any_ca(a_path)
    rb, cb, _ = _read_any_ca(b_path)
    ma = {int(r): c for r, c in zip(ra, ca_)}
    mb = {int(r): c for r, c in zip(rb, cb)}
    shared = sorted(set(ma) & set(mb))
    if len(shared) < 3:
        return float("nan"), 0
    A = np.array([ma[r] for r in shared]); B = np.array([mb[r] for r in shared])
    T = kabsch(A, B)
    return float(np.sqrt(((apply_T(T, A) - B) ** 2).sum(1).mean())), len(shared)


# --------------------------------------------------------------------------- select
def _pool_chains():
    """Chains with an AF3 model AND an AF3 MSA json AND a holo PDB on disk."""
    from masif_graph.io.reference import PDB_DIR
    out = {}
    for pool in AF3_POOLS:
        for m in sorted(glob.glob(os.path.join(pool, "models", "*", "*_model.cif"))):
            cid = os.path.basename(os.path.dirname(m))
            msa = os.path.join(pool, "msa", cid, cid, f"{cid}_data.json")
            pdb_id, _, ch = cid.rpartition("_")
            holo = os.path.join(PDB_DIR, f"{pdb_id}_{ch}.pdb")
            if os.path.exists(msa) and os.path.exists(holo):
                out[cid] = {"cid": cid, "pool": pool, "model": m, "msa_json": msa, "holo": holo}
    return out


def cmd_select(args):
    pool = _pool_chains()
    print(f"{len(pool)} candidate chains with model + MSA + holo on disk", flush=True)
    rows = []
    for i, (cid, r) in enumerate(sorted(pool.items())):
        try:
            _, _, b = read_cif_ca(r["model"])
        except Exception as e:                                          # noqa: BLE001
            print(f"  {cid}: unreadable model ({type(e).__name__}) — skip", flush=True)
            continue
        rows.append({**r, "length": int(len(b)), "plddt_mean": float(b.mean()),
                     "plddt_p10": float(np.percentile(b, 10))})
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(pool)}", flush=True)
    if not rows:
        raise SystemExit("no candidates")

    # Confidence stratification by QUARTILE EXTREMES, not a median split. Measured on this pool:
    # pLDDT p01 = 77.5, p50 = 95.3, and only 1 of 883 chains is below 70 — our corpus is almost
    # entirely confident, so a median split would compare "confident" against "slightly less
    # confident" and the calibrated-spread metric would have almost no dynamic range. Taking the
    # top and bottom quartiles and dropping the middle buys what contrast actually exists. That
    # the contrast is small IS a finding, and it is reported rather than hidden.
    p = np.array([r["plddt_mean"] for r in rows])
    lo, hi = float(np.percentile(p, 25)), float(np.percentile(p, 75))
    strata, per = {}, max(1, args.n // 6)
    for r in rows:
        ln = "short" if r["length"] < 150 else ("medium" if r["length"] <= 350 else "long")
        if r["plddt_mean"] <= lo:
            cf = "lowconf"
        elif r["plddt_mean"] >= hi:
            cf = "highconf"
        else:
            continue                     # middle half dropped on purpose
        strata.setdefault(f"{ln}_{cf}", []).append(r)

    rng = np.random.default_rng(args.seed)
    picked = []
    for k in sorted(strata):
        v = sorted(strata[k], key=lambda x: x["cid"])
        take = min(per, len(v))
        idx = rng.choice(len(v), take, replace=False) if len(v) > take else np.arange(len(v))
        picked += [v[int(j)] for j in sorted(idx)]
    # top up deterministically if a stratum was short
    if len(picked) < args.n:
        rest = [r for r in sorted(rows, key=lambda x: x["cid"]) if r not in picked]
        picked += rest[: args.n - len(picked)]
    picked = picked[: args.n]

    out = {"n": len(picked), "plddt_q25": lo, "plddt_q75": hi, "seed": args.seed,
           "pool_plddt_percentiles": {str(q): float(np.percentile(p, q))
                                      for q in (1, 5, 10, 25, 50, 75, 90)},
           "strata_sizes": {k: len(v) for k, v in sorted(strata.items())},
           "picked_strata": {}, "chains": picked}
    for r in picked:
        ln = "short" if r["length"] < 150 else ("medium" if r["length"] <= 350 else "long")
        cf = "highconf" if r["plddt_mean"] >= hi else "lowconf"
        out["picked_strata"][f"{ln}_{cf}"] = out["picked_strata"].get(f"{ln}_{cf}", 0) + 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    with open(args.out.replace(".json", ".txt"), "w") as f:
        for r in picked:
            f.write(r["cid"] + "\n")
    print(f"picked {len(picked)} chains; strata {out['picked_strata']}")
    print(f"  length  median {np.median([r['length'] for r in picked]):.0f} "
          f"[{min(r['length'] for r in picked)}-{max(r['length'] for r in picked)}]")
    print(f"  pLDDT   median {np.median([r['plddt_mean'] for r in picked]):.1f} "
          f"[{min(r['plddt_mean'] for r in picked):.1f}-{max(r['plddt_mean'] for r in picked):.1f}]")
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------- shared MSA export
def cmd_msa(args):
    ts = json.load(open(args.testset))
    os.makedirs(args.out_dir, exist_ok=True)
    a3m_dir = os.path.join(args.out_dir, "a3m")
    os.makedirs(a3m_dir, exist_ok=True)
    rep = []
    for r in ts["chains"]:
        cid = r["cid"]
        d = json.load(open(r["msa_json"]))
        seqs = d.get("sequences") or []
        prot = None
        for s in seqs:
            if "protein" in s:
                prot = s["protein"]
                break
        if prot is None:
            rep.append({"cid": cid, "ok": False, "err": "no protein entity in AF3 json"})
            continue
        e = {"cid": cid, "ok": True, "query_len": len(prot.get("sequence", "")),
             "n_templates": len(prot.get("templates") or [])}
        for field, suffix in (("unpairedMsa", "unpaired"), ("pairedMsa", "paired")):
            a3m = prot.get(field) or ""
            p = os.path.join(a3m_dir, f"{cid}.{suffix}.a3m")
            if a3m:
                with open(p, "w") as fh:
                    fh.write(a3m)
                e[f"n_{suffix}"] = a3m.count(">")
                e[f"{suffix}_path"] = p
            else:
                e[f"n_{suffix}"] = 0
        rep.append(e)
        print(f"  {cid}: unpaired {e.get('n_unpaired',0)} paired {e.get('n_paired',0)} "
              f"templates {e['n_templates']}", flush=True)

    # chai wants `<sha256 of sequence>.aligned.pqt`; convert with chai's own parser so the format
    # is exactly what its loader expects rather than our guess at it.
    if args.chai_python and os.path.exists(args.chai_python):
        pqt_dir = os.path.join(args.out_dir, "chai_msas")
        os.makedirs(pqt_dir, exist_ok=True)
        script = os.path.join(args.out_dir, "_to_pqt.py")
        with open(script, "w") as fh:
            fh.write(
                "import sys, json, glob, os\n"
                "from pathlib import Path\n"
                "from chai_lab.data.parsing.msas.aligned_pqt import a3m_to_aligned_dataframe\n"
                "from chai_lab.data.parsing.msas.data_source import MSADataSource\n"
                "a3m_dir, out_dir = sys.argv[1], sys.argv[2]\n"
                "ok, bad = 0, []\n"
                "for p in sorted(glob.glob(os.path.join(a3m_dir, '*.a3m'))):\n"
                "    try:\n"
                "        src = MSADataSource.UNIPROT if 'paired' in os.path.basename(p) \\\n"
                "            else MSADataSource.UNIREF90\n"
                "        df = a3m_to_aligned_dataframe(Path(p), src)\n"
                "        df.to_parquet(os.path.join(out_dir, os.path.basename(p) + '.aligned.pqt'))\n"
                "        ok += 1\n"
                "    except Exception as e:\n"
                "        bad.append(f'{os.path.basename(p)}: {type(e).__name__}: {e}')\n"
                "print(json.dumps({'converted': ok, 'failed': bad[:10], 'n_failed': len(bad)}))\n")

        pr = subprocess.run([args.chai_python, script, a3m_dir, pqt_dir],
                            capture_output=True, text=True, timeout=3600)
        tail = [ln for ln in pr.stdout.splitlines() if ln.startswith("{")]
        conv = json.loads(tail[-1]) if tail else {"error": pr.stderr[-800:]}
        print("chai pqt conversion:", conv)
    else:
        conv = {"skipped": "chai python not found"}

    out = {"n": len(rep), "n_ok": sum(r["ok"] for r in rep), "a3m_dir": a3m_dir,
           "chai_conversion": conv, "per_chain": rep}
    p = os.path.join(args.out_dir, "msa_export.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="logs/phase8A/a0/testset.json")
    s.set_defaults(func=cmd_select)

    m = sub.add_parser("msa")
    m.add_argument("--testset", default="logs/phase8A/a0/testset.json")
    m.add_argument("--out-dir", default="/work/upthomae/Meng/phase8A/a0_msa")
    m.add_argument("--chai-python", default="/home/ymeng/miniconda3/envs/chai/bin/python")
    m.set_defaults(func=cmd_msa)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
