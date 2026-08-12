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
        # chai resolves an MSA as `msa_directory / expected_basename(query_sequence)` — a hash of
        # the sequence, ONE file per sequence. Writing `{cid}.unpaired.a3m.aligned.pqt` would not
        # be found, and chai only logs a warning before falling back to a single-sequence MSA: the
        # "chai + shared MSA" arm would silently have become "chai without MSA". So use chai's own
        # `expected_basename`, merge unpaired+paired into one frame, and verify the file lands.
        seqs = {r["cid"]: r for r in ts["chains"]}
        qmap = {}
        for r in ts["chains"]:
            d = json.load(open(r["msa_json"]))
            for s in d.get("sequences") or []:
                if "protein" in s:
                    qmap[r["cid"]] = s["protein"]["sequence"]
                    break
        json.dump(qmap, open(os.path.join(args.out_dir, "query_seqs.json"), "w"))
        script = os.path.join(args.out_dir, "_to_pqt.py")
        with open(script, "w") as fh:
            fh.write(
                "import sys, json, os\n"
                "from pathlib import Path\n"
                "from chai_lab.data.parsing.msas.aligned_pqt import (\n"
                "    merge_multi_a3m_to_aligned_dataframe, expected_basename)\n"
                "from chai_lab.data.parsing.msas.data_source import MSADataSource\n"
                "a3m_dir, out_dir, qpath = sys.argv[1], sys.argv[2], sys.argv[3]\n"
                "qmap = json.load(open(qpath))\n"
                "ok, bad = [], []\n"
                "for cid, seq in sorted(qmap.items()):\n"
                "    try:\n"
                "        src = {}\n"
                "        up = Path(a3m_dir) / f'{cid}.unpaired.a3m'\n"
                "        pa = Path(a3m_dir) / f'{cid}.paired.a3m'\n"
                "        if up.exists(): src[up] = MSADataSource.UNIREF90\n"
                "        if pa.exists(): src[pa] = MSADataSource.UNIPROT\n"
                "        if not src: bad.append(f'{cid}: no a3m'); continue\n"
                "        df = merge_multi_a3m_to_aligned_dataframe(src)\n"
                "        dest = Path(out_dir) / expected_basename(seq)\n"
                "        df.to_parquet(dest)\n"
                "        ok.append({'cid': cid, 'file': dest.name, 'depth': int(len(df)),\n"
                "                   'exists': dest.is_file()})\n"
                "    except Exception as e:\n"
                "        bad.append(f'{cid}: {type(e).__name__}: {e}')\n"
                "print(json.dumps({'converted': len(ok), 'rows': ok,\n"
                "                  'failed': bad[:10], 'n_failed': len(bad)}))\n")

        pr = subprocess.run([args.chai_python, script, a3m_dir, pqt_dir,
                             os.path.join(args.out_dir, "query_seqs.json")],
                            capture_output=True, text=True, timeout=7200)
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



# --------------------------------------------------------------------------- score
MODEL_GLOBS = {
    "af3":   ["{cid}/seed-*_sample-*/*_model.cif"],
    "chai":  ["{cid}/pred.model_idx_*.cif"],
}


def _find_models(pred_dir, cid, method):
    pats = MODEL_GLOBS.get(method, ["{cid}/*.cif"])
    out = []
    for pat in pats:
        out += sorted(glob.glob(os.path.join(pred_dir, pat.format(cid=cid))))
    return out


def _rmsf(models):
    """Per-residue RMSF across an ensemble, after superposing every sample onto the first.

    Returns (res_ids, rmsf, mean_pairwise_rmsd, max_pairwise_rmsd). An ensemble whose members are
    identical scores 0 here no matter how accurate it is — which is the point: the spread is what
    a conformational-landscape sampler is supposed to provide.
    """
    from masif_graph.align.global_align import apply_T, kabsch
    per = []
    for m in models:
        r, c, _ = _read_any_ca(m)
        per.append({int(x): y for x, y in zip(r, c)})
    shared = sorted(set.intersection(*[set(d) for d in per])) if per else []
    if len(shared) < 3 or len(per) < 2:
        return np.array(shared), np.array([]), float("nan"), float("nan")
    X = np.stack([np.array([d[r] for r in shared]) for d in per])       # (S, R, 3)
    ref = X[0]
    X = np.stack([X[0]] + [apply_T(kabsch(x, ref), x) for x in X[1:]])
    mu = X.mean(0)
    rmsf = np.sqrt(((X - mu) ** 2).sum(2).mean(0))
    pw = [float(np.sqrt(((X[i] - X[j]) ** 2).sum(1).mean()))
          for i in range(len(X)) for j in range(i + 1, len(X))]
    return np.array(shared), rmsf, float(np.mean(pw)), float(np.max(pw))


def cmd_score(args):
    from scipy.stats import spearmanr
    ts = json.load(open(args.testset))
    rows = []
    for r in ts["chains"]:
        cid = r["cid"]
        models = _find_models(args.pred_dir, cid, args.method)
        e = {"cid": cid, "length": r["length"], "n_models": len(models)}
        if not models:
            e["error"] = "no models found"
            rows.append(e)
            continue
        tm = tmscore(models[0], r["holo"])
        e.update({("tm_score" if k == "tm_ref_normalised" else k): v for k, v in tm.items()})
        # TMalign's own RMSD over structurally aligned residues is the RMSD to trust. The
        # residue-id-matched value is kept only as a diagnostic: a raw AF3/chai model is numbered
        # 1..N while the holo PDB uses author numbering, so id-matching can pair up unrelated
        # residues and produce a large-but-meaningless RMSD.
        e["ca_rmsd_resid_matched_UNRELIABLE"], e["n_ca_matched"] = ca_rmsd_by_resid(
            models[0], r["holo"])
        res, rmsf, pw_mean, pw_max = _rmsf(models)
        e["pairwise_rmsd_mean"], e["pairwise_rmsd_max"] = pw_mean, pw_max
        e["rmsf_mean"] = float(rmsf.mean()) if rmsf.size else float("nan")
        # calibration: does the ensemble spread track REAL flexibility?
        if rmsf.size:
            hr, _, hb = _read_any_ca(r["holo"])
            hmap = {int(x): y for x, y in zip(hr, hb)}
            m = np.array([hmap.get(int(x), np.nan) for x in res], float)
            ok = np.isfinite(m) & np.isfinite(rmsf)
            if ok.sum() > 10 and np.std(m[ok]) > 0 and np.std(rmsf[ok]) > 0:
                e["spearman_rmsf_vs_bfactor"] = float(spearmanr(rmsf[ok], m[ok]).statistic)
            pr, _, pb = _read_any_ca(models[0])
            pmap = {int(x): y for x, y in zip(pr, pb)}
            q = np.array([pmap.get(int(x), np.nan) for x in res], float)
            ok = np.isfinite(q) & np.isfinite(rmsf)
            if ok.sum() > 10 and np.std(q[ok]) > 0 and np.std(rmsf[ok]) > 0:
                e["spearman_rmsf_vs_plddt"] = float(spearmanr(rmsf[ok], q[ok]).statistic)
        # cost, where the runner recorded it
        rj = os.path.join(args.pred_dir, cid, "run.json")
        if os.path.exists(rj):
            e["seconds"] = json.load(open(rj)).get("seconds")
            e["msa_found"] = json.load(open(rj)).get("msa_found")
        rows.append(e)
        print(f"  {cid}: n={len(models)} TM={e.get('tm_score', float('nan')):.3f} "
              f"alnRMSD={e.get('tm_rmsd', float('nan')):.2f} "
              f"spread={pw_mean:.2f}", flush=True)

    def agg(key):
        v = [x[key] for x in rows if isinstance(x.get(key), (int, float))
             and np.isfinite(x.get(key))]
        if not v:
            return None
        return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}

    out = {"method": args.method, "pred_dir": args.pred_dir, "n_chains": len(rows),
           "n_with_models": sum(1 for x in rows if x["n_models"] > 0),
           "summary": {k: agg(k) for k in
                       ("tm_score", "tm_rmsd", "pairwise_rmsd_mean", "pairwise_rmsd_max",
                        "rmsf_mean", "spearman_rmsf_vs_bfactor", "spearman_rmsf_vs_plddt",
                        "seconds", "n_models")},
           "per_chain": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    s = out["summary"]
    print("=" * 78)
    print(f"A0 {args.method}: {out['n_with_models']}/{len(rows)} chains with models")
    for k in ("tm_score", "tm_rmsd", "pairwise_rmsd_mean", "spearman_rmsf_vs_bfactor",
              "spearman_rmsf_vs_plddt", "seconds"):
        v = s.get(k)
        if v:
            print(f"  {k:26s} median {v['median']:8.3f}  [p25 {v['p25']:.3f}, p75 {v['p75']:.3f}]  n={v['n']}")
    print(f"wrote {args.out}")


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

    c = sub.add_parser("score")
    c.add_argument("--testset", default="logs/phase8A/a0/testset.json")
    c.add_argument("--pred-dir", required=True)
    c.add_argument("--method", required=True, choices=["af3", "chai", "other"])
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
