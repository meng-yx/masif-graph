"""Phase-8 A2 — do cryptic pockets close in apo models, bounding apo protein-ligand screening?

The funnel's protein-ligand branch assumes an apo/predicted protein still presents the pocket the
ligand binds. If AF3 models systematically close their pockets, no amount of model capacity recovers
the interaction from apo input, and the P-L branch has a ceiling that is a property of the *input*,
not of our architecture. That is worth knowing for ~CHF 1 before Stage C rather than after.

Runs on the 298 P-L complexes that already have AF3 apo models superposed into the holo frame
(Phase-7 S5) — no new structure generation. The crystal ligand pose is held fixed (standing project
rule: the protein varies, the ligand stays at its experimental pose).

Measured per complex, for BOTH states, so holo is the control rather than an assumed zero:
  * heavy-atom clashes between protein and ligand at 2.0 / 2.5 A, and the minimum distance;
  * ligand buried SASA and buried FRACTION (biotite `sasa`), and the apo/holo ratio;
  * clashes split **backbone vs sidechain** — a pocket that closes by sidechain rotamer is
    recoverable by a repack step, one that closes by backbone motion is not. This changes what
    Stage B should do, so it is worth separating.

Pre-registered (docs/24 §4, before any result was seen): "pocket collapsed" =
**buried-fraction ratio < 0.5 OR >= 10 heavy-atom clashes at 2.0 A**. The full distribution is
reported, not only the fraction beyond the threshold.

Usage:
  python -m masif_graph.p8.a2_pockets --pdbbind data/pdbbind --npz /work/upthomae/Meng/phase7/npz_pl \
      --out logs/phase8A/a2/pockets.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from masif_graph.io.reference import PDB_DIR
from masif_graph.p6.pl_graph import load_ligand

BACKBONE = {"N", "CA", "C", "O", "OXT"}
CLASH_TIGHT, CLASH_LOOSE = 2.0, 2.5
# Pre-registered collapse criterion.
COLLAPSE_RATIO, COLLAPSE_CLASHES = 0.5, 10


def _read_protein(path):
    """Heavy protein atoms from a masif-prepared PDB (drops H, waters and any HETATM)."""
    import biotite.structure.io.pdb as pdb

    arr = pdb.PDBFile.read(path).get_structure(model=1)
    keep = (arr.element != "H") & (~arr.hetero) & (arr.res_name != "HOH")
    return arr[keep]


def _ligand_array(coord, elements):
    import biotite.structure as struc

    a = struc.AtomArray(len(coord))
    a.coord = np.asarray(coord, np.float32)
    a.element = np.asarray(elements, dtype="U2")
    a.atom_name = np.asarray([f"{e}{i}" for i, e in enumerate(elements)], dtype="U6")
    a.res_name = np.full(len(coord), "LIG", dtype="U5")
    a.res_id = np.ones(len(coord), int)
    a.chain_id = np.full(len(coord), "L", dtype="U4")
    a.hetero = np.ones(len(coord), bool)
    return a


def _lig_sasa(prot, lig):
    """Ligand SASA alone and in complex, with 'Single' vdW radii (ProtOr is undefined for arbitrary
    ligand elements; the choice is identical across states so the apo/holo ratio is unaffected)."""
    import biotite.structure as struc

    alone = float(np.nansum(struc.sasa(lig, vdw_radii="Single", ignore_ions=False)))
    comp = struc.concatenate([prot, lig])
    f = np.zeros(comp.array_length(), bool)
    f[prot.array_length():] = True
    bound = float(np.nansum(struc.sasa(comp, atom_filter=f, vdw_radii="Single", ignore_ions=False)))
    return alone, bound


def probe_one(pid, pdbbind_dir):
    cid = f"pl{pid}"
    paths = {"holo": os.path.join(PDB_DIR, f"{cid}_A.pdb"),
             "af3": os.path.join(PDB_DIR, f"{cid}AF_A.pdb")}
    for st, p in paths.items():
        if not os.path.exists(p):
            return {"pid": pid, "ok": False, "err": f"missing {st} pdb"}
    mol = load_ligand(pdbbind_dir, pid)
    if mol is None:
        return {"pid": pid, "ok": False, "err": "ligand unreadable"}
    conf = mol.GetConformer()
    lcoord = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], float)
    lelem = [a.GetSymbol() for a in mol.GetAtoms()]
    keep = np.array([e != "H" for e in lelem])
    lcoord, lelem = lcoord[keep], [e for e, k in zip(lelem, keep) if k]
    if len(lcoord) < 3:
        return {"pid": pid, "ok": False, "err": "ligand < 3 heavy atoms"}
    lig = _ligand_array(lcoord, lelem)
    ltree = cKDTree(lcoord)

    rec = {"pid": pid, "ok": True, "n_lig_heavy": int(len(lcoord))}
    prots = {}
    for st, path in paths.items():
        prot = _read_protein(path)
        prots[st] = prot
        if prot.array_length() < 10:
            return {"pid": pid, "ok": False, "err": f"{st} protein too small"}
        d, _ = cKDTree(prot.coord).query(lcoord, k=1)
        pd_, _ = ltree.query(prot.coord, k=1)     # per-protein-atom nearest ligand distance
        is_bb = np.isin(prot.atom_name, list(BACKBONE))
        alone, bound = _lig_sasa(prot, lig)
        bf = (alone - bound) / alone if alone > 0 else float("nan")
        rec[st] = {
            "n_prot_heavy": int(prot.array_length()),
            "clash_2.0": int((pd_ < CLASH_TIGHT).sum()),
            "clash_2.5": int((pd_ < CLASH_LOOSE).sum()),
            "clash_2.0_backbone": int((pd_ < CLASH_TIGHT)[is_bb].sum()),
            "clash_2.0_sidechain": int((pd_ < CLASH_TIGHT)[~is_bb].sum()),
            "min_dist": float(pd_.min()),
            "lig_sasa_alone": alone, "lig_sasa_bound": bound,
            "buried_fraction": float(bf),
        }
    # Confound control: the AF3 model was superposed into the holo frame by CA correspondence
    # (p7.pl_af3). A poor superposition inflates clashes globally and would masquerade as a closed
    # pocket, so record how good the superposition actually is and check the correlation later.
    ca = {st: {int(r): c for r, c, n in
               zip(p.res_id, p.coord, p.atom_name) if n == "CA"} for st, p in prots.items()}
    shared = sorted(set(ca["holo"]) & set(ca["af3"]))
    if len(shared) >= 3:
        dh = np.array([ca["holo"][r] for r in shared], float)
        da = np.array([ca["af3"][r] for r in shared], float)
        rec["ca_rmsd_in_frame"] = float(np.sqrt(((dh - da) ** 2).sum(1).mean()))
        rec["n_ca_matched"] = len(shared)
    else:
        rec["ca_rmsd_in_frame"] = float("nan")
        rec["n_ca_matched"] = len(shared)

    h, a = rec["holo"]["buried_fraction"], rec["af3"]["buried_fraction"]
    rec["buried_fraction_ratio"] = float(a / h) if h > 0 else float("nan")
    rec["collapsed"] = bool(
        (np.isfinite(rec["buried_fraction_ratio"]) and rec["buried_fraction_ratio"] < COLLAPSE_RATIO)
        or rec["af3"]["clash_2.0"] >= COLLAPSE_CLASHES)
    return rec


def _q(a, name):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return {"name": name, "n": 0}
    return {"name": name, "n": int(a.size), "mean": float(a.mean()),
            "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbbind", default="data/pdbbind")
    ap.add_argument("--ids", default=None, help="optional pid list; default = every AF3 model on disk")
    ap.add_argument("--out", default="logs/phase8A/a2/pockets.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.ids:
        pids = [l.strip() for l in open(args.ids) if l.strip()]
    else:
        import glob
        pids = sorted(os.path.basename(p)[2:-len("AF_A.pdb")]
                      for p in glob.glob(os.path.join(PDB_DIR, "pl*AF_A.pdb")))
    if args.limit:
        pids = pids[: args.limit]
    print(f"{len(pids)} complexes with an AF3 apo model", flush=True)

    recs, fails = [], []
    for i, pid in enumerate(pids):
        try:
            r = probe_one(pid, args.pdbbind)
        except Exception as e:                                    # noqa: BLE001
            r = {"pid": pid, "ok": False, "err": f"{type(e).__name__}: {e}"}
        (recs if r.get("ok") else fails).append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pids)}  ok={len(recs)} fail={len(fails)}", flush=True)

    out = {"n_attempted": len(pids), "n_ok": len(recs), "n_failed": len(fails),
           "criterion": {"buried_fraction_ratio_below": COLLAPSE_RATIO,
                         "or_clashes_2.0_at_least": COLLAPSE_CLASHES,
                         "pre_registered": "docs/24-phase8A-plan.md sec.4"},
           "failures": fails[:40], "per_complex": recs}
    if recs:
        g = lambda st, k: [r[st][k] for r in recs]                # noqa: E731
        out["summary"] = {
            "buried_fraction_holo": _q(g("holo", "buried_fraction"), "buried_fraction_holo"),
            "buried_fraction_af3": _q(g("af3", "buried_fraction"), "buried_fraction_af3"),
            "buried_fraction_ratio": _q([r["buried_fraction_ratio"] for r in recs], "ratio"),
            "clash20_holo": _q(g("holo", "clash_2.0"), "clash20_holo"),
            "clash20_af3": _q(g("af3", "clash_2.0"), "clash20_af3"),
            "clash25_af3": _q(g("af3", "clash_2.5"), "clash25_af3"),
            "clash20_af3_backbone": _q(g("af3", "clash_2.0_backbone"), "clash20_af3_bb"),
            "clash20_af3_sidechain": _q(g("af3", "clash_2.0_sidechain"), "clash20_af3_sc"),
            "min_dist_holo": _q(g("holo", "min_dist"), "min_dist_holo"),
            "min_dist_af3": _q(g("af3", "min_dist"), "min_dist_af3"),
            "ca_rmsd_in_frame": _q([r["ca_rmsd_in_frame"] for r in recs], "ca_rmsd"),
            "n_collapsed": int(sum(r["collapsed"] for r in recs)),
            "frac_collapsed": float(np.mean([r["collapsed"] for r in recs])),
        }
        # Is "collapsed" really "badly superposed"? Correlate before concluding anything.
        rr = np.array([r["ca_rmsd_in_frame"] for r in recs], float)
        cc = np.array([r["af3"]["clash_2.0"] for r in recs], float)
        bb = np.array([r["buried_fraction_ratio"] for r in recs], float)
        ok = np.isfinite(rr) & np.isfinite(cc) & np.isfinite(bb)
        if ok.sum() > 10:
            from scipy.stats import spearmanr
            out["summary"]["confound_ca_rmsd_vs_clash20"] = {
                "rho": float(spearmanr(rr[ok], cc[ok]).statistic), "n": int(ok.sum())}
            out["summary"]["confound_ca_rmsd_vs_buried_ratio"] = {
                "rho": float(spearmanr(rr[ok], bb[ok]).statistic), "n": int(ok.sum())}
        s = out["summary"]
        print("=" * 78)
        print(f"A2 cryptic-pocket probe   n={len(recs)}  (failed {len(fails)})")
        print(f"  buried fraction   holo median {s['buried_fraction_holo']['median']:.3f}"
              f"   af3 median {s['buried_fraction_af3']['median']:.3f}")
        print(f"  ratio af3/holo    median {s['buried_fraction_ratio']['median']:.3f}"
              f"   p05 {s['buried_fraction_ratio']['p05']:.3f}"
              f"   p95 {s['buried_fraction_ratio']['p95']:.3f}")
        print(f"  clashes @2.0 A    holo median {s['clash20_holo']['median']:.0f}"
              f" (p95 {s['clash20_holo']['p95']:.0f})   af3 median {s['clash20_af3']['median']:.0f}"
              f" (p95 {s['clash20_af3']['p95']:.0f})")
        print(f"    af3 split       backbone median {s['clash20_af3_backbone']['median']:.0f}"
              f"   sidechain median {s['clash20_af3_sidechain']['median']:.0f}")
        print(f"  >>> COLLAPSED (pre-registered): {s['n_collapsed']}/{len(recs)}"
              f" = {100*s['frac_collapsed']:.1f}%")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
