"""Phase-8 A4 — mine biological vs crystal-contact interfaces from REMARK 350.

The funnel's premise (docs/23 D8-1) is that `biological_contact` is separable from
`complementary_contact`: a crystal contact is real, buried and complementary but not biological, and
killing that false-positive mode is the point of Stage 3. Before building Stage 3 we should know
whether ANY signal for that distinction exists — including in the trivial feature, interface area.

Labelling (the subtle part). A REMARK 350 BIOMOLECULE lists chains and the symmetry operations
applied to them. Two ASU chains A and B form a **biological** pair only if some assembly contains
BOTH under the **identity** operator. If the assembly is "A plus a symmetry copy of A", then the
A-B contact seen in the asymmetric unit is a *crystal* contact even though A appears in the
assembly. Co-occurrence in the chain list alone is therefore not enough, and using it would
mislabel exactly the cases the classifier must get right.

  bio=1   both chains receive the identity transform within one BIOMOLECULE
  bio=0   the chains touch in the ASU but never co-occur under identity  (crystal contact)

Emitted per interface: chain pair, contact count, and BSA (biotite `sasa`), which is the
BSA-only baseline Stage 3 must beat.

Usage:
  python -m masif_graph.p8.a4_mine --n-entries 300 --out logs/phase8A/a4/interfaces.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)

IDENTITY = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
CONTACT_CUT = 5.0
MIN_CONTACTS = 30


def parse_remark350(path):
    """{biomolecule_id: set(chains receiving the IDENTITY transform)}.

    REMARK 350 layout: 'APPLY THE FOLLOWING TO CHAINS: A, B' then one or more BIOMT triplets. A
    chain group may be continued on 'AND CHAINS:' lines. Only the identity operator matters here.
    """
    bios, cur, chains, rows = {}, None, [], []

    def flush():
        if cur is None or not chains:
            return
        for i in range(0, len(rows) - 2, 3):
            m = np.array(rows[i:i + 3], float)
            if m.shape == (3, 4) and np.allclose(m, IDENTITY, atol=1e-3):
                bios.setdefault(cur, set()).update(chains)

    with open(path, errors="ignore") as fh:
        for ln in fh:
            if not ln.startswith("REMARK 350"):
                continue
            body = ln[11:].strip()
            if body.startswith("BIOMOLECULE:"):
                flush()
                cur = body.split(":")[1].strip()
                chains, rows = [], []
            elif "APPLY THE FOLLOWING TO CHAINS:" in body:
                flush()
                rows = []
                chains = [c.strip() for c in body.split(":")[1].split(",") if c.strip()]
            elif body.startswith("AND CHAINS:"):
                chains += [c.strip() for c in body.split(":")[1].split(",") if c.strip()]
            elif body.startswith("BIOMT"):
                p = body.split()
                if len(p) >= 6:
                    rows.append([float(x) for x in p[2:6]])
    flush()
    return bios


def _chains_heavy(path):
    import biotite.structure.io.pdb as pdb
    arr = pdb.PDBFile.read(path).get_structure(model=1)
    keep = (arr.element != "H") & (~arr.hetero) & (arr.res_name != "HOH")
    arr = arr[keep]
    return {c: arr[arr.chain_id == c] for c in np.unique(arr.chain_id)}


def _bsa(a, b):
    """Buried surface area = SASA(A) + SASA(B) - SASA(AB), in A^2 (both sides counted)."""
    import biotite.structure as struc
    sa = float(np.nansum(struc.sasa(a, vdw_radii="Single", ignore_ions=False)))
    sb = float(np.nansum(struc.sasa(b, vdw_radii="Single", ignore_ions=False)))
    ab = struc.concatenate([a, b])
    sab = float(np.nansum(struc.sasa(ab, vdw_radii="Single", ignore_ions=False)))
    return sa + sb - sab


def mine_entry(pdb_path, max_chains=16):
    pid = os.path.basename(pdb_path)[:-4]
    bios = parse_remark350(pdb_path)
    if not bios:
        return {"pdb": pid, "ok": False, "err": "no REMARK 350 identity operator"}
    ch = _chains_heavy(pdb_path)
    ch = {k: v for k, v in ch.items() if v.array_length() >= 50}
    if len(ch) < 2:
        return {"pdb": pid, "ok": False, "err": f"only {len(ch)} usable chains"}
    if len(ch) > max_chains:
        return {"pdb": pid, "ok": False, "err": f"{len(ch)} chains > max {max_chains}"}

    names = sorted(ch)
    trees = {c: cKDTree(ch[c].coord) for c in names}
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs = trees[a].query_ball_tree(trees[b], CONTACT_CUT)
            n_ct = int(sum(len(x) for x in pairs))
            if n_ct < MIN_CONTACTS:
                continue
            bio = any(a in s and b in s for s in bios.values())
            out.append({"pdb": pid, "c1": a, "c2": b, "n_contacts": n_ct,
                        "bio": int(bio), "bsa": round(_bsa(ch[a], ch[b]), 1),
                        "n_atom1": int(ch[a].array_length()), "n_atom2": int(ch[b].array_length())})
    return {"pdb": pid, "ok": True, "n_chains": len(ch), "n_biomolecules": len(bios),
            "interfaces": out}


def fetch_pdb(pdb_id, cache_dir):
    """Download one entry from RCSB into `cache_dir` (PDB format, so REMARK 350 is present)."""
    import urllib.request
    dest = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(dest) and os.path.getsize(dest) > 2000:
        return dest
    tmp = f"{dest}.part{os.getpid()}"
    try:
        req = urllib.request.Request(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                                     headers={"User-Agent": "masif-graph/0.1"})
        with urllib.request.urlopen(req, timeout=60) as f, open(tmp, "wb") as g:
            g.write(f.read())
        os.replace(tmp, dest)
        return dest
    except Exception:                                                   # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def main():
    ap = argparse.ArgumentParser()
    # The reference 00-raw_pdbs cache holds the EVAL set (304 of its 305 plain entries come from
    # data/lists/testing.txt), so mining it would probe on evaluation complexes. Fetch training
    # entries fresh instead, into a Phase-8A cache that does not touch the reference tree.
    ap.add_argument("--cache-dir", default="/work/upthomae/Meng/phase8A/a4_pdbs")
    ap.add_argument("--lists", nargs="+", default=["data/lists/training.txt"])
    ap.add_argument("--n-entries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/phase8A/a4/interfaces.json")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    want = set()
    for p in args.lists:
        for ln in open(p):
            f = ln.strip().split("_")
            if len(f) >= 3:
                want.add(f[0].upper())
    # Prefer entries whose stored pair uses single-letter chains and that have >=3 chains in the
    # ASU — an entry with exactly the 2 stored chains can only ever yield the bio=1 pair.
    pool = sorted(want)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    cand, tried = [], 0
    for j in order:
        if len(cand) >= args.n_entries:
            break
        tried += 1
        p = fetch_pdb(pool[int(j)], args.cache_dir)
        if p:
            cand.append(p)
        if tried % 50 == 0:
            print(f"  fetched {len(cand)}/{tried} attempted", flush=True)
    print(f"{len(cand)} entries to mine (fetched from RCSB; {len(want)} corpus entries)", flush=True)
    _ = glob

    recs, fails = [], []
    for i, p in enumerate(cand):
        try:
            r = mine_entry(p)
        except Exception as e:                                          # noqa: BLE001
            r = {"pdb": os.path.basename(p)[:-4], "ok": False,
                 "err": f"{type(e).__name__}: {e}"}
        (recs if r.get("ok") else fails).append(r)
        if (i + 1) % 25 == 0:
            n_if = sum(len(x["interfaces"]) for x in recs)
            print(f"  {i+1}/{len(cand)}  ok={len(recs)} fail={len(fails)} interfaces={n_if}",
                  flush=True)

    iface = [x for r in recs for x in r["interfaces"]]
    n1 = sum(x["bio"] for x in iface)
    out = {"n_entries_attempted": len(cand), "n_entries_ok": len(recs), "n_failed": len(fails),
           "n_interfaces": len(iface), "n_bio": n1, "n_crystal": len(iface) - n1,
           "fail_reasons": {}, "interfaces": iface}
    for f in fails:
        k = f["err"].split(":")[0][:40]
        out["fail_reasons"][k] = out["fail_reasons"].get(k, 0) + 1

    if iface:
        b = np.array([x["bsa"] for x in iface], float)
        y = np.array([x["bio"] for x in iface])
        out["bsa_summary"] = {
            "bio_median": float(np.median(b[y == 1])) if (y == 1).any() else None,
            "crystal_median": float(np.median(b[y == 0])) if (y == 0).any() else None,
            "bio_p25_p75": [float(np.percentile(b[y == 1], q)) for q in (25, 75)] if (y == 1).any() else None,
            "crystal_p25_p75": [float(np.percentile(b[y == 0], q)) for q in (25, 75)] if (y == 0).any() else None,
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("=" * 78)
    print(f"A4 mining: {len(recs)}/{len(cand)} entries, {len(iface)} interfaces "
          f"({n1} bio / {len(iface)-n1} crystal)")
    if out.get("bsa_summary"):
        s = out["bsa_summary"]
        print(f"  BSA median  bio {s['bio_median']} A^2   crystal {s['crystal_median']} A^2")
    print(f"  failures: {out['fail_reasons']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
