#!/usr/bin/env python
"""Assemble the Phase-7 result JSONs, aggregating across SEEDS.

The point of Phase 7's D7-6 is that no number is claimed from one seed, so this reports
`mean ± half-range` over the seeds of each condition and never a bare single value where more than
one seed exists. A difference smaller than the seed spread is reported as *not resolved*, not as an
effect.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

RES = "logs/phase7/results"
COND_ORDER = ["p7randinit", "ppionly", "plonly", "p6comb", "p7comb"]
LABEL = {"p7randinit": "random-init (chance)",
         "ppionly": "PPI-only (do-no-harm control)",
         "plonly": "ligand-only (transfer control)",
         "p6comb": "Phase-6C combined (ligand = atoms only)",
         "p7comb": "**Phase-7 combined (ligand = atoms + SURFACE)**"}


def cond_of(tag):
    return re.sub(r"_s\d+$", "", tag)


def scan(prefix):
    """{condition: [(tag, json)]} for files named <prefix><tag>.json."""
    out = {}
    if not os.path.isdir(RES):
        return out
    for f in sorted(os.listdir(RES)):
        if not (f.startswith(prefix) and f.endswith(".json")):
            continue
        tag = f[len(prefix):-len(".json")]
        out.setdefault(cond_of(tag), []).append((tag, json.load(open(os.path.join(RES, f)))))
    return out


def agg(vals, fmt="{:.3f}"):
    """mean ± half-range; a single seed is shown with an explicit (1 seed) marker."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return (fmt + " (1 seed)").format(v[0])
    return (fmt + " ± " + fmt).format(float(np.mean(v)), (max(v) - min(v)) / 2.0)


def order(conds):
    return [c for c in COND_ORDER if c in conds] + [c for c in conds if c not in COND_ORDER]


def table(title, header, rows_fn, data):
    print(f"### {title}\n")
    print("| model | seeds | " + " | ".join(header) + " |")
    print("|---" * (len(header) + 2) + "|")
    for c in order(data):
        entries = data[c]
        cells = rows_fn(entries)
        if cells is None:
            continue
        print(f"| {LABEL.get(c, c)} | {len(entries)} | " + " | ".join(cells) + " |")
    print()


def axis1():
    # gate_<tag>_pos.json and gate_<tag>_pos_sc.json share a tag; keep only the dense `pos` variant
    raw = {}
    for f in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
        if f.startswith("gate_") and f.endswith("_pos.json"):
            tag = f[len("gate_"):-len("_pos.json")]
            raw.setdefault(cond_of(tag), []).append((tag, json.load(open(os.path.join(RES, f)))))
    data = raw
    if not data:
        return

    def rows(entries):
        g = lambda d, cell, f: d["results"].get(cell, {}).get(f)
        return [agg([g(d, "HH_learned", "top5") for _t, d in entries]),
                agg([g(d, "HH_learned", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "AA_learned", "top5") for _t, d in entries]),
                agg([g(d, "AA_learned", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "HH_learned", "top5") - g(d, "AA_learned", "top5") for _t, d in entries],
                    "{:+.3f}")]

    table("Axis 1 — do-no-harm PPI gate (287-clean, dense `pos`)",
          ["HH top5", "HH medR", "AA top5", "AA medR", "holo→AA drop"], rows, data)
    any_d = next(iter(data.values()))[0][1]
    r = any_d["results"]
    print(f"Frozen MaSIF on the same patches: HH top5 {r['HH_frozen']['top5']:.3f} "
          f"(medR {r['HH_frozen']['median_rank']:.0f}), AA {r['AA_frozen']['top5']:.3f}. "
          f"n={any_d['n']}, DB={any_d['db_chains']}, chance top5 "
          f"{5.0/(any_d['db_chains']-1):.4f}.\n")


def axis2():
    for pref, title in (("mixedtrain_", "Axis 2a — **TRAIN-set** retrieval (the capacity gate: can it "
                                        "fit the ligand axis at all?)"),
                        ("mixed_", "Axis 2b — mixed held-out retrieval"),
                        ("mixed_scafclean_", "Axis 2c — held-out, scaffold-unseen subset "
                                             "(clean on protein cluster AND scaffold)")):
        data = scan(pref)
        if pref == "mixed_":     # don't let the scafclean files fall into the plain bucket
            data = {c: [(t, d) for t, d in v if not t.startswith("scafclean_")]
                    for c, v in data.items()}
            data = {c: v for c, v in data.items() if v}
        if not data:
            continue

        def rows(entries):
            g = lambda d, k, f: d["results"].get(k, {}).get(f)
            return [agg([g(d, "ppi", "top5") for _t, d in entries]),
                    agg([g(d, "pl", "top5") for _t, d in entries]),
                    agg([g(d, "pl", "mrr") for _t, d in entries]),
                    agg([g(d, "pl", "median_rank") for _t, d in entries], "{:.0f}")]

        table(title, ["PPI top5", "P–L top5", "P–L MRR", "P–L medR"], rows, data)
        d0 = next(iter(data.values()))[0][1]["results"]
        for k, lab in (("ppi", "PPI"), ("pl", "P–L")):
            e = d0.get(k, {})
            if e.get("db"):
                print(f"- {lab}: DB {e['db']}, chance top5 {e['chance_top5']}, "
                      f"chance medR ~{e['db']/2:.0f}")
        print()


def axis3():
    data = scan("neosurf_")
    if not data:
        return

    def rows(entries):
        g = lambda d, k, f: (d.get(k) or {}).get(f)
        return [agg([g(d, "with_ligand", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "no_ligand", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "composite", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "composite_noligand", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "with_ligand", "top5") for _t, d in entries]),
                agg([g(d, "composite", "top5") for _t, d in entries])]

    table("Axis 3 — neosurface benchmark (28 cases; median rank, lower is better)",
          ["sep-surf medR", "sep-surf no-lig medR", "**composite medR**",
           "composite no-lig medR", "sep-surf top5", "composite top5"], rows, data)
    d0 = next(iter(data.values()))[0][1]
    print(f"DB = {d0['db_chains']} chains ({d0['n_decoys']} held-out decoys); chance medR "
          f"~{d0['db_chains']/2:.0f}, chance top5 {d0['chance_top5']}. "
          "`composite` = protein and drug on ONE surface; `composite_noligand` drops the drug's own "
          "rows, isolating the drug *reshaping the protein surface*.\n")


def robustness():
    data = scan("robust_")
    if not data:
        return

    def rows(entries):
        g = lambda d, k, f: d["results"].get(k, {}).get(f)
        return [agg([g(d, "Pholo_to_lig", "top5") for _t, d in entries]),
                agg([g(d, "Paf3_to_lig", "top5") for _t, d in entries]),
                agg([d["robustness"]["protein_query_top5_drop"] for _t, d in entries], "{:+.3f}"),
                agg([g(d, "Pholo_to_lig", "median_rank") for _t, d in entries], "{:.0f}"),
                agg([g(d, "Paf3_to_lig", "median_rank") for _t, d in entries], "{:.0f}")]

    table("Axis 4 (north star) — ligand-axis holo→AF3-apo robustness",
          ["P(holo)→lig top5", "P(AF3)→lig top5", "drop", "holo medR", "AF3 medR"], rows, data)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RES = sys.argv[1]
    axis1(); axis2(); axis3(); robustness()
