#!/usr/bin/env python
"""Assemble the Phase-6C result JSONs into the markdown tables for docs/19-phase6C-results.md.

Reads whatever exists under logs/phase6C/results/ and prints one section per axis, so the results
doc quotes numbers that came out of the artefacts rather than out of a transcript.
"""
from __future__ import annotations

import json
import os
import sys

RES = "logs/phase6C/results"
ORDER = ["randinit", "phase5_14d", "ppionly", "plonly", "combined"]
LABEL = {"randinit": "random-init (chance)", "phase5_14d": "Phase-5 14-D encoder",
         "ppionly": "26-D PPI-only (control)", "plonly": "26-D ligand-only (control)",
         "combined": "26-D COMBINED (deliverable)"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def tags():
    seen = []
    for f in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
        for pref in ("gate_", "mixed_", "neosurf_"):
            if f.startswith(pref) and f.endswith(".json"):
                t = f[len(pref):-len(".json")].replace("_pos_sc", "").replace("_pos", "")
                for sub in ("scafclean_", "scafdedup_"):     # axis-2 variants share the model tag
                    if t.startswith(sub):
                        t = t[len(sub):]
                if t not in seen:
                    seen.append(t)
    return [t for t in ORDER if t in seen] + [t for t in seen if t not in ORDER]


def axis1():
    print("### Axis 1 — do-no-harm PPI gate (Phase-5 287-clean, dense `pos` patches)\n")
    print("| model | HH top5 | HH medR | AA top5 | AA medR | holo->AA drop | shuffled top5 |")
    print("|---|---|---|---|---|---|---|")
    for t in tags():
        d = load(f"gate_{t}_pos.json")
        if not d:
            continue
        r = d["results"]
        hh, aa = r["HH_learned"], r["AA_learned"]
        sh = r.get("HH_learned_shuffled", {})
        print(f"| {LABEL.get(t, t)} | {hh.get('top5', 0):.3f} | {hh.get('median_rank', 0):.0f} "
              f"| {aa.get('top5', 0):.3f} | {aa.get('median_rank', 0):.0f} "
              f"| {hh.get('top5', 0) - aa.get('top5', 0):+.3f} | {sh.get('top5', 0):.3f} |")
    d = load(f"gate_{tags()[0]}_pos.json") if tags() else None
    if d:
        r = d["results"]
        print(f"| *frozen MaSIF (same patches)* | {r['HH_frozen'].get('top5', 0):.3f} "
              f"| {r['HH_frozen'].get('median_rank', 0):.0f} | {r['AA_frozen'].get('top5', 0):.3f} "
              f"| {r['AA_frozen'].get('median_rank', 0):.0f} "
              f"| {r['HH_frozen'].get('top5', 0) - r['AA_frozen'].get('top5', 0):+.3f} "
              f"| {r.get('HH_frozen_shuffled', {}).get('top5', 0):.3f} |")
        print(f"\nn = {d['n']} complexes, DB = {d['db_chains']} chains "
              f"(chance top5 ~ {5.0 / (d['db_chains'] - 1):.4f}).\n")


def axis2():
    print("### Axis 2 — mixed held-out retrieval (same-type decoy pool, centered)\n")
    print("| model | PPI top5 | PPI medR | P-L top5 | P-L medR | P-L pocket->ligand | "
          "P-L ligand->pocket | P-L shuffled |")
    print("|---|---|---|---|---|---|---|---|")
    for t in tags():
        d = load(f"mixed_{t}.json")
        if not d:
            continue
        r = d["results"]
        g = lambda k, f="top5": r.get(k, {}).get(f, float("nan"))
        print(f"| {LABEL.get(t, t)} | {g('ppi'):.3f} | {g('ppi','median_rank'):.0f} "
              f"| {g('pl'):.3f} | {g('pl','median_rank'):.0f} | {g('pl_query_protein'):.3f} "
              f"| {g('pl_query_ligand'):.3f} | {g('shuffled_pl'):.3f} |")
    d = load(f"mixed_{tags()[0]}.json") if tags() else None
    if d:
        r = d["results"]
        print(f"\nchance top5: PPI {r.get('ppi', {}).get('chance_top5')}, "
              f"P-L {r.get('pl', {}).get('chance_top5')}.\n")
    for pref, title in (("mixed_scafclean_",
                         "Scaffold-unseen subset (clean on protein cluster AND ligand scaffold)"),
                        ("mixed_scafdedup_",
                         "Scaffold-deduplicated holdout (one complex per scaffold; removes the "
                         "congeneric-decoy ambiguity that depresses top-1)")):
        print(f"{title}:\n")
        print("| model | P-L top5 | P-L top1 | P-L medR | n | chance top5 |")
        print("|---|---|---|---|---|---|")
        for t in tags():
            d = load(f"{pref}{t}.json")
            if not d:
                continue
            p = d["results"].get("pl", {})
            print(f"| {LABEL.get(t, t)} | {p.get('top5', 0):.3f} | {p.get('top1', 0):.3f} "
                  f"| {p.get('median_rank', 0):.0f} | {p.get('n', 0)} | {p.get('chance_top5')} |")
        print()


def axis3():
    print("### Axis 3 — neosurface benchmark (28 ligand-induced cases)\n")
    print("| model | with-ligand top5 | medR | no-ligand top5 | medR | ligand helps/hurts/ties |")
    print("|---|---|---|---|---|---|")
    for t in tags():
        d = load(f"neosurf_{t}.json")
        if not d:
            continue
        w, n, e = d["with_ligand"], d["no_ligand"], d["ligand_effect"]
        print(f"| {LABEL.get(t, t)} | {w['top5']:.3f} | {w['median_rank']:.0f} "
              f"| {n['top5']:.3f} | {n['median_rank']:.0f} "
              f"| {e['n_better_with_ligand']}/{e['n_worse']}/{e['n_tied']} |")
    d = load(f"neosurf_{tags()[0]}.json") if tags() else None
    if d:
        print(f"\nDB = {d['db_chains']} chains ({d['n_decoys']} held-out decoy chains), "
              f"n = 28 cases, chance top5 = {d['chance_top5']}.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RES = sys.argv[1]
    axis1(); axis2(); axis3()
