"""Phase-8 A4 — does ANY bio-vs-crystal signal exist, and does it beat interface area?

Stage 3's whole job is to separate `biological_contact` from a merely `complementary_contact`.
Interface area is the obvious shortcut and D8-7 makes a BSA-only baseline a hard control, so the
question here is not "can we classify?" but "can anything we have beat BSA?".

Reported as a WARNING, not a gate (docs/24 §6.2). A linear probe failing to beat BSA does not
condemn Stage 3 — a trained pose-level network is a different model — but it would say the premise
deserves scrutiny before Stage C rather than after.

Discipline:
  * **grouped 5-fold CV by PDB entry** — interfaces from one entry share chains, so a random split
    would let the same chain appear on both sides and inflate every arm;
  * the rare class is **crystal contacts**, not biological ones, so average precision is computed
    for detecting crystal contacts, and prevalence is printed next to it;
  * a **shuffled-label control** must collapse both arms to chance;
  * arms are compared on the identical folds.

Usage:
  python -m masif_graph.p8.a4_probe --interfaces logs/phase8A/a4/interfaces.json \
      --out logs/phase8A/a4/probe.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _feats(iface, use_embed):
    """Feature matrix + names. BSA-only arm uses column 0 alone."""
    bsa = np.array([x["bsa"] for x in iface], float)
    nct = np.array([x["n_contacts"] for x in iface], float)
    a1 = np.array([x["n_atom1"] for x in iface], float)
    a2 = np.array([x["n_atom2"] for x in iface], float)
    cols = [bsa, np.log1p(nct), np.log1p(np.minimum(a1, a2)), np.log1p(np.maximum(a1, a2)),
            bsa / np.maximum(np.minimum(a1, a2), 1.0)]
    names = ["bsa", "log_n_contacts", "log_min_chain_atoms", "log_max_chain_atoms",
             "bsa_per_atom_small_chain"]
    if use_embed:
        for k in ("emb_score_mean", "emb_score_max", "emb_score_median", "emb_score_p90"):
            if all(k in x for x in iface):
                cols.append(np.array([x[k] for x in iface], float))
                names.append(k)
    return np.column_stack(cols), names


def _cv(X, y, groups, seed=0, folds=5):
    """Grouped CV; returns out-of-fold scores. Standardisation is fit on TRAIN folds only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    gk = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    for tr, te in gk.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=seed))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _metrics(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auroc": None, "ap_crystal": None}
    return {"n": int(len(y)),
            "auroc": float(roc_auc_score(y, s)),
            "ap_crystal": float(average_precision_score(1 - y, -s)),
            "prevalence_crystal": float((y == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfaces", default="logs/phase8A/a4/interfaces.json")
    ap.add_argument("--out", default="logs/phase8A/a4/probe.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.interfaces))
    iface = d["interfaces"]
    y = np.array([x["bio"] for x in iface])
    groups = np.array([x["pdb"] for x in iface])
    use_embed = bool(iface) and all("emb_score_mean" in x for x in iface)
    X, names = _feats(iface, use_embed)
    print(f"{len(iface)} interfaces  bio={int(y.sum())} crystal={int((y==0).sum())}  "
          f"entries={len(np.unique(groups))}  embed_arm={use_embed}", flush=True)
    if len(np.unique(y)) < 2:
        raise SystemExit("only one class present — nothing to probe")

    arms = {"bsa_only": X[:, :1], "structural": X[:, :5]}
    if use_embed:
        arms["structural_plus_embedding"] = X
    out = {"n_interfaces": len(iface), "n_bio": int(y.sum()), "n_crystal": int((y == 0).sum()),
           "n_entries": int(len(np.unique(groups))), "feature_names": names,
           "embed_arm": use_embed, "arms": {}, "controls": {}}
    for k, Xa in arms.items():
        out["arms"][k] = _metrics(y, _cv(Xa, y, groups, seed=args.seed))
    # shuffled-label control: must collapse to chance in every arm
    rng = np.random.default_rng(args.seed)
    ysh = y.copy()
    rng.shuffle(ysh)
    for k, Xa in arms.items():
        out["controls"][f"{k}_shuffled"] = _metrics(ysh, _cv(Xa, ysh, groups, seed=args.seed))

    print("=" * 78)
    print(f"A4 probe (grouped 5-fold CV by PDB entry; crystal prevalence "
          f"{(y == 0).mean():.3f})")
    for k, v in out["arms"].items():
        print(f"  {k:28s} AUROC {v['auroc']:.3f}   AP(crystal) {v['ap_crystal']:.3f}   n={v['n']}")
    print("  -- shuffled-label control (must be ~chance) --")
    for k, v in out["controls"].items():
        print(f"  {k:28s} AUROC {v['auroc']:.3f}   AP(crystal) {v['ap_crystal']:.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
