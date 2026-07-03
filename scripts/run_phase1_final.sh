#!/bin/bash
# Phase-1 final analysis: M1 probe (N=40) + controls + exposure + plots + paired deltas + M2.
# Run from repo root with the masif-graph env python.
set -e
MG=/work/upthomae/Meng/conda_envs/masif-graph
REPO=/scratch/ymeng/masif-graph
cd "$REPO"
export PYTHONPATH="$REPO/src"
N="${1:-40}"

echo "=== [1/6] metric sanity tests ==="
$MG/bin/python - <<'PY'
import tests.test_metric_sanity as t
fns=[f for f in dir(t) if f.startswith("test_")]
for fn in fns: getattr(t,fn)()
print(f"metric sanity: {len(fns)}/{len(fns)} PASS")
PY

echo "=== [2/6] M1 probe (N=$N) ==="
$MG/bin/python -m masif_graph.experiments.run_m1 --n "$N" --out logs/m1

echo "=== [3/6] paired-delta analysis + summary ==="
$MG/bin/python -m masif_graph.experiments.analyze --results logs/m1/m1_results.json --out logs/m1/paired_deltas.json | tee logs/m1/summary.txt

echo "=== [4/6] distance-overlap plots ==="
$MG/bin/python -m masif_graph.experiments.plots --m1_dir logs/m1 --fig_dir docs/figures

echo "=== [5/6] extract used complexes ==="
$MG/bin/python - <<'PY'
import json
r=json.load(open("logs/m1/m1_results.json"))
open("logs/m1/complexes_used.txt","w").write("\n".join(r["complexes_used"])+"\n")
print("used:", len(r["complexes_used"]))
PY

echo "=== [6/6] M2 global-alignment prototype (first 10 used complexes) ==="
$MG/bin/python -m masif_graph.experiments.run_m2 --ids logs/m1/complexes_used.txt --out logs/m2 --n_complexes 10

echo "=== DONE $(date '+%F %T') ==="
