#!/bin/bash
# Phase-6 C(c).3 — the three evaluation axes for one checkpoint. ALWAYS on a compute node
# (the Phase-5 gate OOM-kills the login node).
# Usage: sbatch scripts/p6C_gate.sbatch <tag> <ckpt> <npz_eval_dir>
#SBATCH --job-name=p6C_gate
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/gate_%x_%j.out
# NOTE: no `set -u` here. conda's gromacs deactivate hook dereferences unset variables, which
# under `set -u` kills the job 8 s in (job 65982977). Use the env's interpreter directly instead.
export HOME=/home/ymeng
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
export PYTHONPATH=/scratch/ymeng/masif-graph/src
cd /scratch/ymeng/masif-graph
TAG="${1:?tag}"; CKPT="${2:?ckpt}"; EVALDIR="${3:-/work/upthomae/Meng/phase6C/npz_eval}"
P6=/work/upthomae/Meng/phase6C
OUT=logs/phase6C/results; mkdir -p "$OUT"

echo "=== AXIS 1: do-no-harm PPI gate (Phase-5 287-clean, 4-cell) tag=$TAG $(date) ==="
for PK in pos pos_sc; do
  $PY -u -m masif_graph.p5.retrieval_bench --data "$EVALDIR" \
    --ids logs/phase5/eval_sc304_clean_vs_enc.txt --ckpt "$CKPT" --center --pos-key $PK \
    --out "$OUT/gate_${TAG}_${PK}.json"
done

echo "=== AXIS 2: mixed held-out retrieval (PPI holdout + protein-ligand holdout) $(date) ==="
$PY -u -m masif_graph.p6.mixed_bench \
  --ppi-data "$P6/npz_ppi" --ppi-ids "$P6/split/val_ppi_stageB.txt" \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/val_pl.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_${TAG}.json"
echo "--- axis 2b: scaffold-unseen subset (clean on BOTH leakage axes) ---"
$PY -u -m masif_graph.p6.mixed_bench \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/val_pl_scaffold_unseen.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_scafclean_${TAG}.json"
echo "--- axis 2c: scaffold-DEDUPLICATED holdout (one complex per scaffold) ---"
echo "    38% of val_pl share a Murcko scaffold with another val complex, so a congeneric decoy can"
echo "    legitimately outrank the true ligand and depress top-1. This variant removes that ambiguity."
$PY -u -m masif_graph.p6.mixed_bench \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/val_pl_scafdedup.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_scafdedup_${TAG}.json"

echo "--- axis 2d: TRAIN-set retrieval (same sizes as the holdout) ---"
echo "    Separates 'cannot learn' from 'cannot generalise'. If train-set retrieval is high while"
echo "    the holdout is at chance, the complementarity learned is complex-specific, not general."
$PY -u -m masif_graph.p6.mixed_bench \
  --ppi-data "$P6/npz_ppi" --ppi-ids "$P6/split/train_ppi_sample197.txt" \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/train_pl_sample300.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixedtrain_${TAG}.json"

echo "=== AXIS 3: neosurface benchmark (28 ligand-induced cases) $(date) ==="
$PY -u -m masif_graph.p6.neosurf_bench --neosurf-data "$P6/npz_neosurf" \
  --decoy-data "$P6/npz_eval" --decoy-ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt "$CKPT" --center --lig-radius 6.0 --max-db-atoms 1500 \
  --out "$OUT/neosurf_${TAG}.json"
echo "=== GATE DONE $TAG $(date) ==="
