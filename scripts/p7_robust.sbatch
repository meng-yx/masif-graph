#!/bin/bash
# Phase-7 axis 4 — ligand-axis holo->AF3-apo robustness for one checkpoint.
# Usage: sbatch scripts/p7_robust.sbatch <tag> <ckpt>
#SBATCH --job-name=p7_robust
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/robust_%x_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8 OMP_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
ROOT=/scratch/ymeng/masif-graph; export PYTHONPATH=$ROOT/src; cd $ROOT
TAG="${1:?}"; CKPT="${2:?}"; DATA="${3:-/work/upthomae/Meng/phase7/npz_pl}"
mkdir -p logs/phase7/results
$PY -u -m masif_graph.p7.pl_robustness --data "$DATA" \
  --ids /work/upthomae/Meng/phase7/split/val_pl.txt --ckpt "$CKPT" --center \
  --out logs/phase7/results/robust_${TAG}.json
echo "=== ROBUST DONE $TAG $(date) ==="
