#!/bin/bash
# Phase-8 A4 — 26-D graphs + contacts for the mined ASU chain pairs (embedding arm of the probe).
#SBATCH --job-name=p8a_a4g
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a4/graphs_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
$PY -u -m masif_graph.p4.precompute --ids logs/phase8A/a4/graph_ids.txt \
    --out /work/upthomae/Meng/phase8A/a4_npz --state holo --unified
echo "=== A4 GRAPHS DONE $(date) ==="
ls /work/upthomae/Meng/phase8A/a4_npz/*__holo__p1.npz 2>/dev/null | wc -l
