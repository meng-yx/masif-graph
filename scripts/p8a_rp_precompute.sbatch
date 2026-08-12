#!/bin/bash
# Phase-8 A1.2 — build npz graphs for the FASPR-repacked state, next to the existing holo/af3
# graphs so the same Rec/retrieval machinery can load all three.
#SBATCH --job-name=p8a_rp_pre
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a1/rp_precomp_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
$PY -u -m masif_graph.p4.precompute --ids logs/phase8A/a1/repack_ok.txt \
    --out /work/upthomae/Meng/phase6C/npz_eval --state rp --unified
echo "=== RP PRECOMPUTE DONE $(date) ==="
ls /work/upthomae/Meng/phase6C/npz_eval/*__rp__p1.npz 2>/dev/null | wc -l
