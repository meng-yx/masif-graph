#!/bin/bash
# Stage R prep — 26-D graphs for the repacked (apo-proxy) training corpus, written next to the holo
# graphs so the trainer can pair them. Global context is added at LOAD time, not here.
#SBATCH --job-name=p8r_rpg
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8R/rpgraph_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
N=${SLURM_ARRAY_TASK_COUNT:-8}; I=${SLURM_ARRAY_TASK_ID:-0}
awk -v n="$N" -v i="$I" 'NR % n == i' logs/phase8R/rp_train_ids.txt > /tmp/rpg_$I.txt
$PY -u -m masif_graph.p4.precompute --ids /tmp/rpg_$I.txt \
    --out /work/upthomae/Meng/phase7/npz_ppi --state rp --unified
echo "=== RP GRAPHS TASK $I DONE $(date) ==="
