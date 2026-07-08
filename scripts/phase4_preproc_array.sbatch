#!/bin/bash
# Phase-4 reference-preprocessing as a SLURM ARRAY job (user directive: up to 500 subsets, %500
# concurrency — collapses the single-node loop's ~40 wall-hr into ~1 hr at the same total cost).
# Each array task preprocesses one ~10-complex subset (logs/phase4/preproc_arr/chunk_<idx>.txt),
# resumable (holo_prep_batch.sh SKIPs complexes whose descriptors already exist). CPU-only.
# Submit: sbatch --array=0-477%500 scripts/phase4_preproc_array.sbatch
#SBATCH --job-name=p4_prep_arr
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/prep_arr/%A_%a.out

set -u
CHUNK=$(printf '/scratch/ymeng/masif-graph/logs/phase4/preproc_arr/chunk_%04d.txt' "$SLURM_ARRAY_TASK_ID")
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK n=$(grep -cE '^[A-Za-z0-9]+_' "$CHUNK" 2>/dev/null) date=$(date)"
cd /scratch/ymeng/masif-graph
# NP=1: each array task processes its ~10 ids serially (the .sif descriptor net is the mem hog;
# parallelism comes from the 478 concurrent array tasks, not from within a task).
bash scripts/holo_prep_batch.sh "$CHUNK" 1
echo "=== task $SLURM_ARRAY_TASK_ID DONE $(date) ==="
