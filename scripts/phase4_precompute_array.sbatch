#!/bin/bash
# Phase-4 hetero-graph npz precompute as a SLURM ARRAY (parallelizes p4.precompute over the full
# training set — ~3 hr single-thread otherwise). Each task builds npz for one ~10-complex chunk into
# the shared /work out-dir (Kuma reads it for training). masif-graph env (biotite). Idempotent-ish:
# re-running overwrites the same per-complex npz. CPU-only.
# Submit: sbatch --array=0-<K-1>%500 scripts/phase4_precompute_array.sbatch <chunkdir> <outdir>
#SBATCH --job-name=p4_pre_npz
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/pre_npz/%A_%a.out

set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # cap threads (contended nodes thrash at 72)
CHUNKDIR="${1:-/scratch/ymeng/masif-graph/logs/phase4/precompute_arr}"
OUTDIR="${2:-/work/upthomae/Meng/phase4/stageA_full_npz}"
CHUNK=$(printf '%s/chunk_%04d.txt' "$CHUNKDIR" "$SLURM_ARRAY_TASK_ID")
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
mkdir -p "$OUTDIR"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK out=$OUTDIR date=$(date)"
cd /scratch/ymeng/masif-graph
PYTHONPATH=/scratch/ymeng/masif-graph/src $PY -m masif_graph.p4.precompute --ids "$CHUNK" --out "$OUTDIR"
echo "=== precompute task $SLURM_ARRAY_TASK_ID DONE $(date) ==="
