#!/bin/bash
# Phase-6 C(b).1 — PDBbind refined preprocessing as a SLURM array (one chunk of ids per task).
# Submit: sbatch --array=0-<K-1>%250 scripts/p6C_pdbbind_array.sbatch <chunkdir> <outdir>
#SBATCH --job-name=p6C_plprep
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/plprep/%A_%a.out

set -u
export HOME=/home/ymeng
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CHUNKDIR="${1:-/scratch/ymeng/masif-graph/logs/phase6C/pl_chunks}"
OUTDIR="${2:-/work/upthomae/Meng/phase6C/npz_pl}"
CHUNK=$(printf '%s/chunk_%04d.txt' "$CHUNKDIR" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/ymeng/masif-graph/tmp/p6c_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUTDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK out=$OUTDIR start=$(date)"
n=0; ok=0
while read -r id; do
  [ -z "$id" ] && continue
  n=$((n+1))
  if /scratch/ymeng/masif-graph/scripts/p6C_pdbbind_one.sh "$id" "$OUTDIR"; then ok=$((ok+1)); fi
done < "$CHUNK"
echo "=== task $SLURM_ARRAY_TASK_ID DONE ok=$ok/$n $(date) ==="
