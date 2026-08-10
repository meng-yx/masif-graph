#!/bin/bash
# Phase-7 S2 — ligand-alone surfaces for the whole PDBbind corpus, as a SLURM array.
# Submit: sbatch --array=0-<K-1>%120 scripts/p7_lig_surface_array.sbatch <chunkdir> <outdir>
#SBATCH --job-name=p7_ligsurf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/ligsurf/%A_%a.out
set -u
export HOME=/home/ymeng LC_ALL=C.UTF-8 LANG=C.UTF-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CHUNKDIR="${1:-/scratch/ymeng/masif-graph/logs/phase7/lig_chunks}"
OUTDIR="${2:-/work/upthomae/Meng/phase7/lig_surf}"
CHUNK=$(printf '%s/chunk_%04d.txt' "$CHUNKDIR" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/ymeng/masif-graph/tmp/p7_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUTDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK start=$(date)"
bash /scratch/ymeng/masif-graph/scripts/p7_lig_surface_run.sh "$CHUNK" "$OUTDIR" selfpqr
echo "=== task $SLURM_ARRAY_TASK_ID DONE $(date) ==="
