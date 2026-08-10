#!/bin/bash
#SBATCH --job-name=p7_af3surf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/af3surf/%A_%a.out
set -u
export HOME=/home/ymeng LC_ALL=C.UTF-8 OMP_NUM_THREADS=1
CHUNKDIR="${1:-/scratch/ymeng/masif-graph/logs/phase7/af3_chunks}"
OUTDIR="${2:-/work/upthomae/Meng/phase7/npz_pl}"
CHUNK=$(printf '%s/chunk_%04d.txt' "$CHUNKDIR" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/ymeng/masif-graph/tmp/p7af_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUTDIR"; trap 'rm -rf "$TMPDIR"' EXIT
while read -r id; do [ -z "$id" ] && continue
  /scratch/ymeng/masif-graph/scripts/p7_af3_pl_surface.sh "$id" "$OUTDIR"
done < "$CHUNK"
echo "=== af3surf task $SLURM_ARRAY_TASK_ID DONE $(date) ==="
