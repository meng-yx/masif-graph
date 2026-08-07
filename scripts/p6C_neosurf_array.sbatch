#!/bin/bash
# Phase-6 C(c).3 axis 3 — neosurface benchmark preprocessing (14 ternary systems, 1 per task).
# Submit: sbatch --array=0-13 scripts/p6C_neosurf_array.sbatch <outdir>
#SBATCH --job-name=p6C_neosurf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/neosurf/%A_%a.out
set -u
export HOME=/home/ymeng OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="${1:-/work/upthomae/Meng/phase6C/npz_neosurf}"
BENCH=/scratch/ymeng/masif-graph/masif-neosurf-af2/computational_benchmark/benchmark_pdbs.txt
export TMPDIR="/scratch/ymeng/masif-graph/tmp/nb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUT"; trap 'rm -rf "$TMPDIR"' EXIT
LINE=$(grep -v '^#' "$BENCH" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
PDB=$(echo "$LINE" | cut -d, -f1); C1=$(echo "$LINE" | cut -d, -f2); C2=$(echo "$LINE" | cut -d, -f3)
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID sys=$PDB $C1/$C2 start=$(date)"
/scratch/ymeng/masif-graph/scripts/p6C_neosurf_one.sh "$PDB" "$C1" "$C2" "$OUT"
echo "=== neosurf task $SLURM_ARRAY_TASK_ID rc=$? $(date) ==="
