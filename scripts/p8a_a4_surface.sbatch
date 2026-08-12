#!/bin/bash
# Phase-8 A4 — reference surfaces for the mined ASU chain pairs of one chunk of ENTRIES.
# Chunked by entry (not by pair) so all pairs sharing a chain land in the same task and cannot
# race each other on the per-chain .ply triangulation.
#SBATCH --job-name=p8a_a4surf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a4/surf_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
cd /scratch/ymeng/masif-graph
ENTRIES=logs/phase8A/a4/surf_entries.txt
PAIRS=logs/phase8A/a4/surf_pairs.txt
N=${SLURM_ARRAY_TASK_COUNT:-20}; I=${SLURM_ARRAY_TASK_ID:-0}
k=0
while read -r E; do
  [ -z "$E" ] && continue
  if [ $((k % N)) -eq "$I" ]; then
    grep "^$E " "$PAIRS" | while read -r pdb c1 c2; do
      bash scripts/p8a_asu_surface.sh "$pdb" "$c1" "$c2"
    done
  fi
  k=$((k+1))
done < "$ENTRIES"
echo "=== A4 SURF TASK $I DONE $(date) ==="
