#!/bin/bash
# Stage R prep — FASPR fixed-backbone repack of the FULL PPI training corpus.
# Phase-8A A1.2 measured that a repack reproduces 91% of the AF3 apo perturbation the encoder
# feels, at zero GPU cost, so this is the cheap apo augmentation arm of the D8-12 hybrid.
# Multi-chain interface sides (e.g. 1A14_HL_N) are INCLUDED: FASPR repacks them as one unit.
# Usage: sbatch --array=0-99 scripts/p8r_repack_corpus.sbatch
#SBATCH --job-name=p8r_rp
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8R/repack_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
cd /scratch/ymeng/masif-graph
mkdir -p logs/phase8R
IDS=logs/phase8A/a1/repack_corpus_ids.txt
N=${SLURM_ARRAY_TASK_COUNT:-100}; I=${SLURM_ARRAY_TASK_ID:-0}
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search

run_one() {
  id="$1"
  pdb=${id%%_*}; rest=${id#*_}; c1=${rest%%_*}; c2=${rest#*_}
  dd="$REFDATA/descriptors/sc05/all_feat/${pdb}RP_${c1}_${c2}"
  [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ] && { echo "SKIP $id"; return 0; }
  bash /scratch/ymeng/masif-graph/scripts/repack_one.sh "$id"
}
export -f run_one; export REFDATA

awk -v n="$N" -v i="$I" 'NR % n == i' "$IDS" | xargs -P 8 -I{} bash -c 'run_one "$@"' _ {}
echo "=== REPACK CORPUS TASK $I DONE $(date) ==="
