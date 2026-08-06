#!/bin/bash
#SBATCH --job-name=p5_af3_surf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase5/af3_surf_%j.out
set -u
IDLIST="${1:?}"; NP="${2:-16}"
echo "host=$(hostname) date=$(date) idlist=$IDLIST NP=$NP"
cd /scratch/ymeng/masif-graph
bash scripts/p5_af3_surf_batch.sh "$IDLIST" "$NP"
