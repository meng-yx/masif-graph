#!/bin/bash
# Thin SLURM wrapper so the scale-up launcher runs decoupled from the conductor's claude session.
# Submit chained on the precompute array:
#   sbatch --dependency=afterany:65442563 scripts/p4_scaleup_launch.sbatch
# SLURM starts it when precompute finishes → it builds the leakage-guarded train-ids and submits the
# Kuma 2x2. Idempotent + lock-guarded (see p4_scaleup_launch.sh), so safe to resubmit.
#SBATCH --job-name=p4_launch
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/scaleup_launch_%j.out
export HOME=/home/ymeng
export PATH="/usr/bin:/bin:$PATH"
bash /scratch/ymeng/masif-graph/scripts/p4_scaleup_launch.sh
