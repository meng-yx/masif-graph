#!/bin/bash
# Standalone Jed job that waits for the Kuma 2x2, writes the results table + data-scaling read to
# docs/10, and touches the sentinel — so the scale-up completes even if the conductor session dies.
#SBATCH --job-name=p4_collect
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/scaleup_collect_%j.out
export HOME=/home/ymeng
export PATH="/usr/bin:/bin:$PATH"
bash /scratch/ymeng/masif-graph/scripts/p4_scaleup_collect.sh
