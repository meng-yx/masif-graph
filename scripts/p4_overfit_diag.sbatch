#!/bin/bash
#SBATCH --job-name=p4-overfit
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_ret/overfit-%j.out

cd /scratch/ymeng/masif-graph
source ~/.bashrc
conda activate masif-graph
set -eo pipefail
export OMP_NUM_THREADS=16
echo "[$(date)] overfit diag start"
python -u scripts/p4_overfit_diag.py
echo "[$(date)] overfit diag done"
