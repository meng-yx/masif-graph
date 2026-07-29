#!/bin/bash
#SBATCH --job-name=p4-ret-diag
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_ret/diag-%j.out
set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=8
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph
python -u scripts/p4_ret_diag.py
echo "[diag DONE rc=$?]"
