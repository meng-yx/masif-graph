#!/bin/bash
#SBATCH --job-name=p4-b0-probe
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_b0/b0_probe-%j.out

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

echo "[$(date)] B.0 probe start (pos_sc)"
python -u scripts/p4_b0_probe.py pos_sc
echo "[$(date)] B.0 probe start (pos / dense)"
python -u scripts/p4_b0_probe.py pos
echo "[$(date)] B.0 probe done"
