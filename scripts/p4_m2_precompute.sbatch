#!/bin/bash
#SBATCH --job-name=p4-m2-precomp
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_precomp-%j.out

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=8
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

IDS=logs/phase4/m2_train_ids.txt
OUT=logs/phase4/m2_npz
echo "[$(date)] precompute HOLO (162 train)"
python -u -m masif_graph.p4.precompute --ids $IDS --out $OUT --state holo
echo "[$(date)] precompute AF3 (162 train; ~128 available)"
python -u -m masif_graph.p4.precompute --ids $IDS --out $OUT --state af3
echo "[$(date)] m2 train precompute DONE"
touch logs/phase4/M2_TRAIN_PRECOMP_DONE
