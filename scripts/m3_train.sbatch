#!/bin/bash
#SBATCH --job-name=m3-train
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase3/m3_train-%j.out
# M3 pilot training on a clean Jed node (CPU). Args via env: STEPS, INVW, LR, SEED, TAG.
set -u
export PYTHONPATH=/scratch/ymeng/masif-graph/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /scratch/ymeng/masif-graph
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
TAG="${TAG:-run}"
$PY -m masif_graph.m3.train \
  --data logs/phase3/m3_data \
  --train-ids "${TRAIN_IDS:-logs/phase3/m3_train_ids.txt}" \
  --eval-ids "${EVAL_IDS:-logs/phase3/m3_eval_ids.txt}" \
  --out logs/phase3/m3_${TAG} \
  --steps "${STEPS:-1500}" --eval-every "${EVALEVERY:-100}" \
  --lr "${LR:-5e-4}" --inv-weight "${INVW:-1.0}" --weight-decay "${WD:-1e-4}" --reg-weight "${REGW:-0.5}" \
  --val-frac 0.2 --seed "${SEED:-0}" ${NOGRAPH:+--no-graph}
echo "M3_TRAIN_DONE tag=$TAG $(date '+%F %T')"
