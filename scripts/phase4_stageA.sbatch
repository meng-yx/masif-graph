#!/bin/bash
# Phase-4 Stage-A feasibility training on Kuma H100.
# Trains the from-scratch HeteroEncoder + symmetric bilinear T (InfoNCE, holo-only) on 90 holo
# complexes, evaluates held-out holo->holo separation-AUC on 60 disjoint complexes (both disjoint
# from the Phase-3 m1_ids AF3 benchmark). 3 seeds → robust mean±std. Reports median_step_sec (H100)
# for the full-set GPU cost estimate. Runs from the shared /work stage (Kuma has no /scratch access).
#SBATCH --job-name=p4_stageA
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/work/upthomae/Meng/phase4/stageA_%j.out

set -e
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
BASE=/work/upthomae/Meng/phase4
cd $BASE

echo "host=$(hostname) date=$(date)"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

for SEED in 0 1 2; do
  echo "================= SEED $SEED ================="
  $PY -m masif_graph.p4.train \
    --data $BASE/stageA_npz \
    --train-ids $BASE/stageA_train_ids.txt \
    --val-ids $BASE/stageA_heldout_ids.txt \
    --epochs 150 --eval-every 10 --lr 1e-3 \
    --d 64 --d-out 32 --layers 4 --bank 128 --seed $SEED \
    --device cuda \
    --out $BASE/stageA_result_seed${SEED}.json \
    --save $BASE/stageA_best_seed${SEED}.pt
done
echo "ALL SEEDS DONE $(date)"
