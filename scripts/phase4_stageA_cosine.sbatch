#!/bin/bash
# Phase-4 Stage-A STABILIZATION run: identical to phase4_stageA.sbatch (clean 90 train / 60 held-out,
# 3 seeds, d=64/L=4/bank=128, lr=1e-3, 150 ep) EXCEPT --cosine (LR decays 1e-3 → 1e-5). Isolates whether
# the held-out AUC oscillation (baseline: SC best 0.749 but final 0.559±0.10, one seed collapsed) is a
# fixable LR-schedule issue. Same held-out set → directly comparable. Runs from the shared /work stage.
#SBATCH --job-name=p4_stageA_cos
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/work/upthomae/Meng/phase4/stageA_cos_%j.out

set -e
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
BASE=/work/upthomae/Meng/phase4
cd $BASE

echo "host=$(hostname) date=$(date)"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

for SEED in 0 1 2; do
  echo "================= SEED $SEED (cosine) ================="
  $PY -m masif_graph.p4.train \
    --data $BASE/stageA_npz \
    --train-ids $BASE/stageA_train_ids.txt \
    --val-ids $BASE/stageA_heldout_ids.txt \
    --epochs 150 --eval-every 10 --lr 1e-3 --cosine \
    --d 64 --d-out 32 --layers 4 --bank 128 --seed $SEED \
    --device cuda \
    --out $BASE/stageA_cos_result_seed${SEED}.json \
    --save $BASE/stageA_cos_best_seed${SEED}.pt
done
echo "ALL SEEDS DONE $(date)"
