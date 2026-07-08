#!/bin/bash
# Phase-4 FULL-SCALE Stage-A run (scale-up): the from-scratch hetero-GNN trained on the full clean
# training set (~4,700 complexes) instead of 90 — does 90→~4,700 close the held-out AUC gap to the
# frozen 0.947 sc ceiling (data-limited) or plateau (architecture-limited)?
# One job = one (train_pos, seed). Submit dense+sc × seeds for the 2×N comparison.
# Usage: sbatch phase4_scaleup_stageA.sbatch <dense|sc> <seed> [epochs]
#SBATCH --job-name=p4_full
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --time=06:00:00
# NOTE: Kuma h100 MaxMemPerCPU=5900 MB → mem/cpus must be ≤5.9G. 88G/16 = 5.5G/cpu. Do not lower cpus.
#SBATCH --output=/work/upthomae/Meng/phase4/scaleup_%x_%j.out

set -e
TRAIN_POS="${1:?usage: sbatch phase4_scaleup_stageA.sbatch <dense|sc> <seed> [epochs]}"
SEED="${2:?need seed}"
EPOCHS="${3:-60}"
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
BASE=/work/upthomae/Meng/phase4
cd "$BASE"

echo "host=$(hostname) date=$(date) train_pos=$TRAIN_POS seed=$SEED epochs=$EPOCHS"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

$PY -m masif_graph.p4.train \
  --data "$BASE/stageA_full_npz" \
  --train-ids "$BASE/stageA_full_train_ids.txt" \
  --val-ids "$BASE/stageA_heldout_ids.txt" \
  --train-pos "$TRAIN_POS" \
  --epochs "$EPOCHS" --eval-every 5 --lr 1e-3 --cosine --stream \
  --d 64 --d-out 32 --layers 4 --bank 128 --seed "$SEED" \
  --device cuda \
  --out "$BASE/scaleup_${TRAIN_POS}_seed${SEED}.json" \
  --save "$BASE/scaleup_${TRAIN_POS}_best_seed${SEED}.pt"
echo "DONE train_pos=$TRAIN_POS seed=$SEED $(date)"
