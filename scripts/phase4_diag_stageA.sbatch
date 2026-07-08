#!/bin/bash
# Phase-4 DIAGNOSTIC run — reproduce the full-scale divergence WITH per-epoch instability instrumentation to
# NAME the trigger. Uses the ORIGINAL unstable recipe (lr 1e-3, grad-clip 5.0, cosine) that diverged in the
# 2×2 (3796137-40), now logging every epoch: tau (learnable temperature; ->0.01 floor = temperature runaway),
# |T|2 (bilinear spectral norm; grows = scores blow up), gnorm (pre-clip grad norm; spikes = explosion),
# z_std (embedding spread; ->0 = representation collapse). Whichever moves FIRST at the loss blow-up is the
# root cause. 40 ep captures the transition (dense diverged ~ep35).
# Usage: sbatch phase4_diag_stageA.sbatch <dense|sc> <seed>
#SBATCH --job-name=p4_diag
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --time=04:00:00
#SBATCH --output=/work/upthomae/Meng/phase4/diag_%x_%j.out

set -e
TRAIN_POS="${1:?usage: sbatch phase4_diag_stageA.sbatch <dense|sc> <seed>}"
SEED="${2:?need seed}"
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
BASE=/work/upthomae/Meng/phase4
cd "$BASE"

echo "host=$(hostname) date=$(date) DIAG train_pos=$TRAIN_POS seed=$SEED lr=1e-3 clip=5.0 cosine ep=40"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

$PY -m masif_graph.p4.train \
  --data "$BASE/stageA_full_npz" \
  --train-ids "$BASE/stageA_full_train_ids.txt" \
  --val-ids "$BASE/stageA_heldout_ids.txt" \
  --train-pos "$TRAIN_POS" \
  --epochs 40 --eval-every 5 --lr 1e-3 --cosine --grad-clip 5.0 --stream \
  --d 64 --d-out 32 --layers 4 --bank 128 --seed "$SEED" \
  --device cuda \
  --out "$BASE/diag_${TRAIN_POS}_seed${SEED}.json"
echo "DONE DIAG train_pos=$TRAIN_POS seed=$SEED $(date)"
