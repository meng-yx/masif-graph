#!/bin/bash
# Phase-4 full-scale Stage-A STABILIZATION run. The naive 2×2 (jobs 3796137-40) DIVERGED at 4809 complexes:
# stable to ~ep25 (dense held-out SC ~0.80) then loss explodes to 12-20 and final-epoch AUC collapses to ~0.40
# (below the 0.50 shuffled control). This run tests whether a calmer recipe holds/extends the stable ~0.80 —
# lower peak LR (3e-4 vs 1e-3) + tighter grad-clip (1.0 vs 5.0), cosine, 40 ep. If it plateaus stably → the
# data-scaling read is real (does the plateau beat the 90-complex 0.749?); if it still diverges → deeper recipe
# work needed (honest INCONCLUSIVE stands). Same data/held-out/leakage guard as the scale-up.
# Usage: sbatch phase4_stabilize_stageA.sbatch <dense|sc> <seed>
#SBATCH --job-name=p4_stab
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --time=05:00:00
#SBATCH --output=/work/upthomae/Meng/phase4/stab_%x_%j.out

set -e
TRAIN_POS="${1:?usage: sbatch phase4_stabilize_stageA.sbatch <dense|sc> <seed>}"
SEED="${2:?need seed}"
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
BASE=/work/upthomae/Meng/phase4
cd "$BASE"

echo "host=$(hostname) date=$(date) STABILIZE train_pos=$TRAIN_POS seed=$SEED lr=3e-4 clip=1.0 ep=40"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

$PY -m masif_graph.p4.train \
  --data "$BASE/stageA_full_npz" \
  --train-ids "$BASE/stageA_full_train_ids.txt" \
  --val-ids "$BASE/stageA_heldout_ids.txt" \
  --train-pos "$TRAIN_POS" \
  --epochs 40 --eval-every 5 --lr 3e-4 --cosine --grad-clip 1.0 --stream \
  --d 64 --d-out 32 --layers 4 --bank 128 --seed "$SEED" \
  --device cuda \
  --out "$BASE/stab_${TRAIN_POS}_seed${SEED}.json" \
  --save "$BASE/stab_${TRAIN_POS}_best_seed${SEED}.pt"
echo "DONE STABILIZE train_pos=$TRAIN_POS seed=$SEED $(date)"
