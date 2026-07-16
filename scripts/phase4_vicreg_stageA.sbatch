#!/bin/bash
# Phase-4 full-scale Stage-A ANTI-COLLAPSE run — the clean from-scratch verdict.
# The naive scale-up (lr1e-3) AND the calmer stabilize (lr3e-4/clip1.0) BOTH diverged at 4811 complexes:
# the diagnostic (docs/10) proved the cause is REPRESENTATION COLLAPSE (z_std->0.003 from ep1), which then
# forces the temperature to its 0.01 floor and ||T|| to run away until gradients explode. Optim tuning
# can't fix a collapsed representation — so this run adds the actual anti-collapse levers:
#   --vicreg-var/-cov : force per-dim variance>=1 + decorrelate (stops the point-collapse; THE fix)
#   --freeze-tau      : temperature pinned at 0.1 (can't run to the floor)
#   --t-wd            : weight-decay on the bilinear T (bounds ||T|| growth)
# CPU smoke (12 cplx) confirmed: z_std 0.04 stable (vs 0.003 collapse), tau fixed, ||T|| flat ~1.7,
# grad-norm 4-6 (vs 5.6e5), loss decreasing. This is the scaled test of that fix + the honest verdict:
# does the STABILIZED from-scratch GNN reach the frozen SC ceiling (~0.90) or plateau below it?
# Usage: sbatch phase4_vicreg_stageA.sbatch <dense|sc> <seed>
#SBATCH --job-name=p4_vicreg
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=88G
#SBATCH --time=05:00:00
#SBATCH --output=/work/upthomae/Meng/phase4/vicreg_%x_%j.out

set -e
TRAIN_POS="${1:?usage: sbatch phase4_vicreg_stageA.sbatch <dense|sc> <seed>}"
SEED="${2:?need seed}"
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=/work/upthomae/Meng/phase4/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
BASE=/work/upthomae/Meng/phase4
cd "$BASE"

echo "host=$(hostname) date=$(date) VICREG train_pos=$TRAIN_POS seed=$SEED"
echo "config: vicreg_var=2.0 vicreg_cov=0.04 freeze_tau@0.1 t_wd=1e-3 lr=5e-4 clip=1.0 cosine ep=50"
$PY -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

$PY -m masif_graph.p4.train \
  --data "$BASE/stageA_full_npz" \
  --train-ids "$BASE/stageA_full_train_ids.txt" \
  --val-ids "$BASE/stageA_heldout_ids.txt" \
  --train-pos "$TRAIN_POS" \
  --epochs 50 --eval-every 5 --lr 5e-4 --cosine --grad-clip 1.0 --stream \
  --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 \
  --d 64 --d-out 32 --layers 4 --bank 128 --seed "$SEED" \
  --device cuda \
  --out "$BASE/vicreg_${TRAIN_POS}_seed${SEED}.json" \
  --save "$BASE/vicreg_${TRAIN_POS}_best_seed${SEED}.pt"
echo "DONE VICREG train_pos=$TRAIN_POS seed=$SEED $(date)"
