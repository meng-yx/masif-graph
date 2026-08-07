#!/bin/bash
# Phase-6 C(c).2 — combined-corpus retrain on Kuma (H100). Two stages, Phase-4 anti-collapse recipe.
#   Stage A  VICReg pretrain  (p4.train, atom-level InfoNCE + variance/covariance terms)
#   Stage B  retrieval        (p6.train_unified, chain-level InfoNCE, mixed val, centered)
#
# Usage: sbatch p6C_kuma_pipeline.sbatch <tag> <mode: combined|ppionly> <epochsA> <epochsB> [seed]
# `ppionly` trains the SAME recipe on the PPI half alone — the control that separates "26-D features
# changed something" from "adding protein-ligand data changed something" in the do-no-harm gate.
#SBATCH --job-name=p6C_train
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/work/upthomae/Meng/phase6C/logs/train_%x_%j.out

set -u
TAG="${1:?usage: sbatch p6C_kuma_pipeline.sbatch <tag> <combined|ppionly> <epA> <epB> [seed]}"
MODE="${2:?mode}"
EPA="${3:-30}"; EPB="${4:-40}"; SEED="${5:-0}"
P6=/work/upthomae/Meng/phase6C
PY=/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python
export PYTHONPATH=$P6/src OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$P6/logs"
cd "$P6"

echo "host=$(hostname) tag=$TAG mode=$MODE epA=$EPA epB=$EPB seed=$SEED $(date)"
$PY -c "import torch;print('CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# Three modes, so each claim has the control that isolates its cause:
#   combined  the deliverable            PPI + protein-ligand
#   ppionly   do-no-harm control         same recipe, no ligand data
#   plonly    TRANSFER control           ligands only; if `combined` beats this on the ligand axis,
#                                        PPI complementarity really did transfer through the shared
#                                        26-D atom space, which is the thesis of Workstream C
case "$MODE" in
  combined) PPI_ARGS_B="--ppi-data $P6/npz_ppi --ppi-ids $P6/split/train_ppi.txt"
            PL_ARGS_B="--pl-data $P6/npz_pl --pl-ids $P6/split/train_pl.txt --pl-frac 0.5"
            TRAIN_A=$P6/split/train_all.txt; SELECT=all ;;
  ppionly)  PPI_ARGS_B="--ppi-data $P6/npz_ppi --ppi-ids $P6/split/train_ppi.txt"
            PL_ARGS_B="--pl-frac 0.0"
            TRAIN_A=$P6/split/train_ppi.txt; SELECT=ppi ;;
  plonly)   PPI_ARGS_B=""
            PL_ARGS_B="--pl-data $P6/npz_pl --pl-ids $P6/split/train_pl.txt --pl-frac 1.0"
            TRAIN_A=$P6/split/train_pl.txt; SELECT=pl ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# ---------------- Stage A: VICReg pretrain ----------------
if [ ! -s "$P6/vicreg_${TAG}.pt" ]; then
  $PY -u -m masif_graph.p4.train \
    --data "$P6/npz_all" --train-ids "$TRAIN_A" --val-ids "$P6/split/val_stageA_${MODE}.txt" \
    --train-pos dense --epochs "$EPA" --eval-every 5 --lr 5e-4 --cosine --grad-clip 1.0 --stream \
    --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 \
    --d 64 --d-out 32 --layers 4 --bank 128 --seed "$SEED" --device cuda \
    --out "$P6/vicreg_${TAG}.json" --save "$P6/vicreg_${TAG}.pt"
  echo "[STAGE-A DONE $TAG $(date)]"
else
  echo "[STAGE-A cached $TAG]"
fi

# ---------------- Stage B: chain-level retrieval ----------------
$PY -u -m masif_graph.p6.train_unified \
  $PPI_ARGS_B $PL_ARGS_B \
  --val-ppi-data "$P6/npz_ppi" --val-ppi-ids "$P6/split/val_ppi_stageB.txt" \
  --val-pl-data "$P6/npz_pl" --val-pl-ids "$P6/split/val_pl.txt" \
  --init-ckpt "$P6/vicreg_${TAG}.pt" \
  --epochs "$EPB" --eval-every 4 --batch 32 --patch-key pos --max-patch 128 \
  --w-atom 0.5 --tau-c 0.07 --tau-atom 0.1 \
  --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 \
  --lr 5e-4 --grad-clip 1.0 --cosine --center --seed "$SEED" --device cuda \
  --select-on "$SELECT" \
  --out "$P6/ret_${TAG}_result.json" --save "$P6/ret_${TAG}_best.pt"
echo "[PIPELINE DONE $TAG $(date)]"
touch "$P6/DONE_${TAG}"
