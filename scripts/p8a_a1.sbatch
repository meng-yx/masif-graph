#!/bin/bash
# Phase-8 A1 — sidechain-blindness ablation ladder, one array task per seed (D8-11: >=2 seeds).
# Usage: sbatch --array=0-1 scripts/p8a_a1.sbatch
#SBATCH --job-name=p8a_a1
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a1/a1_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
ROOT=/scratch/ymeng/masif-graph
export PYTHONPATH=$ROOT/src
cd $ROOT

# The two PPI-only seeds whose published HH top5 are 0.651 / 0.638 (Phase-7 §2, 0.644 +/- 0.007).
CKPTS=(/work/upthomae/Meng/phase6C/ret_ppionly_best.pt /work/upthomae/Meng/phase7/ret_ppionly_s1_best.pt)
TAGS=(ppionly_s0 ppionly_s1)
I=${SLURM_ARRAY_TASK_ID:-0}

echo "=== A1 seed $I  ckpt=${CKPTS[$I]}  $(date) ==="
$PY -u -m masif_graph.p8.a1_sidechain \
  --data /work/upthomae/Meng/phase6C/npz_eval \
  --ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt "${CKPTS[$I]}" --tag "${TAGS[$I]}" --seed "$I" \
  --out-dir logs/phase8A/a1
echo "=== A1 DONE seed $I $(date) ==="
