#!/bin/bash
# Phase-8 A3 — rigid pose baseline + F3 timing. One array task per checkpoint.
# Stage-A (atom-level contrastive) and Stage-B (chain-level retrieval) checkpoints are BOTH run:
# pose prediction needs atom-level correspondence, which only Stage A was ever trained for, so
# scoring only the retrieval-tuned model would have been an unfair test.
# Usage: sbatch --array=0-3 scripts/p8a_a3.sbatch
#SBATCH --job-name=p8a_a3
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a3/a3_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
ROOT=/scratch/ymeng/masif-graph
export PYTHONPATH=$ROOT/src
cd $ROOT

CKPTS=(/work/upthomae/Meng/phase6C/vicreg_ppionly.pt \
       /work/upthomae/Meng/phase7/vicreg_ppionly_s1.pt \
       /work/upthomae/Meng/phase6C/ret_ppionly_best.pt \
       /work/upthomae/Meng/phase7/ret_ppionly_s1_best.pt)
TAGS=(stageA_s0 stageA_s1 stageB_s0 stageB_s1)
I=${SLURM_ARRAY_TASK_ID:-0}

echo "=== A3 ${TAGS[$I]}  ckpt=${CKPTS[$I]}  $(date) ==="
$PY -u -m masif_graph.p8.a3_pose \
  --data /work/upthomae/Meng/phase6C/npz_eval \
  --ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt "${CKPTS[$I]}" --seed "$I" \
  --out "logs/phase8A/a3/pose_${TAGS[$I]}.json"
echo "=== A3 DONE ${TAGS[$I]} $(date) ==="
