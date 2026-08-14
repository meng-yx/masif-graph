#!/bin/bash
#SBATCH --job-name=p8r_gate
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8R/gate/gate_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8 OMP_NUM_THREADS=16
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
TAGS=(full_s0 full_s1 noctx_s0 hardtarget_s1)
CTX=(1 1 0 1)
I=${SLURM_ARRAY_TASK_ID:-0}; T=${TAGS[$I]}
[ "${CTX[$I]}" = "1" ] && export MASIF_GLOBAL_CTX=1 || unset MASIF_GLOBAL_CTX
/work/upthomae/Meng/conda_envs/masif-graph/bin/python -u -m masif_graph.p8.eval_r \
  --data /work/upthomae/Meng/phase6C/npz_eval --ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt /work/upthomae/Meng/phase8R/ckpt/r_${T}.pt --tag "$T" \
  --out logs/phase8R/gate/gate_${T}.json --max-n 120
echo "=== GATE DONE $T $(date) ==="
