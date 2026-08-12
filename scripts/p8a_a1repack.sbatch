#!/bin/bash
# Phase-8 A1.2 — repack sensitivity, one array task per seed.
#SBATCH --job-name=p8a_a1rp
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a1/a1rp_%A_%a.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
CKPTS=(/work/upthomae/Meng/phase6C/ret_ppionly_best.pt /work/upthomae/Meng/phase7/ret_ppionly_s1_best.pt)
TAGS=(ppionly_s0 ppionly_s1)
I=${SLURM_ARRAY_TASK_ID:-0}
$PY -u -m masif_graph.p8.a1_repack --data /work/upthomae/Meng/phase6C/npz_eval \
  --ids logs/phase8A/a1/repack_ok.txt --ckpt "${CKPTS[$I]}" --tag "${TAGS[$I]}" \
  --out-dir logs/phase8A/a1
echo "=== A1.2 DONE ${TAGS[$I]} $(date) ==="
