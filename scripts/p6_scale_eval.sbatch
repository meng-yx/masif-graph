#!/bin/bash
#SBATCH --job-name=p6_eval
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1 --cpus-per-task=16 --mem=64G --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6/eval_%j.out
set -u
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph; export PYTHONPATH=src OMP_NUM_THREADS=16
IDS=logs/phase5/eval_sc304_clean_vs_enc.txt
for tag in 600_s0 600_s1 1500_s0 1500_s1 3000_s0 3000_s1; do
  ck=/work/upthomae/Meng/phase6/ret_${tag}_best.pt
  [ ! -s "$ck" ] && { echo "MISSING $ck"; continue; }
  echo "===== gate $tag ====="
  $PY -u -m masif_graph.p5.retrieval_bench --data /work/upthomae/Meng/phase5/npz --ids "$IDS" \
     --ckpt "$ck" --center --pos-key pos --max-patch 128 --out logs/phase6/gate_scale_${tag}_pos.json
done
echo "P6_SCALE_EVAL_DONE"
