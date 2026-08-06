#!/bin/bash
#SBATCH --job-name=p5_gate
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase5/gate_%j.out
set -u
CKPT="${1:?ckpt}"; TAG="${2:?tag}"; NOAA="${3:-0}"; IDS="${4:-logs/phase5/eval_sc304.txt}"
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
cd /scratch/ymeng/masif-graph
export PYTHONPATH=src OMP_NUM_THREADS=16
[ "$NOAA" = "1" ] && export MASIF_NO_AA=1
echo "host=$(hostname) ckpt=$CKPT tag=$TAG NO_AA=${MASIF_NO_AA:-0} $(date)"
for patch in pos_sc pos; do
  echo "===== GATE $TAG $patch ====="
  $PY -u -m masif_graph.p5.retrieval_bench --data /work/upthomae/Meng/phase5/npz \
    --ids "$IDS" --ckpt "$CKPT" --center --pos-key $patch --max-patch 128 \
    --out logs/phase5/gate_${TAG}_${patch}.json
done
echo "P5_GATE_${TAG}_DONE $(date)"
