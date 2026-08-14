#!/bin/bash
# Stage R diagnostic — CAN THE MODEL FIT AT ALL? Train on 20 complexes, evaluate on the SAME 20,
# with the distogram, dustbin and chain terms switched OFF so only the soft-target matching
# objective is under test. If this cannot reach <5 A on data it sees 60 times, the limit is
# representability (R1 insufficient), not data or generalisation.
#SBATCH --job-name=p8r_of
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8R/overfit_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8 OMP_NUM_THREADS=8 MASIF_GLOBAL_CTX=1
cd /scratch/ymeng/masif-graph
export PYTHONPATH=/scratch/ymeng/masif-graph/src
/work/upthomae/Meng/conda_envs/masif-graph/bin/python -u -m masif_graph.p8.train_r \
  --data /work/upthomae/Meng/phase7/npz_ppi \
  --train-ids logs/phase8R/overfit20.txt --val-ids logs/phase8R/overfit20.txt \
  --out /work/upthomae/Meng/phase8R/ckpt/r_overfit.pt --log logs/phase8R/hist_overfit.json \
  --epochs 60 --batch 4 --apo-prob 0 --w-disto 0 --w-chain 0 --n-neg-query 0 \
  --eval-every 5 --val-n 20 --seed 0
echo "=== OVERFIT PROBE DONE $(date) ==="
