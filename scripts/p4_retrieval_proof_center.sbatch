#!/bin/bash
#SBATCH --job-name=p4-ret-ctr
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_ret/proofctr-%j.out

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=16
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

W=/work/upthomae/Meng/phase4
echo "[$(date)] retrieval-train PROOF+CENTER: 126 m2, chain objective, DC-offset de-collapse"
python -u -m masif_graph.p4.train_retrieval \
  --data logs/phase4/m2_npz --train-ids logs/phase4/m2_train_ids.txt --val-ids logs/phase4/m2_eval_ids.txt \
  --init-ckpt $W/vicreg_sc_best_seed0.pt \
  --epochs 30 --eval-every 3 --batch 24 --patch dense --w-atom 0.5 \
  --tau-c 0.07 --tau-atom 0.1 --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 \
  --lr 5e-4 --grad-clip 1.0 --cosine --center --seed 0 --device cpu \
  --out logs/phase4/m2_ret/proofctr_result.json --save logs/phase4/m2_ret/proofctr_best.pt
echo "[$(date)] proof DONE"
touch logs/phase4/m2_ret/PROOFCTR_DONE
