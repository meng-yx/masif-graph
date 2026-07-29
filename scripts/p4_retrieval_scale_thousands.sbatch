#!/bin/bash
#SBATCH --job-name=p4-ret-1k
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_ret_scale/thousands-%j.out

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=16
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

W=/work/upthomae/Meng/phase4
# thousands-scale: 1500 training-set decoys (SEEN by learned as negatives -> learned may be mildly
# optimistic; FROZEN never trained, so frozen's degradation curve is clean). Run BOTH patch protocols.
for PK in pos_sc pos; do
  echo "[$(date)] thousands-scale pos_key=$PK"
  python -u scripts/p4_retrieval_scale.py \
    --query-data logs/phase4/m2_npz --query-ids $W/m2_eval_ids.txt \
    --decoy-data $W/stageA_full_npz --decoy-ids logs/phase4/m2_ret_scale/train_decoy_1500.txt \
    --ckpt $W/ret_full_ctr_best.pt --center --pos-key $PK --min-pos 8 --min-pos-decoy 3 \
    --out logs/phase4/m2_ret_scale/thousands_${PK}.json --device cpu
done
echo "[$(date)] thousands DONE"
touch logs/phase4/m2_ret_scale/THOUSANDS_DONE
