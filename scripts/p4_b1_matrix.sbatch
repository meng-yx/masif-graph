#!/bin/bash
#SBATCH --job-name=p4-b1
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_b1/b1-%A_%a.out

# B.1 Stage-B (M2) invariance/ablation matrix: graph {full,no-atom-graph} x {1conf,2conf} x seed {0,1}.
# Fine-tune the stabilized holo-only Stage-A checkpoint with conformer-augmented queries; eval AF3->holo
# on the 30 held-out m1_eval complexes (disjoint from training by PDB stem). Fixed anti-collapse recipe.

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=8
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

W=/work/upthomae/Meng/phase4
OUT=logs/phase4/m2_b1
mkdir -p $OUT

# config table indexed by SLURM_ARRAY_TASK_ID
GRAPHS=(full full nograph nograph full full nograph nograph)
CONFS=(1 1 1 1 2 2 2 2)
SEEDS=(0 1 0 1 0 1 0 1)
i=$SLURM_ARRAY_TASK_ID
G=${GRAPHS[$i]}; CF=${CONFS[$i]}; S=${SEEDS[$i]}
TAG="${G}_${CF}conf_seed${S}"

GRAPH_FLAG=""; [ "$G" = "nograph" ] && GRAPH_FLAG="--no-atom-graph"
CONF_FLAG="";  [ "$CF" = "2" ] && CONF_FLAG="--two-conformer"

echo "[$(date)] B.1 task $i: $TAG (init sc_seed$S)"
python -u -m masif_graph.p4.train_stageb \
  --data logs/phase4/m2_npz \
  --train-ids logs/phase4/m2_train_ids.txt \
  --val-ids logs/phase4/m2_eval_ids.txt \
  --init-ckpt $W/vicreg_sc_best_seed${S}.pt \
  --epochs 40 --eval-every 5 --eval-seeds 3 \
  --train-pos sc --af3-prob 0.5 $CONF_FLAG $GRAPH_FLAG \
  --vicreg-var 2.0 --vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 \
  --lr 5e-4 --grad-clip 1.0 --cosine --bank 128 --min-retention 0.5 --min-pos 8 \
  --seed $S --device cpu \
  --out $OUT/b1_${TAG}.json --save $OUT/b1_${TAG}_best.pt
echo "[$(date)] B.1 task $i DONE ($TAG)"
touch $OUT/B1_TASK_${i}_DONE
