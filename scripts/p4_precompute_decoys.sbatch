#!/bin/bash
#SBATCH --job-name=p4-decoy
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/m2_ret_scale/decoy-%A_%a.out

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=4
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph

C=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
CHUNK=logs/phase4/m2_ret_scale/chunk_${C}.txt
echo "[$(date)] decoy precompute chunk $C ($(wc -l < $CHUNK) complexes)"
python -u -m masif_graph.p4.precompute --ids $CHUNK \
  --out logs/phase4/m2_ret_scale/npz --state holo
echo "[$(date)] chunk $C DONE"
touch logs/phase4/m2_ret_scale/CHUNK_${C}_DONE
