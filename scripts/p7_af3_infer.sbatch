#!/bin/bash
# Phase-7 S5 — AF3 inference on Kuma (GPU). Run AFTER the MSA array.
#SBATCH --job-name=p7-af3-inf
#SBATCH --account=upthomae
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=/work/upthomae/Meng/phase7_af3/logs/inf-%A_%a.out
set -u
SIF=/work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif
CHAINS=${INF_CHAINS:?}; NSAMP=${INF_NSAMP:-1}; PER=${INF_PER:-6}
MSA=/work/upthomae/Meng/phase7_af3/msa
OUT=/work/upthomae/Meng/phase7_af3/models
start=$(( (SLURM_ARRAY_TASK_ID - 1) * PER + 1 ))
for k in $(seq 0 $((PER-1))); do
  idx=$((start + k))
  name=$(sed -n "${idx}p" "$CHAINS"); [ -z "$name" ] && continue
  [ -s "$OUT/$name/${name}_model.cif" ] && { echo "SKIP $name"; continue; }
  data="$MSA/$name/$name/${name}_data.json"
  [ ! -s "$data" ] && { echo "INF_FAIL $name no_msa"; continue; }
  echo "INF $name start $(date '+%T')"; t0=$(date +%s)
  apptainer exec --nv --bind /work:/work "$SIF" \
    /opt/conda/envs/af3/bin/python /AlphaPulldown/alphafold3/run_alphafold.py \
    --json_path="$data" --model_dir=/work/upthomae/Meng/AF3_weights \
    --output_dir="$OUT" --norun_data_pipeline --run_inference --num_diffusion_samples="$NSAMP"
  echo "INF $name done rc=$? in $(( $(date +%s)-t0 ))s"
done
echo "=== INF TASK $SLURM_ARRAY_TASK_ID DONE $(date) ==="
