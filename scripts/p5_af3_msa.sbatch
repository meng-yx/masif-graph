#!/bin/bash
#SBATCH --job-name=p5-af3-msa
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase5/msa_arr-%A_%a.out
set -u
CHAINS=${MSA_CHAINS:?set MSA_CHAINS}
IN=/work/upthomae/Meng/phase5_af3/inputs
MSA=/work/upthomae/Meng/phase5_af3/msa
name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CHAINS")
[ -z "$name" ] && { echo "no chain for idx $SLURM_ARRAY_TASK_ID"; exit 0; }
done_json="$MSA/$name/${name}_data.json"
[ -s "$done_json" ] && { echo "SKIP $name (MSA exists)"; exit 0; }
echo "MSA_TASK $name start $(date '+%F %T') host=$(hostname)"
t0=$(date +%s)
bash /work/upthomae/Meng/phase3_af3/run_msa.sh "$IN/${name}.json" "$MSA/$name" 16
rc=$?
echo "MSA_TASK $name done rc=$rc in $(( $(date +%s)-t0 ))s $(date '+%F %T')"
[ -s "$done_json" ] && echo "MSA_OK $name" || echo "MSA_FAIL $name"
