#!/bin/bash
#SBATCH --job-name=p6C_smk_o
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --nodes=1 --cpus-per-task=2 --mem=8G --time=00:15:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/smoke-opus-%j.out
set +e
export HOME=/home/ymeng
export PATH=/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH
cd /scratch/ymeng/masif-graph
echo "host=$(hostname)"
echo "== opus-5 smoke =="
claude -p "Reply with exactly the token OPUS_OK_4321 and nothing else." \
  --model claude-opus-5 --effort max --dangerously-skip-permissions --max-turns 1 2>&1 | tail -3
echo "== rc=$? =="
