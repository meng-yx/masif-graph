#!/bin/bash
#SBATCH --job-name=p6C_smoke
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --nodes=1 --cpus-per-task=2 --mem=8G --time=00:20:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/smoke-%j.out
set +e
export HOME=/home/ymeng
export PATH=/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH
cd /scratch/ymeng/masif-graph
echo "host=$(hostname) date=$(date)"
echo "== internet (api.anthropic.com) =="; curl -sI --max-time 15 https://api.anthropic.com | head -1
echo "== singularity =="; command -v singularity apptainer | head -1
echo "== claude path =="; command -v claude
echo "== claude smoke (fable-5) =="
claude -p "Reply with exactly the token SMOKE_OK_7788 and nothing else." \
  --model claude-fable-5 --dangerously-skip-permissions --max-turns 1 2>&1 | tail -3
echo "== rc=$? =="
