#!/bin/bash
#SBATCH --job-name=p1-smoke
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/smoke-%j.out

set +e
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:$PATH"

echo "=== HOST/TIME ==="
hostname; date

echo "=== INTERNET (compute node) ==="
curl -sS -m 12 -o /dev/null -w "anthropic api: HTTP %{http_code}\n" https://api.anthropic.com/v1/messages 2>&1
curl -sS -m 12 -o /dev/null -w "rcsb pdb:     HTTP %{http_code}\n" https://files.rcsb.org/download/4ZQK.pdb 2>&1

echo "=== SINGULARITY ==="
which singularity && singularity --version

echo "=== NODE / CLAUDE ==="
which node claude
node --version

echo "=== CLAUDE HEADLESS AUTH+MODEL TEST (Fable 5) ==="
claude -p "Reply with exactly the token SMOKE_OK and nothing else." \
  --model claude-fable-5 --dangerously-skip-permissions --max-turns 1 --effort max 2>&1
echo "claude_exit=$?"

echo "=== DONE ==="
