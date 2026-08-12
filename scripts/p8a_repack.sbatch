#!/bin/bash
# Phase-8 A1.2 — FASPR fixed-backbone repack (each chain in isolation) + reference surfaces,
# for 100 eval complexes. This is the CONFORMATIONAL test A1 does not cover: A1 perturbs sidechain
# identity, a repack keeps identity and changes only the rotamer, which is what an apo structure
# actually does. Reuses the Phase-2 machinery unchanged (scripts/repack_one.sh).
#SBATCH --job-name=p8a_repack
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase8A/a1/repack_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
cd /scratch/ymeng/masif-graph
bash scripts/repack_batch.sh logs/phase8A/a1/repack_ids.txt 16
echo "=== REPACK BATCH DONE $(date) ==="
grep -h "REPACK_STATUS" logs/p2_repack/*.log 2>/dev/null | awk '{print $3}' | sort | uniq -c
