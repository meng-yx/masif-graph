#!/bin/bash
# Phase-4 reference-preprocessing batch (Jed compute node — NOT the login node; the login node
# OOM-kills the TF descriptor net, rc=137). Runs the MaSIF `.sif` pipeline (data_prepare_one +
# compute_descriptors) over a complex-id list, P workers in parallel, resumable (holo_prep_batch.sh
# SKIPs complexes whose descriptors already exist). CPU-only → ~CHF 0 (no Kuma GPU).
# Usage: sbatch phase4_preproc.sbatch <idlist> <NP>
#SBATCH --job-name=p4_preproc
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4/preproc_%j.out

set -u
IDLIST="${1:-/scratch/ymeng/masif-graph/logs/phase4/preproc_chunk1.txt}"
NP="${2:-16}"
echo "host=$(hostname) date=$(date) idlist=$IDLIST NP=$NP n_ids=$(grep -cE '^[A-Za-z0-9]+_' "$IDLIST")"
cd /scratch/ymeng/masif-graph
bash scripts/holo_prep_batch.sh "$IDLIST" "$NP"
echo "=== p4_preproc DONE $(date) ==="
# report how many of the list now have descriptors
DD=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/descriptors/sc05/all_feat
ok=0; n=0
while read -r id; do [ -z "$id" ] && continue; n=$((n+1)); [ -s "$DD/$id/p1_desc_straight.npy" ] && ok=$((ok+1)); done < "$IDLIST"
echo "RESULT: $ok/$n complexes have descriptors"
