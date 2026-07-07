#!/bin/bash
# Parallel holo preprocessing (reference pipeline) for a complex list. Usage: <idlist> <NP>
set -u
idlist="$1"; NP="${2:-8}"
export OMP_NUM_THREADS=2
run_one(){ id="$1"
  DD=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/descriptors/sc05/all_feat
  [ -s "$DD/$id/p1_desc_straight.npy" ] && { echo "SKIP $id"; return 0; }
  bash /scratch/ymeng/masif-graph/scripts/m0_run_one.sh "$id"; }
export -f run_one
grep -E '^[A-Za-z0-9]+_' "$idlist" | xargs -P "$NP" -I{} bash -c 'run_one "$@"' _ {}
echo "=== holo_prep_batch done $(date '+%T') ==="
grep -h M0_STATUS /scratch/ymeng/masif-graph/logs/m0/*.log 2>/dev/null | awk '{print $3}' | sort | uniq -c
