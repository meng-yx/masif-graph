#!/bin/bash
# Repack a list of complex ids in parallel (N concurrent). Skips ids whose repacked
# descriptors already exist. Usage: repack_batch.sh <idlist_file> <n_parallel>
set -u
idlist="$1"
NP="${2:-6}"
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search
HERE=/scratch/ymeng/masif-graph

run_one() {
  id="$1"
  pdb=$(echo "$id" | cut -d_ -f1); c1=$(echo "$id" | cut -d_ -f2); c2=$(echo "$id" | cut -d_ -f3)
  dd="$REFDATA/descriptors/sc05/all_feat/${pdb}RP_${c1}_${c2}"
  if [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ]; then
    echo "SKIP $id (already repacked)"; return 0
  fi
  bash "$HERE/scripts/repack_one.sh" "$id"
}
export -f run_one
export REFDATA HERE

# only single-character chain ids (avoid multi-char extraction edge cases)
grep -E '^[A-Za-z0-9]+_[A-Za-z0-9]_[A-Za-z0-9]$' "$idlist" | \
  xargs -P "$NP" -I{} bash -c 'run_one "$@"' _ {}
echo "=== repack_batch done $(date '+%F %T') ==="
