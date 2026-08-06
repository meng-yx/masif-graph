#!/bin/bash
# Parallel AF3-model -> surface+descriptors for a complex list (needs both sides' AF3 models).
# Usage: p5_af3_surf_batch.sh <idlist> <NP>
set -u
idlist="$1"; NP="${2:-8}"
run_one(){ id="$1"
  PDBID=$(echo "$id"|cut -d_ -f1)
  DD=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/descriptors/sc05/all_feat
  [ -s "$DD/${PDBID}AF_$(echo "$id"|cut -d_ -f2)_$(echo "$id"|cut -d_ -f3)/p1_desc_straight.npy" ] && { echo "SKIP $id"; return 0; }
  bash /scratch/ymeng/masif-graph/scripts/p5_af3_model_to_surface.sh "$id"; }
export -f run_one
grep -E '^[A-Za-z0-9]+_' "$idlist" | xargs -P "$NP" -I{} bash -c 'run_one "$@"' _ {}
echo "=== p5_af3_surf_batch done $(date '+%T') ==="
