#!/bin/bash
# Run af3_sample_to_surface for complexes x a sample index, in parallel. Usage: <idlist> <sample_s> <NP>
set -u
idlist="$1"; S="$2"; NP="${3:-5}"
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search
HERE=/scratch/ymeng/masif-graph
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
run_one(){ id="$1"; s="$2"; pdb=$(echo "$id"|cut -d_ -f1); c1=$(echo "$id"|cut -d_ -f2); c2=$(echo "$id"|cut -d_ -f3)
  dd="$REFDATA/descriptors/sc05/all_feat/${pdb}AS${s}_${c1}_${c2}"
  [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ] && { echo "SKIP $id AS$s"; return 0; }
  bash "$HERE/scripts/af3_sample_to_surface.sh" "$id" "$s"; }
export -f run_one; export REFDATA HERE
grep -E '^[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+$' "$idlist" | xargs -P "$NP" -I{} bash -c 'run_one "$@"' _ {} "$S"
echo "=== sample-$S surf batch done $(date '+%F %T') ==="
